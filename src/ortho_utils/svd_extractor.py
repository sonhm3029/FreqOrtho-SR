"""
SVD Extractor for Pixel LoRA Subspace

For each LoRA layer:
1. Get lora_A and lora_B weights
2. Compute effective weight change: ΔW = B @ A (handling Linear vs Conv2d)
3. Flatten ΔW to 2D matrix for SVD
4. Perform SVD: U, S, Vt = svd(ΔW_flat)
5. Select top-k singular vectors where cumsum(S^2) >= threshold (e.g., 95%)
6. Store U_k and Vt_k for gradient projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, NamedTuple
import logging

logger = logging.getLogger("train_logger")


class SubspaceInfo(NamedTuple):
    """Store SVD subspace information for a LoRA layer."""
    U_k: torch.Tensor      # Left singular vectors (out_features, k)
    Vt_k: torch.Tensor     # Right singular vectors (k, in_features)
    original_shape: tuple  # Original ΔW shape before flattening
    layer_type: str        # 'linear' or 'conv2d'
    k: int                 # Number of singular vectors retained


def _single_delta_weight(lora_B_module, lora_A_module) -> Tuple[torch.Tensor, str]:
    """
    Compute ΔW = B @ A for a single expert pair.

    Returns:
        delta_W: 2D tensor [out_features, in_features] (flattened for conv)
        layer_type: 'linear' or 'conv2d'
    """
    if isinstance(lora_A_module, nn.Linear):
        A = lora_A_module.weight.data  # (r, in_features)
        B = lora_B_module.weight.data  # (out_features, r)
        delta_W = B @ A                # (out_features, in_features)
        return delta_W, 'linear'

    elif isinstance(lora_A_module, nn.Conv2d):
        weight_A = lora_A_module.weight.data  # (r, in_ch, kH, kW)
        weight_B = lora_B_module.weight.data  # (out_ch, r, 1, 1)

        if weight_A.shape[2:] == (1, 1):
            A_squeezed = weight_A.squeeze(3).squeeze(2)  # (r, in_ch)
            B_squeezed = weight_B.squeeze(3).squeeze(2)  # (out_ch, r)
            delta_W = B_squeezed @ A_squeezed            # (out_ch, in_ch)
        else:
            delta_W = F.conv2d(
                weight_A.permute(1, 0, 2, 3),  # (in_ch, r, kH, kW)
                weight_B,                       # (out_ch, r, 1, 1)
            ).permute(1, 0, 2, 3)              # (out_ch, in_ch, kH, kW)
            # Flatten to 2D: [out_ch, in_ch * kH * kW]
            out_ch = delta_W.shape[0]
            delta_W = delta_W.reshape(out_ch, -1)

        return delta_W, 'conv2d'

    else:
        raise ValueError(f"Unsupported LoRA layer type: {type(lora_A_module)}")


def compute_delta_weight(module, adapter_name: str) -> Tuple[torch.Tensor, str]:
    """
    Compute ΔW = B @ A for a LoRA layer.

    Handles both:
    - Single LoRA: lora_A/lora_B are nn.Linear or nn.Conv2d
    - MoLE: lora_A/lora_B are nn.ModuleList of experts

    For MoLE layers, concatenates all expert ΔWs horizontally
    [out_features, N * in_features] to capture the union of all expert subspaces.
    This preserves the output dimension for lora_B gradient projection.

    Returns:
        delta_W: 2D tensor [out_features, total_in_features]
        layer_type: 'linear' or 'conv2d'
    """
    lora_A = module.lora_A[adapter_name]
    lora_B = module.lora_B[adapter_name]

    if isinstance(lora_A, nn.ModuleList):
        # MoLE: multiple experts — compute ΔW for each and concatenate horizontally
        delta_ws = []
        layer_type = None
        for expert_A, expert_B in zip(lora_A, lora_B):
            dw, lt = _single_delta_weight(expert_B, expert_A)
            delta_ws.append(dw)
            layer_type = lt

        # Include shared experts if present
        if (hasattr(module, 'lora_shared_A') and
            adapter_name in getattr(module, 'lora_shared_A', {})):
            shared_A = module.lora_shared_A[adapter_name]
            shared_B = module.lora_shared_B[adapter_name]
            if isinstance(shared_A, nn.ModuleList):
                for sa, sb in zip(shared_A, shared_B):
                    dw, lt = _single_delta_weight(sb, sa)
                    delta_ws.append(dw)
            else:
                dw, lt = _single_delta_weight(shared_B, shared_A)
                delta_ws.append(dw)

        # Horizontal concatenation: [out_features, N * in_features]
        return torch.cat(delta_ws, dim=1), layer_type
    else:
        # Single LoRA
        return _single_delta_weight(lora_B, lora_A)


def extract_lora_subspace(
    unet,
    adapter_names: list = ['default_encoder_pix', 'default_decoder_pix', 'default_others_pix'],
    energy_threshold: float = 0.95,
    min_rank: int = 1,
) -> Dict[str, SubspaceInfo]:
    """
    Extract orthonormal subspace from trained Pixel LoRA layers.

    Args:
        unet: The UNet model with LoRA adapters
        adapter_names: List of pixel adapter names to extract from
        energy_threshold: Fraction of energy to retain (e.g., 0.95 = 95%)
        min_rank: Minimum number of singular vectors to keep

    Returns:
        Dict mapping layer names to SubspaceInfo objects
    """
    subspaces = {}

    for name, module in unet.named_modules():
        # Check if this module has LoRA layers
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            continue

        # Process each pixel adapter
        for adapter_name in adapter_names:
            if adapter_name not in module.lora_A:
                continue

            # Compute ΔW (always 2D: [out_features, total_in_features])
            delta_W, layer_type = compute_delta_weight(module, adapter_name)
            original_shape = delta_W.shape

            # SVD decomposition
            # U: (out, min(out,in)), S: (min(out,in),), Vt: (min(out,in), in)
            U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)

            # Compute energy and select top-k
            energy = S ** 2
            total_energy = energy.sum()

            if total_energy > 1e-10:  # Add small epsilon for numerical stability
                cumsum_energy = torch.cumsum(energy, dim=0)
                k = torch.searchsorted(cumsum_energy, energy_threshold * total_energy).item() + 1
                k = max(k, min_rank)
                k = min(k, len(S))
                energy_retained = cumsum_energy[k-1] / total_energy
            else:
                k = min_rank
                energy_retained = 0.0

            # Store subspace info
            U_k = U[:, :k].clone().contiguous()
            Vt_k = Vt[:k, :].clone().contiguous()

            layer_key = f"{name}.{adapter_name}"
            subspaces[layer_key] = SubspaceInfo(
                U_k=U_k,
                Vt_k=Vt_k,
                original_shape=original_shape,
                layer_type=layer_type,
                k=k,
            )

            logger.info(f"  {layer_key} [{layer_type}]: shape={original_shape}, "
                       f"rank={len(S)}, k={k}, energy={energy_retained:.4f}")

    logger.info(f"Extracted subspaces from {len(subspaces)} LoRA layers")
    return subspaces


def save_subspaces(subspaces: Dict[str, SubspaceInfo], path: str):
    """Save extracted subspaces to file."""
    # Convert NamedTuple to dict for saving
    save_dict = {}
    for key, info in subspaces.items():
        save_dict[key] = {
            'U_k': info.U_k,
            'Vt_k': info.Vt_k,
            'original_shape': info.original_shape,
            'layer_type': info.layer_type,
            'k': info.k,
        }
    torch.save(save_dict, path)
    logger.info(f"Saved subspaces to {path}")


def load_subspaces(path: str) -> Dict[str, SubspaceInfo]:
    """Load subspaces from file."""
    save_dict = torch.load(path)
    subspaces = {}
    for key, info_dict in save_dict.items():
        subspaces[key] = SubspaceInfo(
            U_k=info_dict['U_k'],
            Vt_k=info_dict['Vt_k'],
            original_shape=info_dict['original_shape'],
            layer_type=info_dict['layer_type'],
            k=info_dict['k'],
        )
    logger.info(f"Loaded subspaces from {path} ({len(subspaces)} layers)")
    return subspaces
