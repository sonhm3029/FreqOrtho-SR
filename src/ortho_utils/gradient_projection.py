"""
Orthogonal Gradient Projection for Semantic LoRA

The key insight from GPM is to project gradients onto the null space of
the subspace learned by previous tasks (Pixel LoRA in our case).

For LoRA, we have: ΔW = B @ A
- B captures the output space transformation
- A captures the input space transformation

We project the combined gradient of ΔW (reconstructed from A and B gradients)
onto the null space, then distribute back to A and B.

Simplified approach: Project lora_B gradient only, since it directly
corresponds to the output (row) space of ΔW where U_k lives.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable
import logging

from .svd_extractor import SubspaceInfo

logger = logging.getLogger("train_logger")


class OrthogonalGradientProjector:
    """
    Manages gradient projection hooks for Semantic LoRA layers.

    Projects gradients of lora_B onto null space of Pixel LoRA's U_k.
    This ensures Semantic LoRA updates are orthogonal to Pixel LoRA's
    learned output space directions.
    """

    def __init__(
        self,
        subspaces: Dict[str, SubspaceInfo],
        semantic_adapter_names: list = ['default_encoder_sem', 'default_decoder_sem', 'default_others_sem'],
    ):
        """
        Args:
            subspaces: Dict from SVD extractor {layer_key: SubspaceInfo}
            semantic_adapter_names: Names of semantic adapters to apply projection
        """
        self.subspaces = subspaces
        self.semantic_adapter_names = semantic_adapter_names
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Build mapping from pixel adapter names to semantic adapter names
        self.adapter_mapping = {
            'default_encoder_pix': 'default_encoder_sem',
            'default_decoder_pix': 'default_decoder_sem',
            'default_others_pix': 'default_others_sem',
        }

    def _create_projection_hook_linear(
        self,
        U_k: torch.Tensor,
        layer_name: str,
    ) -> Callable:
        """
        Create gradient projection hook for Linear lora_B.

        For Linear lora_B with weight shape (out_features, r):
        - U_k shape: (out_features, k)
        - Project each column of grad onto null space of U_k

        g_ortho = g - U_k @ (U_k.T @ g)
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad

            # grad shape: (out_features, r)
            # U_k shape: (out_features, k)
            U = U_k.to(grad.device, dtype=grad.dtype)

            # Project: g_ortho = g - U @ (U.T @ g)
            # U.T @ grad: (k, r)
            # U @ (U.T @ grad): (out_features, r)
            projection = U @ (U.T @ grad)
            grad_ortho = grad - projection

            return grad_ortho

        return hook

    def _create_projection_hook_conv2d(
        self,
        U_k: torch.Tensor,
        original_shape: tuple,
        layer_name: str,
    ) -> Callable:
        """
        Create gradient projection hook for Conv2d lora_B.

        For Conv2d lora_B with weight shape (out_ch, r, 1, 1):
        - U_k shape: (out_ch, k) - from flattened ΔW
        - Squeeze grad to (out_ch, r), project, then unsqueeze

        g_ortho = g - U_k @ (U_k.T @ g)
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad

            # grad shape: (out_ch, r, 1, 1)
            U = U_k.to(grad.device, dtype=grad.dtype)

            # Squeeze to 2D
            grad_squeezed = grad.squeeze(3).squeeze(2)  # (out_ch, r)

            # Project
            projection = U @ (U.T @ grad_squeezed)
            grad_ortho = grad_squeezed - projection

            # Unsqueeze back to 4D
            grad_ortho = grad_ortho.unsqueeze(2).unsqueeze(3)  # (out_ch, r, 1, 1)

            return grad_ortho

        return hook

    def register_hooks(self, unet) -> int:
        """
        Register gradient projection hooks on Semantic LoRA lora_B layers.

        Args:
            unet: The UNet model with LoRA adapters

        Returns:
            Number of hooks registered
        """
        hooks_registered = 0

        for name, module in unet.named_modules():
            if not hasattr(module, 'lora_B'):
                continue

            for sem_adapter in self.semantic_adapter_names:
                if sem_adapter not in module.lora_B:
                    continue

                # Find corresponding pixel adapter subspace
                pix_adapter = None
                for pix, sem in self.adapter_mapping.items():
                    if sem == sem_adapter:
                        pix_adapter = pix
                        break

                if pix_adapter is None:
                    continue

                # Look up the subspace info for this layer
                layer_key = f"{name}.{pix_adapter}"
                if layer_key not in self.subspaces:
                    logger.warning(f"No subspace found for {layer_key}, skipping")
                    continue

                subspace_info = self.subspaces[layer_key]

                # Get lora_B parameter
                lora_B = module.lora_B[sem_adapter]
                if hasattr(lora_B, 'weight'):
                    param = lora_B.weight
                else:
                    param = lora_B

                # Create appropriate hook based on layer type
                if subspace_info.layer_type == 'linear':
                    hook_fn = self._create_projection_hook_linear(
                        subspace_info.U_k,
                        f"{name}.{sem_adapter}"
                    )
                else:  # conv2d
                    hook_fn = self._create_projection_hook_conv2d(
                        subspace_info.U_k,
                        subspace_info.original_shape,
                        f"{name}.{sem_adapter}"
                    )

                hook = param.register_hook(hook_fn)
                self.hooks.append(hook)
                hooks_registered += 1

                logger.info(f"  Registered hook on {name}.{sem_adapter}.lora_B "
                           f"[{subspace_info.layer_type}, k={subspace_info.k}]")

        logger.info(f"Registered {hooks_registered} gradient projection hooks")
        return hooks_registered

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("Removed all gradient projection hooks")


def register_ortho_hooks(
    unet,
    subspaces: Dict[str, SubspaceInfo],
    semantic_adapter_names: list = ['default_encoder_sem', 'default_decoder_sem', 'default_others_sem'],
) -> OrthogonalGradientProjector:
    """
    Convenience function to create projector and register hooks.

    Args:
        unet: The UNet model
        subspaces: Dict from SVD extractor
        semantic_adapter_names: Names of semantic adapters

    Returns:
        OrthogonalGradientProjector instance (keep reference to remove hooks later)
    """
    projector = OrthogonalGradientProjector(subspaces, semantic_adapter_names)
    projector.register_hooks(unet)
    return projector
