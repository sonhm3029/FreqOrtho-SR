"""
Degradation Feature Extractor for Frequency-Based Routing.

This module extracts frequency domain features from input images to help
the MoLE router better select experts for different degradation types.

Each degradation type has a distinct fingerprint in frequency domain:
- Blur: Loss of high frequency, dominant low frequency
- Gaussian Noise: Elevated high frequency energy (random)
- Poisson Noise: Signal-dependent high frequency
- JPEG Compression: 8x8 block artifacts, loss of mid-high frequency
- Downscale + Upscale: Aliasing patterns, ringing artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationFeatureExtractor(nn.Module):
    """
    Comprehensive degradation feature extraction combining:
    - Frequency band energies (radial bands in FFT domain)
    - Blur score (high-to-low frequency ratio)
    - JPEG blocking artifacts score
    - Noise score (local variance via Laplacian)

    Args:
        num_bands: Number of radial frequency bands to extract (default: 4)

    Output dimension: num_bands + 3 (blur, jpeg, noise scores)
    """

    def __init__(self, num_bands: int = 4):
        super().__init__()
        self.num_bands = num_bands
        # Output dimension: num_bands + 3 (blur_score, jpeg_score, noise_score)
        self.output_dim = num_bands + 3

        # Pre-register Laplacian kernel for noise detection
        # This is a 3x3 Laplacian kernel for edge/noise detection
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', laplacian.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract degradation features from input image.

        Args:
            x: [B, C, H, W] input image tensor (normalized to [-1, 1] or [0, 1])

        Returns:
            features: [B, output_dim] degradation features
                - [:, :num_bands]: Normalized frequency band energies
                - [:, num_bands]: Blur score (low/high frequency ratio)
                - [:, num_bands+1]: JPEG blocking score
                - [:, num_bands+2]: Noise score
        """
        B, C, H, W = x.shape

        # 1. Extract frequency band energies
        freq_features = self._extract_freq_bands(x)  # [B, num_bands]

        # 2. Compute blur score: ratio of low to high frequency
        # High blur_score = more blur (low freq dominates)
        blur_score = freq_features[:, 0] / (freq_features[:, -1] + 1e-8)
        blur_score = blur_score.unsqueeze(1)  # [B, 1]

        # 3. Extract JPEG blocking score
        jpeg_score = self._extract_jpeg_score(x)  # [B, 1]

        # 4. Extract noise score
        noise_score = self._extract_noise_score(x)  # [B, 1]

        # Concatenate all features
        features = torch.cat([
            freq_features,   # [B, num_bands]
            blur_score,      # [B, 1]
            jpeg_score,      # [B, 1]
            noise_score      # [B, 1]
        ], dim=1)

        return features

    def _extract_freq_bands(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract radial frequency band energies using 2D FFT.

        The frequency spectrum is divided into num_bands concentric rings
        from low frequency (center) to high frequency (edges).

        Args:
            x: [B, C, H, W] input image

        Returns:
            freq_features: [B, num_bands] normalized energy per band
        """
        B, C, H, W = x.shape

        # Convert to grayscale for frequency analysis
        # Using standard luminance weights
        if C == 3:
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        else:
            gray = x[:, 0]

        # 2D FFT
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)  # Center low frequencies
        magnitude = torch.abs(fft_shift)      # [B, H, W]

        # Create radial distance map from center
        cy, cx = H // 2, W // 2
        y_coords = torch.arange(H, device=x.device, dtype=x.dtype).view(-1, 1) - cy
        x_coords = torch.arange(W, device=x.device, dtype=x.dtype).view(1, -1) - cx
        radius = torch.sqrt(y_coords ** 2 + x_coords ** 2)  # [H, W]
        max_radius = min(cy, cx)

        # Extract energy per radial band
        band_energies = []
        for i in range(self.num_bands):
            r_min = i * max_radius / self.num_bands
            r_max = (i + 1) * max_radius / self.num_bands
            mask = (radius >= r_min) & (radius < r_max)  # [H, W]
            mask_float = mask.float()

            # Compute mean energy in this band for each sample
            # magnitude: [B, H, W], mask_float: [H, W]
            masked_magnitude = magnitude * mask_float.unsqueeze(0)  # [B, H, W]
            energy = masked_magnitude.sum(dim=(-2, -1)) / (mask_float.sum() + 1e-8)  # [B]
            band_energies.append(energy)

        # Stack and normalize to get relative distribution
        freq_features = torch.stack(band_energies, dim=1)  # [B, num_bands]
        freq_features = freq_features / (freq_features.sum(dim=1, keepdim=True) + 1e-8)

        return freq_features

    def _extract_jpeg_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect JPEG 8x8 block artifacts.

        JPEG compression creates visible boundaries at 8x8 block edges.
        We detect this by comparing gradient magnitude at block boundaries
        vs overall gradient magnitude.

        High score = more blocking artifacts = likely JPEG compressed

        Args:
            x: [B, C, H, W] input image

        Returns:
            jpeg_score: [B, 1] blocking artifact score
        """
        B, C, H, W = x.shape

        # Compute gradients
        grad_h = x[:, :, 1:, :] - x[:, :, :-1, :]  # [B, C, H-1, W]
        grad_w = x[:, :, :, 1:] - x[:, :, :, :-1]  # [B, C, H, W-1]

        # Gradient magnitude at 8-pixel block boundaries
        # Block boundaries occur at positions 7, 15, 23, ... (every 8th pixel - 1)
        block_pos_h = list(range(7, H-1, 8))
        block_pos_w = list(range(7, W-1, 8))

        if len(block_pos_h) == 0 or len(block_pos_w) == 0:
            return torch.zeros(B, 1, device=x.device, dtype=x.dtype)

        # Average gradient at block boundaries
        block_grad_h = grad_h[:, :, block_pos_h, :].abs().mean(dim=(1, 2, 3))  # [B]
        block_grad_w = grad_w[:, :, :, block_pos_w].abs().mean(dim=(1, 2, 3))  # [B]

        # Overall average gradient
        overall_grad = (grad_h.abs().mean(dim=(1, 2, 3)) + grad_w.abs().mean(dim=(1, 2, 3))) / 2  # [B]

        # Ratio: elevated block boundary gradients indicate JPEG artifacts
        # If block boundary gradients are higher than average, it suggests blocking
        jpeg_score = (block_grad_h + block_grad_w) / (2 * overall_grad + 1e-8)

        return jpeg_score.unsqueeze(1)

    def _extract_noise_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate noise level via Laplacian response.

        The Laplacian operator is sensitive to rapid intensity changes (noise).
        High variance of Laplacian response indicates high noise level.

        Args:
            x: [B, C, H, W] input image

        Returns:
            noise_score: [B, 1] estimated noise level
        """
        B, C, H, W = x.shape

        # Expand Laplacian kernel for all channels
        laplacian = self.laplacian_kernel.expand(C, 1, 3, 3).to(x.dtype)

        # Apply Laplacian filter (groups=C for depthwise conv)
        edges = F.conv2d(x, laplacian, padding=1, groups=C)  # [B, C, H, W]

        # Estimate noise as mean absolute Laplacian response
        # This is a robust noise estimator (similar to MAD-based estimators)
        noise_estimate = edges.abs().mean(dim=(1, 2, 3))  # [B]

        return noise_estimate.unsqueeze(1)


class FrequencyFeatureExtractor(nn.Module):
    """
    Simplified frequency-only feature extractor.

    Extracts only frequency band energies without additional scores.
    Use this for lighter-weight routing if full degradation features aren't needed.

    Args:
        num_bands: Number of radial frequency bands (default: 4)
    """

    def __init__(self, num_bands: int = 4):
        super().__init__()
        self.num_bands = num_bands
        self.output_dim = num_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency band energies.

        Args:
            x: [B, C, H, W] image tensor

        Returns:
            freq_features: [B, num_bands] normalized frequency band energies
        """
        B, C, H, W = x.shape

        # Convert to grayscale
        if C == 3:
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        else:
            gray = x[:, 0]

        # 2D FFT
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)

        # Create radial distance map
        cy, cx = H // 2, W // 2
        y_coords = torch.arange(H, device=x.device, dtype=x.dtype).view(-1, 1) - cy
        x_coords = torch.arange(W, device=x.device, dtype=x.dtype).view(1, -1) - cx
        radius = torch.sqrt(y_coords ** 2 + x_coords ** 2)
        max_radius = min(cy, cx)

        # Extract energy per band
        band_energies = []
        for i in range(self.num_bands):
            r_min = i * max_radius / self.num_bands
            r_max = (i + 1) * max_radius / self.num_bands
            mask = (radius >= r_min) & (radius < r_max)
            energy = (magnitude * mask.float()).sum(dim=(-2, -1)) / (mask.sum() + 1e-8)
            band_energies.append(energy)

        freq_features = torch.stack(band_energies, dim=1)
        freq_features = freq_features / (freq_features.sum(dim=1, keepdim=True) + 1e-8)

        return freq_features
