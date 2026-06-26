"""
Orthogonal Gradient Projection utilities for Ortho-PiSA.

Based on Gradient Projection Memory (GPM) from Continual Learning.
"""

from .svd_extractor import (
    extract_lora_subspace,
    save_subspaces,
    load_subspaces,
)

from .gradient_projection import (
    OrthogonalGradientProjector,
    register_ortho_hooks,
)

__all__ = [
    'extract_lora_subspace',
    'save_subspaces',
    'load_subspaces',
    'OrthogonalGradientProjector',
    'register_ortho_hooks',
]
