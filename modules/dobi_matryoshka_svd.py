"""
Dobi-Matryoshka SVD Layer

Combines the proven Dobi-SVD architecture with Matryoshka multi-scale training.

Key features:
- Dynamic SVD on activations (from Dobi-SVD)
- Learnable rank gamma (from Dobi-SVD)
- Multi-scale training support (from Matryoshka)
- Custom stable gradients (from Dobi-SVD)

Author: Claude (based on Dobi-SVD)
Date: 2025-12-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from modules.stable_svd import stable_lowrank_SVD

# Dobi-SVD configurations
computeSVD_dtype = torch.float32
model_load_dtype = torch.float16


class DobiMatryoshkaSVDLayer(nn.Module):
    """
    Hybrid layer combining Dobi-SVD's dynamic approach with Matryoshka's multi-scale training.

    Architecture (from Dobi-SVD):
    - Preserves original weight matrix (trainable)
    - Performs SVD on activations during forward pass
    - Uses learnable rank parameter gamma
    - Applies soft truncation with tanh

    Matryoshka extension:
    - Supports fixed_rank for multi-scale training
    - Enables nested rank structure (r_min to r_max)
    - Single model works at multiple compression levels
    """

    def __init__(
        self,
        gamma: float,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        r_max: int = 256,
        r_min: int = 64,
        beta: float = 10.0,
        name: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            gamma: Initial rank (will be learned)
            weight: Original weight matrix
            bias: Original bias vector
            r_max: Maximum rank (for Matryoshka)
            r_min: Minimum rank (for Matryoshka)
            beta: Sharpness of truncation (default 10.0 from Dobi)
            name: Layer name for debugging
            device: Device to place layer on
        """
        super().__init__()

        # === CORE: Learnable rank (from Dobi-SVD) ===
        self.gamma = nn.Parameter(
            torch.tensor(gamma, dtype=computeSVD_dtype, device=device),
            requires_grad=True
        )

        # === CORE: Preserve original weight (from Dobi-SVD) ===
        input_size = weight.size(1)
        output_size = weight.size(0)

        if bias is None:
            self.ori = nn.Linear(input_size, output_size, bias=False).to(device)
        else:
            self.ori = nn.Linear(input_size, output_size, bias=True).to(device)
            self.ori.bias = nn.Parameter(bias.to(device))

        self.ori.weight = nn.Parameter(weight.to(device))

        # === Matryoshka parameters ===
        self.r_max = r_max
        self.r_min = r_min

        # === Dobi parameters ===
        self.beta = beta  # Truncation sharpness
        self.name = name

        # === For multi-scale training ===
        self.fixed_rank_override = None  # Temporarily override gamma

    def forward(self, x: torch.Tensor, fixed_rank: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass with dynamic SVD (from Dobi-SVD).

        Args:
            x: Input tensor [batch, seq, input_size] or [batch*seq, input_size]
            fixed_rank: If specified, use this rank instead of gamma (for multi-scale training)

        Returns:
            Output tensor with same shape as x
        """
        # Store original dtype and shape
        input_dtype = x.dtype
        original_shape = x.shape

        # === STEP 1: Apply original weight (from Dobi-SVD) ===
        # Keep in original dtype (FP16) to avoid dtype mismatch
        x = self.ori(x)

        # === STEP 2: Convert to FP32 for SVD ===
        # CRITICAL: SVD requires FP32
        x = x.to(computeSVD_dtype).contiguous()

        # Handle 3D inputs: [batch, seq, hidden] -> [batch*seq, hidden]
        if x.dim() == 3:
            batch_size, seq_len, hidden_size = x.shape
            x = x.reshape(batch_size * seq_len, hidden_size)
            needs_reshape = True
        elif x.dim() == 2:
            needs_reshape = False
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")

        m, n = x.shape
        full_rank = min(m, n)

        # === STEP 2: Determine rank to use ===
        if fixed_rank is not None:
            # Multi-scale training mode
            real_gamma = fixed_rank
        elif self.fixed_rank_override is not None:
            # Temporary override (alternative multi-scale approach)
            real_gamma = self.fixed_rank_override
        else:
            # Normal mode: use learnable gamma, clamped to [r_min, r_max]
            real_gamma = self.gamma.clamp(self.r_min, self.r_max)

        # === STEP 3: Dynamic SVD (from Dobi-SVD) ===
        # Add buffer for safer SVD computation
        gamma_range = int(real_gamma.detach()) + 5
        gamma_range = min(full_rank, max(1, gamma_range))

        # CRITICAL: Disable autocast for SVD computation
        # autocast forces FP16 but SVD requires FP32
        with torch.cuda.amp.autocast(enabled=False):
            # Ensure x is truly FP32 (autocast might have changed it)
            x = x.to(torch.float32)
            U, S, V = stable_lowrank_SVD.apply(x, gamma_range)

        # === STEP 4: Soft truncation (from Dobi-SVD) ===
        sequence = torch.arange(1, len(S) + 1, device=x.device, dtype=computeSVD_dtype)

        # Tanh-based soft truncation (smoother than sigmoid)
        # Trunc = 0.5 * tanh(beta * (gamma - k)) + 0.5
        # - At k = gamma: Trunc = 0.5
        # - At k < gamma: Trunc → 1
        # - At k > gamma: Trunc → 0
        Trunc = 0.5 * torch.tanh(self.beta * (real_gamma - sequence)) + 0.5
        S_transformed = S * Trunc

        # === STEP 5: Reconstruct ===
        S_diag = torch.diag_embed(S_transformed)
        x_transformed = torch.matmul(torch.matmul(U, S_diag), V.T)

        # Reshape back to original shape if needed
        if needs_reshape:
            x_transformed = x_transformed.reshape(batch_size, seq_len, -1)

        # Convert back to input dtype (usually FP16 for model efficiency)
        output = x_transformed.to(input_dtype)

        return output

    def get_compression_loss(self) -> torch.Tensor:
        """
        Regularization loss to encourage compression (from Dobi-SVD).

        Encourages gamma to decrease, leading to higher compression.

        Returns:
            Scalar loss proportional to gamma
        """
        # Normalize gamma to [0, 1]
        normalized_gamma = (self.gamma - self.r_min) / (self.r_max - self.r_min)
        return normalized_gamma.clamp(0, 1)


def replace_linear_with_dobi_matryoshka(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    r_max: int = 256,
    r_min: int = 64,
    beta: float = 10.0,
    verbose: bool = True
) -> nn.Module:
    """
    Replace Linear layers with DobiMatryoshkaSVDLayer.

    Args:
        model: Model to modify
        target_layers: List of layer name patterns to replace (e.g., ['q_proj', 'v_proj'])
                      If None, replace all linear layers
        r_max: Maximum rank
        r_min: Minimum rank
        beta: Truncation sharpness
        verbose: Print progress

    Returns:
        Modified model
    """
    if verbose:
        print("\n" + "="*80)
        print("Replacing Linear layers with DobiMatryoshkaSVDLayer")
        print("="*80)
        print(f"Target layers: {target_layers}")
        print(f"Rank range: [{r_min}, {r_max}]")
        print(f"Beta (sharpness): {beta}")

    # Collect layers to replace
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer should be replaced
            if target_layers is None:
                should_replace = True
            else:
                should_replace = any(pattern in name for pattern in target_layers)

            if should_replace:
                layers_to_replace.append((name, module))

    if verbose:
        print(f"\nFound {len(layers_to_replace)} layers to replace")

    # Replace layers
    replaced_count = 0
    for name, module in layers_to_replace:
        # Get parent module and attribute name
        *parent_names, attr_name = name.split('.')
        parent = model
        for pname in parent_names:
            parent = getattr(parent, pname)

        # Initialize gamma as midpoint
        gamma_init = (r_max + r_min) / 2.0

        # Create new layer
        new_layer = DobiMatryoshkaSVDLayer(
            gamma=gamma_init,
            weight=module.weight.data.clone().cpu(),
            bias=module.bias.data.clone().cpu() if module.bias is not None else None,
            r_max=r_max,
            r_min=r_min,
            beta=beta,
            name=name,
            device=torch.device('cpu')  # Start on CPU
        )

        # Replace
        setattr(parent, attr_name, new_layer)
        replaced_count += 1

        if verbose and replaced_count % 10 == 0:
            print(f"  Replaced {replaced_count}/{len(layers_to_replace)} layers...")

    if verbose:
        print(f"\n✅ Successfully replaced {replaced_count} layers")

        # Calculate compression stats
        original_params = sum(
            m.in_features * m.out_features + (m.out_features if m.bias is not None else 0)
            for _, m in layers_to_replace
        )
        compressed_params = sum(p.numel() for p in model.parameters())

        print(f"\nOriginal parameters (in replaced layers): {original_params:,}")
        print(f"Model parameters (total): {compressed_params:,}")
        print("(Note: Compression is dynamic based on learned gamma)")

    return model


def compute_multiscale_loss(
    model: nn.Module,
    forward_fn,
    sampled_ranks: List[int],
    *args,
    **kwargs
) -> torch.Tensor:
    """
    Compute loss at multiple ranks for Matryoshka training.

    Args:
        model: Model with DobiMatryoshkaSVDLayer layers
        forward_fn: Function that takes model and returns loss
                   Signature: forward_fn(model, *args, **kwargs) -> loss
        sampled_ranks: List of ranks to train at (e.g., [64, 128, 256])
        *args, **kwargs: Arguments to pass to forward_fn

    Returns:
        Average loss across all sampled ranks
    """
    total_loss = 0

    for rank in sampled_ranks:
        # Set all layers to use this fixed rank
        for module in model.modules():
            if isinstance(module, DobiMatryoshkaSVDLayer):
                module.fixed_rank_override = float(rank)

        # Forward pass
        loss = forward_fn(model, *args, **kwargs)
        total_loss += loss

        # Clear override
        for module in model.modules():
            if isinstance(module, DobiMatryoshkaSVDLayer):
                module.fixed_rank_override = None

    return total_loss / len(sampled_ranks)


def get_compression_regularization(model: nn.Module) -> torch.Tensor:
    """
    Get compression regularization loss from all DobiMatryoshkaSVDLayer layers.

    This encourages the model to learn smaller gamma values (higher compression).

    Args:
        model: Model with DobiMatryoshkaSVDLayer layers

    Returns:
        Total compression loss
    """
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for module in model.modules():
        if isinstance(module, DobiMatryoshkaSVDLayer):
            total_loss += module.get_compression_loss()

    return total_loss
