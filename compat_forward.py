#!/usr/bin/env python3
"""
Compatible forward functions for standard Llama models with MatryoshkaSVDLayer attributes.

These functions work with standard LlamaAttention/LlamaMLP classes (not SVD-LLM variants)
and use matryoshka attributes when available.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def create_compat_attention_forward(attn):
    """
    Create a compatible forward function for standard LlamaAttention that can use MatryoshkaSVDLayer.

    This function:
    - Works with standard LlamaAttention (not SVD_LlamaAttention)
    - Uses matryoshka attributes (q_matryoshka, etc.) if they exist
    - Falls back to standard projections (q_proj) if matryoshka not present
    - Matches standard Llama's return signature exactly

    Args:
        attn: LlamaAttention instance (may have matryoshka attributes)

    Returns:
        Compatible forward function
    """

    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        # Use matryoshka layers if available, otherwise standard projections
        if hasattr(attn, 'q_matryoshka'):
            query_states = attn.q_matryoshka(hidden_states)
            key_states = attn.k_matryoshka(hidden_states)
            value_states = attn.v_matryoshka(hidden_states)
        else:
            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

        # Handle past_key_value for caching
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Apply rotary embeddings
        cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past_key_value
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads for GQA if needed
        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

        if attn_weights.size() != (bsz, attn.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, attn.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, attn.num_heads, q_len, attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn.num_heads, q_len, attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

        # Output projection (use matryoshka if available)
        if hasattr(attn, 'o_matryoshka'):
            attn_output = attn.o_matryoshka(attn_output)
        else:
            attn_output = attn.o_proj(attn_output)

        # Match standard Llama return signature
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return forward


def create_compat_mlp_forward(mlp):
    """
    Create a compatible forward function for standard LlamaMLP that can use MatryoshkaSVDLayer.

    Args:
        mlp: LlamaMLP instance (may have matryoshka attributes)

    Returns:
        Compatible forward function
    """

    def forward(x):
        # Use matryoshka layers if available, otherwise standard projections
        if hasattr(mlp, 'gate_matryoshka'):
            gate = mlp.gate_matryoshka(x)
            up = mlp.up_matryoshka(x)
        else:
            gate = mlp.gate_proj(x)
            up = mlp.up_proj(x)

        # Apply activation
        intermediate = mlp.act_fn(gate) * up

        # Down projection (use matryoshka if available)
        if hasattr(mlp, 'down_matryoshka'):
            output = mlp.down_matryoshka(intermediate)
        else:
            output = mlp.down_proj(intermediate)

        return output

    return forward


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        cos: Cosine values from rotary embeddings
        sin: Sine values from rotary embeddings
        position_ids: Position indices

    Returns:
        Rotated query and key tensors
    """
    # Gather cos and sin values
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for Grouped Query Attention.

    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def patch_model_with_compat_forward(model):
    """
    Patch all attention and MLP layers in the model with compatible forward functions.

    Args:
        model: LlamaForCausalLM model with MatryoshkaSVDLayer attributes

    Returns:
        Number of layers patched
    """
    layers_patched = 0

    for layer in model.model.layers:
        # Patch attention
        if hasattr(layer.self_attn, 'q_matryoshka'):
            layer.self_attn.forward = create_compat_attention_forward(layer.self_attn)
            layers_patched += 1

        # Patch MLP
        if hasattr(layer.mlp, 'gate_matryoshka'):
            layer.mlp.forward = create_compat_mlp_forward(layer.mlp)
            layers_patched += 1

    return layers_patched
