"""VideoConditionByKeyframeIndex — appended tokens with attention mask.

Ported from ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.conditioning.mask_utils import update_attention_mask
from ltx_core_mlx.conditioning.types.latent_cond import LatentState


class VideoConditionByKeyframeIndex:
    """Condition generation by appending keyframe tokens with attention masking.

    Args:
        keyframe_indices: List of frame indices for keyframes.
        keyframe_latents: Clean latents for keyframes, (B, num_kf_tokens, C).
        keyframe_positions: Positional embeddings for keyframes, (B, num_kf_tokens, num_axes).
        strength: Conditioning strength. 1.0 = preserved, 0.0 = denoised.
        num_noisy_tokens: Number of original noisy tokens (for attention mask).
    """

    def __init__(
        self,
        keyframe_indices: list[int],
        keyframe_latents: mx.array,
        keyframe_positions: mx.array | None = None,
        strength: float = 1.0,
        num_noisy_tokens: int | None = None,
    ):
        self.keyframe_indices = keyframe_indices
        self.keyframe_latents = keyframe_latents
        self.keyframe_positions = keyframe_positions
        self.strength = strength
        self.num_noisy_tokens = num_noisy_tokens

    def apply(self, state: LatentState, spatial_dims: tuple[int, int, int]) -> LatentState:
        """Apply keyframe conditioning by appending tokens."""
        num_kf = self.keyframe_latents.shape[1]
        mask_value = 1.0 - self.strength

        new_latent = mx.concatenate([state.latent, self.keyframe_latents], axis=1)
        new_clean = mx.concatenate([state.clean_latent, self.keyframe_latents], axis=1)

        kf_mask = mx.full((state.denoise_mask.shape[0], num_kf, 1), mask_value)
        new_mask = mx.concatenate([state.denoise_mask, kf_mask], axis=1)

        # Extend positions if available
        new_positions = state.positions
        if state.positions is not None and self.keyframe_positions is not None:
            new_positions = mx.concatenate([state.positions, self.keyframe_positions], axis=1)

        # Build attention mask
        num_noisy = self.num_noisy_tokens or state.latent.shape[1]
        new_attn_mask = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=num_noisy,
            num_new_tokens=num_kf,
            batch_size=state.latent.shape[0],
        )

        return LatentState(
            latent=new_latent,
            clean_latent=new_clean,
            denoise_mask=new_mask,
            positions=new_positions,
            attention_mask=new_attn_mask,
        )
