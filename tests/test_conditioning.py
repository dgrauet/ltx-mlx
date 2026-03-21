"""Tests for the conditioning system."""

import mlx.core as mx

from ltx_core_mlx.conditioning.mask_utils import build_attention_mask, update_attention_mask
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    TemporalRegionMask,
    VideoConditionByLatentIndex,
    apply_denoise_mask,
    create_initial_state,
)
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent


class TestLatentState:
    def test_create_initial_state(self):
        state = create_initial_state((1, 16, 8), seed=42)
        assert state.latent.shape == (1, 16, 8)
        assert state.clean_latent.shape == (1, 16, 8)
        assert state.denoise_mask.shape == (1, 16, 1)
        # All mask values should be 1 (full denoise)
        assert float(mx.min(state.denoise_mask)) == 1.0

    def test_create_with_clean_latent(self):
        clean = mx.ones((1, 16, 8))
        state = create_initial_state((1, 16, 8), seed=42, clean_latent=clean)
        assert mx.allclose(state.clean_latent, clean).item()

    def test_default_optional_fields(self):
        state = create_initial_state((1, 16, 8), seed=42)
        assert state.positions is None
        assert state.attention_mask is None

    def test_create_with_positions(self):
        positions = mx.zeros((1, 16, 3))
        state = create_initial_state((1, 16, 8), seed=42, positions=positions)
        assert state.positions is not None
        assert state.positions.shape == (1, 16, 3)


class TestApplyDenoiseMask:
    def test_full_denoise(self):
        x0 = mx.ones((1, 4, 8))
        clean = mx.zeros((1, 4, 8))
        mask = mx.ones((1, 4, 1))
        result = apply_denoise_mask(x0, clean, mask)
        assert mx.allclose(result, x0).item()

    def test_full_preserve(self):
        x0 = mx.ones((1, 4, 8))
        clean = mx.zeros((1, 4, 8))
        mask = mx.zeros((1, 4, 1))
        result = apply_denoise_mask(x0, clean, mask)
        assert mx.allclose(result, clean).item()

    def test_half_mask(self):
        x0 = mx.ones((1, 4, 8))
        clean = mx.zeros((1, 4, 8))
        mask = mx.array([[[1.0], [1.0], [0.0], [0.0]]])
        result = apply_denoise_mask(x0, clean, mask)
        # First two tokens: x0 (ones), last two: clean (zeros)
        assert float(mx.sum(result[:, :2, :])) > 0
        assert float(mx.sum(result[:, 2:, :])) == 0


class TestTemporalRegionMask:
    def test_mask_shape(self):
        region = TemporalRegionMask(start_frame=1, end_frame=3)
        mask = region.create_mask(num_frames=4, tokens_per_frame=2)
        assert mask.shape == (1, 8, 1)


class TestVideoConditionByLatentIndex:
    def test_preserves_frame(self):
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        # First frame's mask should be 0 (preserved)
        assert float(new_state.denoise_mask[0, 0, 0]) == 0.0
        assert float(new_state.denoise_mask[0, 1, 0]) == 0.0

    def test_updates_latent_field(self):
        """apply() must set state.latent at conditioned frames, not just clean_latent."""
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        # The latent at frame 0 tokens should equal the clean conditioning
        assert mx.allclose(new_state.latent[0, :2, :], clean[0], atol=1e-6).item()

    def test_strength_partial(self):
        """strength=0.5 should set denoise_mask to 0.5 for conditioned frames."""
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean, strength=0.5)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        assert abs(float(new_state.denoise_mask[0, 0, 0]) - 0.5) < 1e-6
        assert abs(float(new_state.denoise_mask[0, 1, 0]) - 0.5) < 1e-6
        # Non-conditioned frames stay at 1.0
        assert float(new_state.denoise_mask[0, 2, 0]) == 1.0

    def test_positions_preserved(self):
        """apply() should preserve positions from the input state."""
        positions = mx.zeros((1, 8, 3))
        state = create_initial_state((1, 8, 4), seed=42, positions=positions)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.positions is not None
        assert new_state.positions.shape == (1, 8, 3)


class TestBuildAttentionMask:
    def test_first_conditioning_item(self):
        """First item: existing_mask=None, builds (B, N+M, N+M) with 1s top-left."""
        cross_mask = mx.ones((1, 4))  # 4 new tokens, full attention
        mask = build_attention_mask(
            existing_mask=None,
            num_noisy_tokens=8,
            num_new_tokens=4,
            num_existing_tokens=8,
            cross_mask=cross_mask,
        )
        assert mask.shape == (1, 12, 12)
        # Top-left 8x8: all 1s (no prior mask)
        assert float(mx.min(mask[0, :8, :8])) == 1.0
        # Bottom-right 4x4: all 1s (self-attend)
        assert float(mx.min(mask[0, 8:, 8:])) == 1.0
        # Noisy->new_ref: cross_mask (1s)
        assert float(mx.min(mask[0, :8, 8:])) == 1.0
        # New_ref->noisy: cross_mask (1s)
        assert float(mx.min(mask[0, 8:, :8])) == 1.0

    def test_cross_mask_zeros(self):
        """cross_mask=0 blocks attention between noisy and reference."""
        cross_mask = mx.zeros((1, 4))
        mask = build_attention_mask(
            existing_mask=None,
            num_noisy_tokens=8,
            num_new_tokens=4,
            num_existing_tokens=8,
            cross_mask=cross_mask,
        )
        # Noisy->new_ref: blocked
        assert float(mx.max(mask[0, :8, 8:])) == 0.0
        # New_ref->noisy: blocked
        assert float(mx.max(mask[0, 8:, :8])) == 0.0

    def test_second_conditioning_preserves_existing(self):
        """Second item preserves previous mask in top-left block."""
        # Existing mask: 12x12, with prev_ref->noisy blocked (zeros)
        existing = mx.ones((1, 12, 12))
        existing = existing.at[:, 8:, :8].add(mx.full((1, 4, 8), -1.0))  # zero block
        cross_mask = mx.ones((1, 3))  # 3 new tokens
        mask = build_attention_mask(
            existing_mask=existing,
            num_noisy_tokens=8,
            num_new_tokens=3,
            num_existing_tokens=12,
            cross_mask=cross_mask,
        )
        assert mask.shape == (1, 15, 15)
        # Top-left 12x12 preserved from existing
        assert mx.allclose(mask[:, :12, :12], existing, atol=1e-6).item()
        # prev_ref->new_ref: 0 (no cross-ref attention)
        assert float(mx.max(mask[0, 8:12, 12:])) == 0.0


class TestUpdateAttentionMask:
    def test_none_mask_none_existing(self):
        """No mask needed: returns None."""
        state = LatentState(
            latent=mx.zeros((1, 8, 4)),
            clean_latent=mx.zeros((1, 8, 4)),
            denoise_mask=mx.ones((1, 8, 1)),
        )
        result = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=8,
            num_new_tokens=4,
            batch_size=1,
        )
        assert result is None

    def test_scalar_mask(self):
        """Scalar attention_mask creates uniform cross_mask."""
        state = LatentState(
            latent=mx.zeros((1, 8, 4)),
            clean_latent=mx.zeros((1, 8, 4)),
            denoise_mask=mx.ones((1, 8, 1)),
        )
        result = update_attention_mask(
            latent_state=state,
            attention_mask=1.0,
            num_noisy_tokens=8,
            num_new_tokens=4,
            batch_size=1,
        )
        assert result is not None
        assert result.shape == (1, 12, 12)


class TestVideoConditionByKeyframeIndex:
    def test_appends_tokens(self):
        state = create_initial_state((1, 8, 4), seed=42)
        kf_latents = mx.ones((1, 2, 4))
        cond = VideoConditionByKeyframeIndex(keyframe_indices=[0], keyframe_latents=kf_latents)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.latent.shape == (1, 10, 4)
        assert new_state.clean_latent.shape == (1, 10, 4)
        assert new_state.denoise_mask.shape == (1, 10, 1)

    def test_strength_default(self):
        """Default strength=1.0: keyframe mask = 0.0 (fully preserved)."""
        state = create_initial_state((1, 8, 4), seed=42)
        kf_latents = mx.ones((1, 2, 4))
        cond = VideoConditionByKeyframeIndex(keyframe_indices=[0], keyframe_latents=kf_latents)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert float(new_state.denoise_mask[0, 8, 0]) == 0.0
        assert float(new_state.denoise_mask[0, 9, 0]) == 0.0

    def test_strength_partial(self):
        """strength=0.5: keyframe mask = 0.5."""
        state = create_initial_state((1, 8, 4), seed=42)
        kf_latents = mx.ones((1, 2, 4))
        cond = VideoConditionByKeyframeIndex(keyframe_indices=[0], keyframe_latents=kf_latents, strength=0.5)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert abs(float(new_state.denoise_mask[0, 8, 0]) - 0.5) < 1e-6

    def test_positions_extended(self):
        """Positions should grow when tokens are appended."""
        positions = mx.zeros((1, 8, 3))
        state = create_initial_state((1, 8, 4), seed=42, positions=positions)
        kf_positions = mx.ones((1, 2, 3))
        kf_latents = mx.ones((1, 2, 4))
        cond = VideoConditionByKeyframeIndex(
            keyframe_indices=[0],
            keyframe_latents=kf_latents,
            keyframe_positions=kf_positions,
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.positions is not None
        assert new_state.positions.shape == (1, 10, 3)

    def test_no_attention_mask_when_none(self):
        """No attention mask created when none exists and none requested."""
        state = create_initial_state((1, 8, 4), seed=42)
        kf_latents = mx.ones((1, 2, 4))
        cond = VideoConditionByKeyframeIndex(
            keyframe_indices=[0],
            keyframe_latents=kf_latents,
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.attention_mask is None

    def test_attention_mask_padded_when_existing(self):
        """Existing attention mask is padded with 1s for new tokens."""
        existing_mask = mx.ones((1, 8, 8))
        state = LatentState(
            latent=mx.zeros((1, 8, 4)),
            clean_latent=mx.zeros((1, 8, 4)),
            denoise_mask=mx.ones((1, 8, 1)),
            attention_mask=existing_mask,
        )
        kf_latents = mx.ones((1, 2, 4))
        cond = VideoConditionByKeyframeIndex(
            keyframe_indices=[0],
            keyframe_latents=kf_latents,
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.attention_mask is not None
        assert new_state.attention_mask.shape == (1, 10, 10)


class TestVideoConditionByReferenceLatent:
    def test_appends_tokens(self):
        state = create_initial_state((1, 8, 4), seed=42)
        ref = mx.ones((1, 4, 4))
        cond = VideoConditionByReferenceLatent(reference_latent=ref)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.latent.shape == (1, 12, 4)
        assert new_state.denoise_mask.shape == (1, 12, 1)

    def test_strength_default(self):
        """Default strength=1.0: ref mask = 0.0."""
        state = create_initial_state((1, 8, 4), seed=42)
        ref = mx.ones((1, 4, 4))
        cond = VideoConditionByReferenceLatent(reference_latent=ref)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert float(new_state.denoise_mask[0, 8, 0]) == 0.0

    def test_downscale_factor_scales_spatial_only(self):
        """downscale_factor=2 should scale H,W positions by 2, not temporal."""
        positions = mx.ones((1, 8, 3))  # (B, N, 3) with axes [time, h, w]
        state = create_initial_state((1, 8, 4), seed=42, positions=positions)
        ref = mx.ones((1, 4, 4))
        ref_positions = mx.ones((1, 4, 3))
        cond = VideoConditionByReferenceLatent(
            reference_latent=ref,
            reference_positions=ref_positions,
            downscale_factor=2,
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        ref_pos = new_state.positions[:, 8:, :]
        assert abs(float(ref_pos[0, 0, 0]) - 1.0) < 1e-6  # time unchanged
        assert abs(float(ref_pos[0, 0, 1]) - 2.0) < 1e-6  # h scaled by 2
        assert abs(float(ref_pos[0, 0, 2]) - 2.0) < 1e-6  # w scaled by 2

    def test_downscale_factor_1_no_scaling(self):
        """Default downscale_factor=1: no position scaling."""
        positions = mx.ones((1, 8, 3))
        state = create_initial_state((1, 8, 4), seed=42, positions=positions)
        ref = mx.ones((1, 4, 4))
        ref_positions = mx.ones((1, 4, 3))
        cond = VideoConditionByReferenceLatent(
            reference_latent=ref,
            reference_positions=ref_positions,
            downscale_factor=1,
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        ref_pos = new_state.positions[:, 8:, :]
        assert mx.allclose(ref_pos, ref_positions, atol=1e-6).item()
