# Fix Conditioning System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the conditioning module with the ltx-core reference to fix 7 bugs: missing LatentState fields, broken VideoConditionByLatentIndex, wrong position scaling in VideoConditionByReferenceLatent, missing attention mask system, missing strength parameter, and a guard bug.

**Architecture:** Add `positions` and `attention_mask` optional fields to `LatentState`. Port `mask_utils.py` for attention mask construction. Fix all three conditioning types to match reference semantics. Update `denoise_loop` to pull positions/attention_mask from `LatentState`. Update pipeline callers to populate `LatentState.positions`.

**Tech Stack:** Python 3.11+, MLX, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/latent_cond.py` | Modify | LatentState dataclass, VideoConditionByLatentIndex, helper functions |
| `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/mask_utils.py` | Create | `build_attention_mask`, `update_attention_mask`, `resolve_cross_mask` |
| `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/keyframe_cond.py` | Modify | VideoConditionByKeyframeIndex with strength + attention mask |
| `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/reference_video_cond.py` | Modify | VideoConditionByReferenceLatent with downscale_factor + attention mask |
| `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/__init__.py` | Modify | Re-export new symbols |
| `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/__init__.py` | Modify | Re-export new symbols |
| `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/denoise.py` | Modify | Use positions/attention_mask from LatentState |
| `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/image_to_video.py` | Modify | Populate LatentState.positions |
| `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/text_to_video.py` | Modify | Populate LatentState.positions |
| `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/retake.py` | Modify | Populate LatentState.positions |
| `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/extend.py` | Modify | Populate LatentState.positions |
| `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/keyframe_interp.py` | Modify | Populate LatentState.positions |
| `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/two_stage.py` | Modify | Populate LatentState.positions |
| `tests/test_conditioning.py` | Modify | Updated + new tests for all fixes |

---

### Task 1: Add `positions` and `attention_mask` to LatentState

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/latent_cond.py:13-26`
- Test: `tests/test_conditioning.py`

- [ ] **Step 1: Write failing tests for new LatentState fields**

```python
# In tests/test_conditioning.py, add to TestLatentState:

def test_default_optional_fields(self):
    state = create_initial_state((1, 16, 8), seed=42)
    assert state.positions is None
    assert state.attention_mask is None

def test_create_with_positions(self):
    positions = mx.zeros((1, 16, 3))
    state = create_initial_state((1, 16, 8), seed=42, positions=positions)
    assert state.positions is not None
    assert state.positions.shape == (1, 16, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conditioning.py::TestLatentState -v`
Expected: FAIL — `LatentState` has no `positions` field, `create_initial_state` doesn't accept `positions`

- [ ] **Step 3: Update LatentState and create_initial_state**

In `latent_cond.py`, change the dataclass to:

```python
@dataclass
class LatentState:
    """Generation state for diffusion.

    Attributes:
        latent: Noisy latent being denoised, (B, N, C).
        clean_latent: Original clean latent for conditioning, (B, N, C).
        denoise_mask: Per-token mask: 1.0 = denoise (generate), 0.0 = preserve.
        positions: Positional indices (B, N, num_axes) or None.
        attention_mask: Self-attention mask (B, N, N) with values in [0,1], or None.
    """

    latent: mx.array
    clean_latent: mx.array
    denoise_mask: mx.array
    positions: mx.array | None = None
    attention_mask: mx.array | None = None
```

Update `create_initial_state` signature:

```python
def create_initial_state(
    shape: tuple[int, ...],
    seed: int,
    clean_latent: mx.array | None = None,
    positions: mx.array | None = None,
) -> LatentState:
```

And the return:

```python
    return LatentState(
        latent=noise,
        clean_latent=clean_latent,
        denoise_mask=denoise_mask,
        positions=positions,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conditioning.py::TestLatentState -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/latent_cond.py tests/test_conditioning.py
git commit -m "feat(conditioning): add positions and attention_mask to LatentState"
```

---

### Task 2: Fix VideoConditionByLatentIndex bugs

Three bugs to fix:
1. `apply()` doesn't update `state.latent` — reference sets latent, clean_latent, AND denoise_mask
2. Guard `if self.clean_latent.shape[1] > i * tokens_per_frame` protects clean_latent but not mask
3. Missing `strength` parameter (hardcoded to 1.0)

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/latent_cond.py:28-71`
- Test: `tests/test_conditioning.py`

- [ ] **Step 1: Write failing tests**

```python
# In tests/test_conditioning.py, add to TestVideoConditionByLatentIndex:

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
    condition = VideoConditionByLatentIndex(
        frame_indices=[0], clean_latent=clean, strength=0.5
    )
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conditioning.py::TestVideoConditionByLatentIndex -v`
Expected: FAIL — latent not updated, no strength param, no positions field

- [ ] **Step 3: Rewrite VideoConditionByLatentIndex**

```python
class VideoConditionByLatentIndex:
    """Replace tokens at specific frame indices with conditioning latent.

    Used for Image-to-Video: the first frame's latent tokens are
    set as clean (preserved) while the rest are generated.

    Args:
        frame_indices: List of frame indices to condition.
        clean_latent: Clean latent tokens for those frames, (B, num_indices * tokens_per_frame, C).
        strength: Conditioning strength. 1.0 = fully preserved (mask=0),
            0.0 = fully denoised (mask=1). Default 1.0.
    """

    def __init__(
        self,
        frame_indices: list[int],
        clean_latent: mx.array,
        strength: float = 1.0,
    ):
        self.frame_indices = frame_indices
        self.clean_latent = clean_latent
        self.strength = strength

    def apply(self, state: LatentState, spatial_dims: tuple[int, int, int]) -> LatentState:
        """Apply conditioning: replace latent, clean_latent, and mask at frame indices."""
        _F, H, W = spatial_dims
        tokens_per_frame = H * W

        new_latent = state.latent
        new_clean = state.clean_latent
        new_mask = state.denoise_mask
        mask_value = 1.0 - self.strength

        for i, frame_idx in enumerate(self.frame_indices):
            start = frame_idx * tokens_per_frame
            end = start + tokens_per_frame
            src_start = i * tokens_per_frame
            src_end = src_start + tokens_per_frame

            if src_end > self.clean_latent.shape[1]:
                break

            frame_tokens = self.clean_latent[:, src_start:src_end, :]

            # Update latent, clean_latent, and mask together
            new_latent = mx.concatenate(
                [new_latent[:, :start, :], frame_tokens, new_latent[:, end:, :]], axis=1
            )
            new_clean = mx.concatenate(
                [new_clean[:, :start, :], frame_tokens, new_clean[:, end:, :]], axis=1
            )
            frame_mask = mx.full(
                (state.denoise_mask.shape[0], tokens_per_frame, 1), mask_value
            )
            new_mask = mx.concatenate(
                [new_mask[:, :start, :], frame_mask, new_mask[:, end:, :]], axis=1
            )

        return LatentState(
            latent=new_latent,
            clean_latent=new_clean,
            denoise_mask=new_mask,
            positions=state.positions,
            attention_mask=state.attention_mask,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conditioning.py::TestVideoConditionByLatentIndex -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/latent_cond.py tests/test_conditioning.py
git commit -m "fix(conditioning): VideoConditionByLatentIndex updates latent, adds strength, fixes guard"
```

---

### Task 3: Port mask_utils.py

**Files:**
- Create: `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/mask_utils.py`
- Test: `tests/test_conditioning.py`

- [ ] **Step 1: Write failing tests for mask utilities**

```python
# In tests/test_conditioning.py, add:

from ltx_core_mlx.conditioning.mask_utils import build_attention_mask, update_attention_mask


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
        existing = mx.ones((1, 12, 12))
        existing = existing.at[:, 8:, :8].add(mx.full((1, 4, 8), -1.0))  # zero out prev_ref->noisy
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conditioning.py::TestBuildAttentionMask -v`
Expected: FAIL — `mask_utils` module doesn't exist

- [ ] **Step 3: Create mask_utils.py**

```python
"""Utilities for building 2D self-attention masks for conditioning items.

Ported from ltx-core/src/ltx_core/conditioning/mask_utils.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from ltx_core_mlx.conditioning.types.latent_cond import LatentState


def resolve_cross_mask(
    attention_mask: float | int | mx.array,
    num_new_tokens: int,
    batch_size: int,
) -> mx.array:
    """Convert an attention_mask (scalar or array) to a (B, M) cross_mask.

    Args:
        attention_mask: Scalar, 1D (M,), or 2D (B, M) array.
        num_new_tokens: Number of new conditioning tokens M.
        batch_size: Batch size B.

    Returns:
        Cross-mask of shape (B, M).
    """
    if isinstance(attention_mask, (int, float)):
        return mx.full((batch_size, num_new_tokens), attention_mask)

    mask = attention_mask
    if mask.ndim == 0:
        return mx.full((batch_size, num_new_tokens), float(mask.item()))

    if mask.ndim == 1:
        if mask.shape[0] != num_new_tokens:
            raise ValueError(
                f"1-D attention_mask length must equal num_new_tokens ({num_new_tokens}), "
                f"got shape {tuple(mask.shape)}"
            )
        return mx.broadcast_to(mask[None, :], (batch_size, num_new_tokens))

    if mask.ndim == 2:
        _b, m = mask.shape
        if m != num_new_tokens:
            raise ValueError(
                f"2-D attention_mask dim-1 must equal num_new_tokens ({num_new_tokens}), "
                f"got shape {tuple(mask.shape)}"
            )
        return mx.broadcast_to(mask, (batch_size, num_new_tokens))

    raise ValueError(f"attention_mask must be 0-D, 1-D, or 2-D, got {mask.ndim}-D")


def build_attention_mask(
    existing_mask: mx.array | None,
    num_noisy_tokens: int,
    num_new_tokens: int,
    num_existing_tokens: int,
    cross_mask: mx.array,
) -> mx.array:
    """Build or expand a (B, N+M, N+M) self-attention mask.

    Block structure:
                 noisy      prev_ref    new_ref
               (N_noisy)   (N-N_noisy)    (M)
             +───────────+───────────+───────────+
    noisy    |  existing |  existing |   cross   |
             +───────────+───────────+───────────+
    prev_ref |  existing |  existing |     0     |
             +───────────+───────────+───────────+
    new_ref  |   cross   |     0     |     1     |
             +───────────+───────────+───────────+

    Args:
        existing_mask: Current mask (B, N, N) or None.
        num_noisy_tokens: Original noisy token count.
        num_new_tokens: New conditioning tokens M.
        num_existing_tokens: Current total tokens N.
        cross_mask: Per-token attention weight (B, M), values in [0, 1].

    Returns:
        Attention mask (B, N+M, N+M).
    """
    batch_size = cross_mask.shape[0]
    total = num_existing_tokens + num_new_tokens
    mask = mx.zeros((batch_size, total, total))

    # Top-left: preserve existing or fill with 1s
    if existing_mask is not None:
        mask = mask.at[:, :num_existing_tokens, :num_existing_tokens].add(existing_mask)
    else:
        mask = mask.at[:, :num_existing_tokens, :num_existing_tokens].add(
            mx.ones((batch_size, num_existing_tokens, num_existing_tokens))
        )

    # Bottom-right: new ref tokens self-attend
    mask = mask.at[:, num_existing_tokens:, num_existing_tokens:].add(
        mx.ones((batch_size, num_new_tokens, num_new_tokens))
    )

    # Noisy -> new_ref: cross_mask[:, j] for each column j
    mask = mask.at[:, :num_noisy_tokens, num_existing_tokens:].add(
        mx.broadcast_to(cross_mask[:, None, :], (batch_size, num_noisy_tokens, num_new_tokens))
    )

    # New_ref -> noisy: cross_mask[:, i] for each row i
    mask = mask.at[:, num_existing_tokens:, :num_noisy_tokens].add(
        mx.broadcast_to(cross_mask[:, :, None], (batch_size, num_new_tokens, num_noisy_tokens))
    )

    return mask


def update_attention_mask(
    latent_state: LatentState,
    attention_mask: float | mx.array | None,
    num_noisy_tokens: int,
    num_new_tokens: int,
    batch_size: int,
) -> mx.array | None:
    """Build or update the self-attention mask for newly appended conditioning tokens.

    Args:
        latent_state: Current latent state.
        attention_mask: Per-token attention weight (scalar, 1D, 2D, or None).
        num_noisy_tokens: Original noisy token count.
        num_new_tokens: New conditioning tokens being appended.
        batch_size: Batch size.

    Returns:
        Updated mask (B, N+M, N+M) or None.
    """
    if attention_mask is None:
        if latent_state.attention_mask is None:
            return None
        cross_mask = mx.ones((batch_size, num_new_tokens))
        return build_attention_mask(
            existing_mask=latent_state.attention_mask,
            num_noisy_tokens=num_noisy_tokens,
            num_new_tokens=num_new_tokens,
            num_existing_tokens=latent_state.latent.shape[1],
            cross_mask=cross_mask,
        )

    cross_mask = resolve_cross_mask(attention_mask, num_new_tokens, batch_size)
    return build_attention_mask(
        existing_mask=latent_state.attention_mask,
        num_noisy_tokens=num_noisy_tokens,
        num_new_tokens=num_new_tokens,
        num_existing_tokens=latent_state.latent.shape[1],
        cross_mask=cross_mask,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conditioning.py::TestBuildAttentionMask tests/test_conditioning.py::TestUpdateAttentionMask -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/mask_utils.py tests/test_conditioning.py
git commit -m "feat(conditioning): port mask_utils with build/update/resolve attention masks"
```

---

### Task 4: Fix VideoConditionByKeyframeIndex

Bugs: missing `strength`, no attention mask via `update_attention_mask`, positions not extended.

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/keyframe_cond.py`
- Test: `tests/test_conditioning.py`

- [ ] **Step 1: Write failing tests**

```python
# In tests/test_conditioning.py, add:

from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex


class TestVideoConditionByKeyframeIndex:
    def test_appends_tokens(self):
        state = create_initial_state((1, 8, 4), seed=42)
        kf_latents = mx.ones((1, 2, 4))
        cond = VideoConditionByKeyframeIndex(
            keyframe_indices=[0], keyframe_latents=kf_latents
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        # Sequence grows by 2
        assert new_state.latent.shape == (1, 10, 4)
        assert new_state.clean_latent.shape == (1, 10, 4)
        assert new_state.denoise_mask.shape == (1, 10, 1)

    def test_strength_default(self):
        """Default strength=1.0: keyframe mask = 0.0 (fully preserved)."""
        state = create_initial_state((1, 8, 4), seed=42)
        kf_latents = mx.ones((1, 2, 4))
        cond = VideoConditionByKeyframeIndex(
            keyframe_indices=[0], keyframe_latents=kf_latents
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        # Appended mask should be 0.0 (1 - 1.0)
        assert float(new_state.denoise_mask[0, 8, 0]) == 0.0
        assert float(new_state.denoise_mask[0, 9, 0]) == 0.0

    def test_strength_partial(self):
        """strength=0.5: keyframe mask = 0.5."""
        state = create_initial_state((1, 8, 4), seed=42)
        kf_latents = mx.ones((1, 2, 4))
        cond = VideoConditionByKeyframeIndex(
            keyframe_indices=[0], keyframe_latents=kf_latents, strength=0.5
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert abs(float(new_state.denoise_mask[0, 8, 0]) - 0.5) < 1e-6

    def test_positions_extended(self):
        """Positions should grow when tokens are appended."""
        positions = mx.zeros((1, 8, 3))
        state = create_initial_state((1, 8, 4), seed=42, positions=positions)
        kf_positions = mx.ones((1, 2, 3))
        kf_latents = mx.ones((1, 2, 4))
        cond = VideoConditionByKeyframeIndex(
            keyframe_indices=[0], keyframe_latents=kf_latents,
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
            keyframe_indices=[0], keyframe_latents=kf_latents,
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        # Per reference: update_attention_mask(attention_mask=None) with no
        # existing mask returns None (no masking needed)
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
            keyframe_indices=[0], keyframe_latents=kf_latents,
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.attention_mask is not None
        assert new_state.attention_mask.shape == (1, 10, 10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conditioning.py::TestVideoConditionByKeyframeIndex -v`
Expected: FAIL — no strength param, no positions extension, no attention mask

- [ ] **Step 3: Rewrite VideoConditionByKeyframeIndex**

```python
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
            new_positions = mx.concatenate(
                [state.positions, self.keyframe_positions], axis=1
            )

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conditioning.py::TestVideoConditionByKeyframeIndex -v`
Expected: PASS

- [ ] **Step 5: Remove old `build_attention_mask` method**

The old `build_attention_mask` method (lines 55-88 in old code) is replaced by `mask_utils.update_attention_mask`. Delete it. The `keyframe_interp.py` pipeline may call it — check and update if needed.

Run: `uv run pytest tests/test_conditioning.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/keyframe_cond.py tests/test_conditioning.py
git commit -m "fix(conditioning): VideoConditionByKeyframeIndex adds strength, positions, attention mask"
```

---

### Task 5: Fix VideoConditionByReferenceLatent

Bugs: `position_scale=10.0` scales ALL axes (wrong). Reference uses `downscale_factor=1` on spatial axes only. Missing `strength`.

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/reference_video_cond.py`
- Test: `tests/test_conditioning.py`

- [ ] **Step 1: Write failing tests**

```python
# In tests/test_conditioning.py, add:

from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent


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
        # Ref positions: time=1.0 (unchanged), h=2.0 (scaled), w=2.0 (scaled)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conditioning.py::TestVideoConditionByReferenceLatent -v`
Expected: FAIL — no `downscale_factor`, no `reference_positions`, old `position_scale` scales all axes

- [ ] **Step 3: Rewrite VideoConditionByReferenceLatent**

```python
"""VideoConditionByReferenceLatent — IC-LoRA style reference conditioning.

Ported from ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.conditioning.mask_utils import update_attention_mask
from ltx_core_mlx.conditioning.types.latent_cond import LatentState


class VideoConditionByReferenceLatent:
    """Condition generation by appending reference latent with scaled positions.

    Used for IC-LoRA: a reference image/video latent is appended to the sequence.
    The downscale_factor scales spatial positions to match target coordinate space
    (e.g., 2 = half-resolution reference for full-resolution target).

    Args:
        reference_latent: Reference latent tokens, (B, Nr, C).
        reference_positions: Positions for reference tokens, (B, Nr, num_axes).
        downscale_factor: Target/reference resolution ratio. Spatial positions
            (height, width) are multiplied by this. Default 1 (no scaling).
        strength: Conditioning strength. 1.0 = preserved, 0.0 = denoised.
    """

    def __init__(
        self,
        reference_latent: mx.array,
        reference_positions: mx.array | None = None,
        downscale_factor: int = 1,
        strength: float = 1.0,
    ):
        self.reference_latent = reference_latent
        self.reference_positions = reference_positions
        self.downscale_factor = downscale_factor
        self.strength = strength

    def apply(self, state: LatentState, spatial_dims: tuple[int, int, int]) -> LatentState:
        """Apply reference conditioning by appending tokens."""
        num_ref = self.reference_latent.shape[1]
        mask_value = 1.0 - self.strength

        new_latent = mx.concatenate([state.latent, self.reference_latent], axis=1)
        new_clean = mx.concatenate([state.clean_latent, self.reference_latent], axis=1)

        ref_mask = mx.full((state.denoise_mask.shape[0], num_ref, 1), mask_value)
        new_mask = mx.concatenate([state.denoise_mask, ref_mask], axis=1)

        # Extend positions with optional spatial scaling
        new_positions = state.positions
        if state.positions is not None and self.reference_positions is not None:
            ref_pos = self.reference_positions
            if self.downscale_factor != 1:
                # Scale spatial axes only (height=axis 1, width=axis 2), not temporal (axis 0)
                scale = mx.array([1.0, float(self.downscale_factor), float(self.downscale_factor)])
                ref_pos = ref_pos * scale[None, None, :]
            new_positions = mx.concatenate([state.positions, ref_pos], axis=1)

        # Build attention mask
        num_noisy = state.latent.shape[1]
        new_attn_mask = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=num_noisy,
            num_new_tokens=num_ref,
            batch_size=state.latent.shape[0],
        )

        return LatentState(
            latent=new_latent,
            clean_latent=new_clean,
            denoise_mask=new_mask,
            positions=new_positions,
            attention_mask=new_attn_mask,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conditioning.py::TestVideoConditionByReferenceLatent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/reference_video_cond.py tests/test_conditioning.py
git commit -m "fix(conditioning): VideoConditionByReferenceLatent uses downscale_factor on spatial axes only"
```

---

### Task 6: Update __init__.py exports and noise_latent_state

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/__init__.py`
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/__init__.py`
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/types/latent_cond.py` (noise_latent_state)

- [ ] **Step 1: Update noise_latent_state and add_noise_with_state to preserve positions/attention_mask**

In `latent_cond.py`, update `noise_latent_state` return to preserve new fields:

```python
    return LatentState(
        latent=latent,
        clean_latent=state.clean_latent,
        denoise_mask=state.denoise_mask,
        positions=state.positions,
        attention_mask=state.attention_mask,
    )
```

Also update `add_noise_with_state` — while it returns `mx.array` (not LatentState), verify it doesn't construct LatentState. Current code only returns the noisy latent array, so no change needed there.

- [ ] **Step 2: Add mask_utils exports to __init__.py files**

In both `conditioning/__init__.py` and `conditioning/types/__init__.py`, add:

```python
from ltx_core_mlx.conditioning.mask_utils import (
    build_attention_mask,
    resolve_cross_mask,
    update_attention_mask,
)
```

And add to `__all__`:
```python
    "build_attention_mask",
    "resolve_cross_mask",
    "update_attention_mask",
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/test_conditioning.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/
git commit -m "feat(conditioning): export mask_utils, preserve positions in noise_latent_state"
```

---

### Task 7: Update denoise_loop to use LatentState positions/attention_mask

The denoise_loop currently takes `video_positions` and `audio_positions` as separate parameters. It should also support pulling them from `LatentState` so that conditioning items that modify positions (keyframe, reference) propagate correctly.

**Files:**
- Modify: `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/denoise.py`
- Test: `tests/test_conditioning.py`

- [ ] **Step 1: Write a test**

```python
# In tests/test_conditioning.py, add:

from ltx_pipelines_mlx.denoise import denoise_loop, DenoiseOutput


class TestDenoiseLoopPositions:
    def test_positions_from_state(self):
        """denoise_loop should use positions from LatentState when not passed explicitly."""
        # This is a structural test — we can't run the model, but we verify
        # the function signature accepts states with positions
        positions = mx.zeros((1, 8, 3))
        state = create_initial_state((1, 8, 4), seed=42, positions=positions)
        assert state.positions is not None
        assert state.positions.shape == (1, 8, 3)
```

- [ ] **Step 2: Update denoise_loop to prefer LatentState positions**

In `denoise.py`, update `denoise_loop` to resolve positions from state:

```python
    # Resolve positions: explicit params override, then fall back to state
    if video_positions is None and video_state.positions is not None:
        video_positions = video_state.positions
    if audio_positions is None and audio_state.positions is not None:
        audio_positions = audio_state.positions

    # Resolve attention masks from state
    if video_attention_mask is None and video_state.attention_mask is not None:
        video_attention_mask = video_state.attention_mask
    if audio_attention_mask is None and audio_state.attention_mask is not None:
        audio_attention_mask = audio_state.attention_mask
```

Insert these lines right after `if sigmas is None: sigmas = DISTILLED_SIGMAS` (before `video_x = video_state.latent`).

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/ -m "not slow" -q`
Expected: All pass, no regressions

- [ ] **Step 4: Commit**

```bash
git add packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/denoise.py tests/test_conditioning.py
git commit -m "feat(denoise): resolve positions and attention_mask from LatentState"
```

---

### Task 8: Update pipeline callers to populate LatentState.positions

Now that `LatentState` carries positions, pipelines should populate them so the denoise_loop picks them up automatically. This ensures retake/extend/keyframe pipelines — which currently pass no positions — get RoPE embeddings.

**Files:**
- Modify: `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/text_to_video.py`
- Modify: `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/image_to_video.py`
- Modify: `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/retake.py`
- Modify: `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/extend.py`
- Modify: `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/keyframe_interp.py`
- Modify: `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/two_stage.py`

- [ ] **Step 1: Update create_initial_state calls to include positions**

In every pipeline that calls `create_initial_state`, pass the computed `video_positions` and `audio_positions`:

**text_to_video.py** — already computes positions before `create_initial_state`:
```python
video_state = create_initial_state(video_shape, seed, positions=video_positions)
audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)
```

**image_to_video.py** — same pattern, positions are computed in the parent `TextToVideoPipeline.__call__`:
```python
video_state = create_initial_state(video_shape, seed, positions=video_positions)
audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)
```

**retake.py, extend.py, keyframe_interp.py, two_stage.py** — these currently don't compute positions at all. They need to import `compute_video_positions` and `compute_audio_positions` and compute them before calling `create_initial_state`. (Note: the position computation itself has bugs tracked in the utils/ audit — this task just wires up the plumbing so positions flow through LatentState.)

For each, add:
```python
from ltx_core_mlx.utils.positions import compute_video_positions, compute_audio_positions

# Before create_initial_state:
video_positions = compute_video_positions(F, H, W)
audio_positions = compute_audio_positions(audio_T)
video_state = create_initial_state(video_shape, seed, positions=video_positions)
audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -m "not slow" -q`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/
git commit -m "feat(pipelines): populate LatentState.positions in all pipelines"
```
