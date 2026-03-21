# Fix Transformer Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 2 bugs in the transformer module: X0Model using scalar sigma instead of per-token timesteps, and transformer block per-token AdaLN reshape producing wrong shapes.

**Architecture:** X0Model must use per-token timesteps (sigma * denoise_mask) for the denoising formula so preserved tokens (mask=0) get timestep=0 and are kept unchanged. The transformer block must reshape per-token AdaLN params as (B, N, 9, dim) not (B*N, 9, dim).

**Tech Stack:** Python 3.11+, MLX, pytest

---

### Task 1: Fix X0Model to use per-token timesteps

The reference `X0Model.forward` calls `to_denoised(video.latent, vx, video.timesteps)` where `video.timesteps` = `sigma * denoise_mask` (shape B, N). Our port uses scalar `sigma[:, None, None]` for all tokens.

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/model.py` (X0Model class, ~line 393-440)

Change X0Model.__call__ to use per-token timesteps when available:

```python
    def __call__(self, ...) -> tuple[mx.array, mx.array]:
        video_v, audio_v = self.model(
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=sigma,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            **kwargs,
        )

        # x0 = x_t - sigma * v
        # Use per-token timesteps when available (preserved tokens get timestep=0)
        if video_timesteps is not None:
            video_sigma = video_timesteps[:, :, None].astype(mx.float32)
        else:
            video_sigma = sigma[:, None, None].astype(mx.float32)

        if audio_timesteps is not None:
            audio_sigma = audio_timesteps[:, :, None].astype(mx.float32)
        else:
            audio_sigma = sigma[:, None, None].astype(mx.float32)

        video_x0 = (video_latent.astype(mx.float32) - video_sigma * video_v.astype(mx.float32)).astype(video_latent.dtype)
        audio_x0 = (audio_latent.astype(mx.float32) - audio_sigma * audio_v.astype(mx.float32)).astype(audio_latent.dtype)

        return video_x0, audio_x0
```

---

### Task 2: Fix transformer block per-token AdaLN reshape

When per-token AdaLN params are (B, N, 9*dim), the reshape `(-1, 9, dim)` gives (B*N, 9, dim). Then `[:, i, :]` gives (B*N, dim) and `[:, None, :]` gives (B*N, 1, dim), which doesn't broadcast correctly with (B, N, dim) hidden states.

Fix: detect per-token (3D) vs scalar (2D) and reshape accordingly.

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/transformer.py` (BasicAVTransformerBlock.__call__, ~line 202-230)

Add a helper method and update the param unpacking:

```python
    @staticmethod
    def _unpack_adaln(params: mx.array, num_params: int, dim: int) -> list[mx.array]:
        """Unpack AdaLN parameters, handling both scalar (B, P*dim) and per-token (B, N, P*dim).

        Returns list of P arrays, each (B, 1, dim) for scalar or (B, N, dim) for per-token.
        """
        if params.ndim == 2:
            # Scalar: (B, P*dim) → (B, P, dim) → list of (B, dim) → (B, 1, dim)
            p = params.reshape(-1, num_params, dim)
            return [p[:, i, :][:, None, :] for i in range(num_params)]
        else:
            # Per-token: (B, N, P*dim) → (B, N, P, dim) → list of (B, N, dim)
            B, N, _ = params.shape
            p = params.reshape(B, N, num_params, dim)
            return [p[:, :, i, :] for i in range(num_params)]
```

Then replace the param unpacking code to use this helper.
