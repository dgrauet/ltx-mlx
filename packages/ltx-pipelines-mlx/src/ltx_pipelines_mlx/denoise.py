"""Euler denoising loop for joint audio+video diffusion.

Ported from ltx-pipelines denoising loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from tqdm import tqdm

from ltx_core_mlx.conditioning.types.latent_cond import LatentState, apply_denoise_mask
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS


@dataclass
class DenoiseOutput:
    """Output of the denoising loop."""

    video_latent: mx.array  # (B, N_video, C)
    audio_latent: mx.array  # (B, N_audio, C)


def euler_step(
    x: mx.array,
    x0: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    """Single Euler step: x_{t-1} = x_t + (sigma_next - sigma) * (x_t - x0) / sigma.

    Args:
        x: Current noisy sample.
        x0: Predicted clean sample.
        sigma: Current noise level.
        sigma_next: Next noise level.

    Returns:
        Updated sample at sigma_next.
    """
    if sigma == 0:
        return x0
    d = (x - x0) / sigma
    return x + (sigma_next - sigma) * d


def _is_uniform_mask(mask: mx.array) -> bool:
    """Check if denoise mask is all-ones (full denoise, no conditioning)."""
    return bool(mx.all(mask == 1.0).item())


def _compute_per_token_timesteps(
    sigma: float,
    denoise_mask: mx.array,
) -> mx.array:
    """Compute per-token timesteps from sigma and denoise mask.

    Preserved tokens (mask=0) get timestep=0, generated tokens (mask=1)
    get timestep=sigma.

    Args:
        sigma: Current noise level scalar.
        denoise_mask: (B, N, 1) mask.

    Returns:
        Per-token timesteps (B, N).
    """
    return (denoise_mask * sigma).squeeze(-1)


def denoise_loop(
    model: X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    sigmas: list[float] | None = None,
    video_positions: mx.array | None = None,
    audio_positions: mx.array | None = None,
    video_attention_mask: mx.array | None = None,
    audio_attention_mask: mx.array | None = None,
    show_progress: bool = True,
) -> DenoiseOutput:
    """Run the Euler denoising loop for joint audio+video.

    Args:
        model: X0Model wrapping the LTXModel.
        video_state: Video latent state.
        audio_state: Audio latent state.
        video_text_embeds: Text embeddings for video conditioning.
        audio_text_embeds: Text embeddings for audio conditioning.
        sigmas: Sigma schedule (defaults to DISTILLED_SIGMAS).
            The schedule already includes the terminal 0.0, so pairs are
            formed directly: ``zip(sigmas[:-1], sigmas[1:])``.
        video_positions: Positional embeddings for video.
        audio_positions: Positional embeddings for audio.
        video_attention_mask: Attention mask for video.
        audio_attention_mask: Attention mask for audio.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        DenoiseOutput with final video and audio latents.
    """
    if sigmas is None:
        sigmas = DISTILLED_SIGMAS

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

    video_x = video_state.latent
    audio_x = audio_state.latent

    # sigmas already includes the terminal value (e.g. 0.0), so iterate
    # consecutive pairs directly — no extra phantom step.
    steps = list(zip(sigmas[:-1], sigmas[1:]))
    iterator = tqdm(steps, desc="Denoising", disable=not show_progress)

    # Determine whether we need per-token timesteps (for conditioning masks).
    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)

    for sigma, sigma_next in iterator:
        # Build sigma / per-token timesteps
        sigma_arr = mx.array([sigma], dtype=mx.bfloat16)
        B = video_x.shape[0]

        call_kwargs: dict = dict(
            video_latent=video_x,
            audio_latent=audio_x,
            sigma=mx.broadcast_to(sigma_arr, (B,)),
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_attention_mask=video_attention_mask,
            audio_attention_mask=audio_attention_mask,
        )

        # Pass per-token timesteps when mask is not uniform
        if not video_uniform:
            call_kwargs["video_timesteps"] = _compute_per_token_timesteps(sigma, video_state.denoise_mask)
        if not audio_uniform:
            call_kwargs["audio_timesteps"] = _compute_per_token_timesteps(sigma, audio_state.denoise_mask)

        # Predict x0
        video_x0, audio_x0 = model(**call_kwargs)

        # Apply denoise mask: blend with clean latent
        video_x0 = apply_denoise_mask(video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(audio_x0, audio_state.clean_latent, audio_state.denoise_mask)

        # Euler step
        video_x = euler_step(video_x, video_x0, sigma, sigma_next)
        audio_x = euler_step(audio_x, audio_x0, sigma, sigma_next)

        # Force computation for memory efficiency
        mx.async_eval(video_x, audio_x)

    aggressive_cleanup()

    return DenoiseOutput(video_latent=video_x, audio_latent=audio_x)
