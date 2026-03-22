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


# --- Res2s second-order sampler ---


def _res2s_phi(j: int, neg_h: float) -> float:
    """Compute phi_j(z) for the res_2s ODE solver."""
    import math

    if abs(neg_h) < 1e-10:
        return 1.0 / math.factorial(j)
    remainder = sum(neg_h**k / math.factorial(k) for k in range(j))
    return (math.exp(neg_h) - remainder) / (neg_h**j)


def _res2s_coefficients(h: float, c2: float = 0.5) -> tuple[float, float, float]:
    """Compute res_2s Runge-Kutta coefficients for a given step size.

    Args:
        h: Step size in log-space = log(sigma / sigma_next).
        c2: Substep position (default 0.5 = midpoint).

    Returns:
        (a21, b1, b2) coefficients.
    """
    a21 = c2 * _res2s_phi(1, -h * c2)
    b2 = _res2s_phi(2, -h) / c2
    b1 = _res2s_phi(1, -h) - b2
    return a21, b1, b2


def _res2s_sde_coeff(
    sigma_next: float,
    sigma_up_fraction: float = 0.5,
) -> tuple[float, float, float]:
    """Compute SDE coefficients for variance-preserving noise injection.

    Returns:
        (alpha_ratio, sigma_down, sigma_up).
    """
    sigma_up = min(sigma_next * sigma_up_fraction, sigma_next * 0.9999)
    sigma_signal = 1.0 - sigma_next
    sigma_residual = max(0.0, sigma_next**2 - sigma_up**2) ** 0.5
    alpha_ratio = sigma_signal + sigma_residual
    sigma_down = sigma_residual / alpha_ratio if alpha_ratio > 0 else sigma_next
    return alpha_ratio, sigma_down, sigma_up


def res2s_denoise_loop(
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
    """Run the res_2s second-order denoising loop for joint audio+video.

    Uses a second-order Runge-Kutta method with SDE noise injection for
    higher quality at fewer steps.

    Args:
        model: X0Model wrapping the LTXModel.
        video_state: Video latent state.
        audio_state: Audio latent state.
        video_text_embeds: Text embeddings for video conditioning.
        audio_text_embeds: Text embeddings for audio conditioning.
        sigmas: Sigma schedule.
        video_positions: Positional embeddings for video.
        audio_positions: Positional embeddings for audio.
        video_attention_mask: Attention mask for video.
        audio_attention_mask: Attention mask for audio.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        DenoiseOutput with final video and audio latents.
    """
    import math

    if sigmas is None:
        sigmas = DISTILLED_SIGMAS

    # Resolve positions and attention masks from state
    if video_positions is None and video_state.positions is not None:
        video_positions = video_state.positions
    if audio_positions is None and audio_state.positions is not None:
        audio_positions = audio_state.positions
    if video_attention_mask is None and video_state.attention_mask is not None:
        video_attention_mask = video_state.attention_mask
    if audio_attention_mask is None and audio_state.attention_mask is not None:
        audio_attention_mask = audio_state.attention_mask

    video_x = video_state.latent
    audio_x = audio_state.latent

    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)

    steps = list(zip(sigmas[:-1], sigmas[1:]))
    iterator = tqdm(steps, desc="Denoising (res2s)", disable=not show_progress)

    def _predict(
        v_x: mx.array,
        a_x: mx.array,
        sig: float,
    ) -> tuple[mx.array, mx.array]:
        """Run model prediction at a given sigma."""
        sig_arr = mx.array([sig], dtype=mx.bfloat16)
        B = v_x.shape[0]
        kw: dict = dict(
            video_latent=v_x,
            audio_latent=a_x,
            sigma=mx.broadcast_to(sig_arr, (B,)),
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_attention_mask=video_attention_mask,
            audio_attention_mask=audio_attention_mask,
        )
        if not video_uniform:
            kw["video_timesteps"] = _compute_per_token_timesteps(sig, video_state.denoise_mask)
        if not audio_uniform:
            kw["audio_timesteps"] = _compute_per_token_timesteps(sig, audio_state.denoise_mask)

        v_x0, a_x0 = model(**kw)
        v_x0 = apply_denoise_mask(v_x0, video_state.clean_latent, video_state.denoise_mask)
        a_x0 = apply_denoise_mask(a_x0, audio_state.clean_latent, audio_state.denoise_mask)
        return v_x0, a_x0

    for step_idx, (sigma, sigma_next) in enumerate(iterator):
        # First evaluation
        video_x0, audio_x0 = _predict(video_x, audio_x, sigma)

        if sigma_next == 0.0:
            video_x = video_x0
            audio_x = audio_x0
        else:
            h = math.log(sigma / sigma_next) if sigma_next > 0 else 0.0
            a21, b1, b2 = _res2s_coefficients(h)
            alpha_ratio, sigma_down, sigma_up = _res2s_sde_coeff(sigma_next)

            # Epsilon prediction
            video_eps = (video_x - video_x0) / sigma if sigma > 0 else mx.zeros_like(video_x)
            audio_eps = (audio_x - audio_x0) / sigma if sigma > 0 else mx.zeros_like(audio_x)

            # Midpoint
            sigma_mid = sigma * (1.0 - a21) + sigma_next * a21
            video_mid = video_x0 + sigma_mid * video_eps
            audio_mid = audio_x0 + sigma_mid * audio_eps

            # Second evaluation at midpoint
            video_x0_mid, audio_x0_mid = _predict(video_mid, audio_mid, sigma_mid)

            video_eps_mid = (video_mid - video_x0_mid) / sigma_mid if sigma_mid > 0 else mx.zeros_like(video_mid)
            audio_eps_mid = (audio_mid - audio_x0_mid) / sigma_mid if sigma_mid > 0 else mx.zeros_like(audio_mid)

            # Combine first and second order estimates
            video_denoised = video_x0 + sigma_down * (b1 * video_eps + b2 * video_eps_mid)
            audio_denoised = audio_x0 + sigma_down * (b1 * audio_eps + b2 * audio_eps_mid)

            # SDE noise injection
            if sigma_up > 0:
                mx.random.seed(step_idx * 1000 + 42)
                video_noise = mx.random.normal(video_x.shape).astype(mx.bfloat16)
                audio_noise = mx.random.normal(audio_x.shape).astype(mx.bfloat16)
                video_x = alpha_ratio * video_denoised + sigma_up * video_noise
                audio_x = alpha_ratio * audio_denoised + sigma_up * audio_noise
            else:
                video_x = video_denoised
                audio_x = audio_denoised

        # Force computation for memory efficiency
        mx.eval(video_x, audio_x)

    aggressive_cleanup()
    return DenoiseOutput(video_latent=video_x, audio_latent=audio_x)
