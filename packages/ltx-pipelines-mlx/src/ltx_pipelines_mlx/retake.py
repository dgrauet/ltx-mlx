"""Retake pipeline — regenerate a time segment of an existing video.

Ported from ltx-pipelines retake functionality.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.conditioning.types.latent_cond import LatentState, TemporalRegionMask, noise_latent_state
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.video_vae.patchifier import compute_video_latent_shape
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions
from ltx_pipelines_mlx.denoise import denoise_loop
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline


class RetakePipeline(TextToVideoPipeline):
    """Retake pipeline: regenerate a time segment while preserving the rest.

    Args:
        model_dir: Path to model weights.
        low_memory: Aggressive memory management.
    """

    def retake(
        self,
        prompt: str,
        source_video_latent: mx.array,
        source_audio_latent: mx.array,
        start_frame: int,
        end_frame: int,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Regenerate a time segment of a video.

        Args:
            prompt: Text prompt for the regenerated segment.
            source_video_latent: (B, C, F, H, W) source video latent.
            source_audio_latent: (B, 8, T, 16) source audio latent.
            start_frame: First frame to regenerate (inclusive, in latent space).
            end_frame: Last frame to regenerate (exclusive, in latent space).
            height: Video height.
            width: Video width.
            num_frames: Total number of frames.
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        self.load()
        assert self.dit is not None

        F, H, W = compute_video_latent_shape(num_frames, height, width)
        tokens_per_frame = H * W

        # Patchify source
        source_tokens, _ = self.video_patchifier.patchify(source_video_latent)
        audio_tokens, audio_T = self.audio_patchifier.patchify(source_audio_latent)

        # Compute positions for RoPE
        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(audio_T)

        # Create video state with temporal mask
        region = TemporalRegionMask(start_frame, end_frame)
        denoise_mask = region.create_mask(F, tokens_per_frame)

        video_state = LatentState(
            latent=source_tokens,
            clean_latent=source_tokens,
            denoise_mask=denoise_mask,
            positions=video_positions,
        )
        video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)

        # Audio state: apply same temporal mask to audio
        audio_tokens_per_video_frame = audio_T / F
        audio_start = int(start_frame * audio_tokens_per_video_frame)
        audio_end = int(end_frame * audio_tokens_per_video_frame)

        audio_mask = mx.zeros((1, audio_T, 1), dtype=mx.bfloat16)
        audio_mask = audio_mask.at[:, audio_start:audio_end, :].add(
            mx.ones((1, audio_end - audio_start, 1), dtype=mx.bfloat16)
        )

        audio_state = LatentState(
            latent=audio_tokens,
            clean_latent=audio_tokens,
            denoise_mask=audio_mask,
            positions=audio_positions,
        )
        audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        # Encode text
        video_embeds, audio_embeds = self._encode_text(prompt)
        if self.low_memory:
            aggressive_cleanup()

        # Denoise
        sigmas = DISTILLED_SIGMAS[: num_steps + 1] if num_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)

        output = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent
