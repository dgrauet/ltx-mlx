"""Extend pipeline — add frames before or after an existing video.

Ported from ltx-pipelines extend functionality.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.conditioning.types.latent_cond import LatentState, create_initial_state
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.video_vae.patchifier import compute_video_latent_shape
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions
from ltx_pipelines_mlx.denoise import denoise_loop
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.text_to_video import TextToVideoPipeline


class ExtendPipeline(TextToVideoPipeline):
    """Extend pipeline: add frames before or after an existing video.

    Args:
        model_dir: Path to model weights.
        low_memory: Aggressive memory management.
    """

    def extend(
        self,
        prompt: str,
        source_video_latent: mx.array,
        source_audio_latent: mx.array,
        extend_frames: int,
        direction: str = "after",
        height: int = 480,
        width: int = 704,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Extend a video by adding frames.

        Args:
            prompt: Text prompt for new frames.
            source_video_latent: (B, C, F, H, W) source video latent.
            source_audio_latent: (B, 8, T, 16) source audio latent.
            extend_frames: Number of latent frames to add.
            direction: "before" or "after".
            height: Video height.
            width: Video width.
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Tuple of (extended_video_latent, extended_audio_latent).
        """
        self.load()
        assert self.dit is not None

        B = source_video_latent.shape[0]
        F_source = source_video_latent.shape[2]
        _, H, W = compute_video_latent_shape(1, height, width)
        F_total = F_source + extend_frames
        tokens_per_frame = H * W

        # Patchify source
        source_tokens, _ = self.video_patchifier.patchify(source_video_latent)

        # Create new tokens for extension
        new_shape = (B, extend_frames * tokens_per_frame, 128)
        mx.random.seed(seed)
        new_noise = mx.random.normal(new_shape).astype(mx.bfloat16)

        # Combine source (preserved) and new (to denoise)
        if direction == "after":
            combined_latent = mx.concatenate([source_tokens, new_noise], axis=1)
            denoise_mask = mx.concatenate(
                [
                    mx.zeros((B, source_tokens.shape[1], 1)),
                    mx.ones((B, new_noise.shape[1], 1)),
                ],
                axis=1,
            )
        else:  # before
            combined_latent = mx.concatenate([new_noise, source_tokens], axis=1)
            denoise_mask = mx.concatenate(
                [
                    mx.ones((B, new_noise.shape[1], 1)),
                    mx.zeros((B, source_tokens.shape[1], 1)),
                ],
                axis=1,
            )

        # Compute positions for RoPE
        video_positions = compute_video_positions(F_total, H, W)
        audio_tokens, audio_T = self.audio_patchifier.patchify(source_audio_latent)
        extend_audio_T = int(audio_T * extend_frames / F_source)
        audio_total_T = audio_T + extend_audio_T
        audio_positions = compute_audio_positions(audio_total_T)

        video_state = LatentState(
            latent=combined_latent,
            clean_latent=combined_latent * (1.0 - denoise_mask),  # clean only preserved parts
            denoise_mask=denoise_mask,
            positions=video_positions,
        )

        # Audio: extend proportionally
        audio_state = create_initial_state((B, audio_total_T, 128), seed + 1, positions=audio_positions)

        # Text encoding
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

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F_total, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent
