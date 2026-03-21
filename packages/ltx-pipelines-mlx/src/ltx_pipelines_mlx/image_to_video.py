"""Image-to-Video pipeline — image + prompt to video+audio.

Ported from ltx-pipelines/src/ltx_pipelines/image_to_video.py
"""

from __future__ import annotations

import mlx.core as mx
from PIL import Image

from ltx_core_mlx.conditioning.types.latent_cond import (
    VideoConditionByLatentIndex,
    apply_conditioning,
    create_initial_state,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.video_vae.patchifier import compute_video_latent_shape
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.denoise import denoise_loop
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.text_to_video import TextToVideoPipeline


class ImageToVideoPipeline(TextToVideoPipeline):
    """Image-to-Video generation pipeline.

    Extends TextToVideoPipeline to condition on a reference image.
    The first frame is encoded and preserved during denoising.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        low_memory: If True, aggressively free memory between stages.
    """

    def __init__(self, model_dir: str, low_memory: bool = True):
        super().__init__(model_dir, low_memory)
        self.vae_encoder: VideoEncoder | None = None

    def load(self) -> None:
        """Load all model components including VAE encoder."""
        super().load()

        if self.vae_encoder is None:
            from ltx_core_mlx.utils.weights import load_split_safetensors

            self.vae_encoder = VideoEncoder()
            enc_weights = load_split_safetensors(self.model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
            self.vae_encoder.load_weights(list(enc_weights.items()))
            aggressive_cleanup()

    def generate_from_image(
        self,
        prompt: str,
        image: Image.Image | str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video conditioned on a reference image.

        Args:
            prompt: Text prompt.
            image: Reference image (PIL Image or path).
            height: Video height.
            width: Video width.
            num_frames: Number of frames.
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None

        # Encode reference image
        img_tensor = prepare_image_for_encoding(image, height, width)
        # Add temporal dim: (1, 3, H, W) -> (1, 3, 1, H, W)
        img_tensor = img_tensor[:, :, None, :, :]
        ref_latent = self.vae_encoder.encode(img_tensor)
        if self.low_memory:
            aggressive_cleanup()

        # Compute shapes
        F, H, W = compute_video_latent_shape(num_frames, height, width)
        video_shape = (1, F * H * W, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        # Encode text
        video_embeds, audio_embeds = self._encode_text(prompt)
        if self.low_memory:
            aggressive_cleanup()

        # Compute positions for RoPE
        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(audio_T)

        # Create initial state with positions
        video_state = create_initial_state(video_shape, seed, positions=video_positions)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # Apply I2V conditioning: preserve first frame
        ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_tokens)
        video_state = apply_conditioning(video_state, [condition], (F, H, W))

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

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        image: Image.Image | str | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> str:
        """Generate and save I2V video+audio.

        Args:
            prompt: Text prompt.
            output_path: Output video path.
            image: Reference image. If None, falls back to T2V.
            height: Video height.
            width: Video width.
            num_frames: Number of frames.
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Path to output video.
        """
        if image is None:
            return super().generate_and_save(prompt, output_path, height, width, num_frames, seed, num_steps)

        video_latent, audio_latent = self.generate_from_image(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_steps=num_steps,
        )

        # Decode and save (reuse parent logic)
        assert self.audio_decoder is not None
        assert self.vocoder is not None
        mel = self.audio_decoder.decode(audio_latent)
        waveform = self.vocoder(mel)
        if self.low_memory:
            aggressive_cleanup()

        import tempfile
        from pathlib import Path

        audio_path = tempfile.mktemp(suffix=".wav")
        self._save_waveform(waveform, audio_path, sample_rate=48000)

        assert self.vae_decoder is not None
        self.vae_decoder.decode_and_stream(video_latent, output_path, fps=24.0, audio_path=audio_path)
        Path(audio_path).unlink(missing_ok=True)
        aggressive_cleanup()

        return output_path
