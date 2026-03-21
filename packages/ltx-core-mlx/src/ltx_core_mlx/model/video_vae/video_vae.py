"""Video VAE Decoder and Encoder.

Ported from ltx-core/src/ltx_core/model/video_vae/video_vae.py

Weight key structure (decoder, after stripping 'vae_decoder.' prefix):
    conv_in.conv.{weight,bias}
    conv_out.conv.{weight,bias}
    per_channel_statistics.{mean,std}
    up_blocks.{0,2,4,6,8}.res_blocks.{N}.conv{1,2}.conv.{weight,bias}  (ResStages)
    up_blocks.{1,3,5,7}.conv.conv.{weight,bias}                         (DepthToSpaceUpsamples)

Weight key structure (encoder, after stripping 'vae_encoder.' prefix):
    conv_in.conv.{weight,bias}          -- (128, 3,3,3, 48)
    conv_out.conv.{weight,bias}         -- (129, 3,3,3, 1024)
    per_channel_statistics.{_mean_of_means, _std_of_means}  -- (128,)
    down_blocks.{0,2,4,6,8}.res_blocks.{N}.conv{1,2}.conv.{weight,bias}
    down_blocks.{1,3,5,7}.conv.conv.{weight,bias}

Note: The encoder weight file uses ``_mean_of_means`` / ``_std_of_means`` but MLX
nn.Module skips underscore-prefixed attributes in ``parameters()``.
We store them as ``mean_of_means`` / ``std_of_means`` and remap during loading
via :func:`~ltx_2_mlx.model.video_vae.ops.remap_encoder_weight_keys`.
"""

from __future__ import annotations

import subprocess
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.convolution import Conv3dBlock
from ltx_core_mlx.model.video_vae.normalization import pixel_norm
from ltx_core_mlx.model.video_vae.ops import EncoderPerChannelStatistics, PerChannelStatistics
from ltx_core_mlx.model.video_vae.resnet import ResBlockStage
from ltx_core_mlx.model.video_vae.sampling import (
    DepthToSpaceUpsample,
    SpaceToDepthDownsample,
    patchify_spatial,
    pixel_shuffle_3d,
)
from ltx_core_mlx.utils.ffmpeg import find_ffmpeg
from ltx_core_mlx.utils.memory import aggressive_cleanup


class VideoDecoder(nn.Module):
    """Video VAE Decoder with streaming frame output.

    Decodes latent (B, C, F', H', W') to pixels, streaming frames
    to ffmpeg for memory efficiency.

    Architecture matches the weight file exactly:
        conv_in -> up_blocks (alternating ResStage / DepthToSpaceUpsample) -> conv_out

    up_blocks layout:
        0: ResStage  1024, 2 blocks
        1: DepthToSpaceUpsample 1024 -> 4096  (pixel-shuffle 2xspatial + 2xtemporal -> 512ch)
        2: ResStage  512,  2 blocks
        3: DepthToSpaceUpsample 512 -> 4096   (pixel-shuffle 2xspatial + 2xtemporal -> 512ch)
        4: ResStage  512,  4 blocks
        5: DepthToSpaceUpsample 512 -> 512    (pixel-shuffle 2xtemporal -> 256ch)
        6: ResStage  256,  6 blocks
        7: DepthToSpaceUpsample 256 -> 512    (pixel-shuffle 2xspatial -> 128ch)
        8: ResStage  128,  4 blocks

    Args:
        causal: If True, uses causal temporal padding (replicate first frame,
            remove first frame after temporal upsample). If False (LTX-2.3
            default), uses symmetric zero-padding and no frame removal.
    """

    def __init__(self, causal: bool = False):
        super().__init__()
        self._causal = causal

        sp_mode = "reflect"

        # Input convolution: 128 latent channels -> 1024
        self.conv_in = Conv3dBlock(
            128,
            1024,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=sp_mode,
        )

        # Flat list of up_blocks -- indices must match weight keys exactly.
        self.up_blocks: list[Any] = [
            ResBlockStage(1024, num_blocks=2, causal=causal, spatial_padding_mode=sp_mode),  # 0
            DepthToSpaceUpsample(1024, 4096, causal=causal, spatial_padding_mode=sp_mode),  # 1
            ResBlockStage(512, num_blocks=2, causal=causal, spatial_padding_mode=sp_mode),  # 2
            DepthToSpaceUpsample(512, 4096, causal=causal, spatial_padding_mode=sp_mode),  # 3
            ResBlockStage(512, num_blocks=4, causal=causal, spatial_padding_mode=sp_mode),  # 4
            DepthToSpaceUpsample(512, 512, causal=causal, spatial_padding_mode=sp_mode),  # 5
            ResBlockStage(256, num_blocks=6, causal=causal, spatial_padding_mode=sp_mode),  # 6
            DepthToSpaceUpsample(256, 512, causal=causal, spatial_padding_mode=sp_mode),  # 7
            ResBlockStage(128, num_blocks=4, causal=causal, spatial_padding_mode=sp_mode),  # 8
        ]

        # Output convolution: 128 -> 48 (3 RGB x 16 for spatial pixel shuffle)
        self.conv_out = Conv3dBlock(
            128,
            48,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=sp_mode,
        )

        # Per-channel normalization statistics
        self.per_channel_statistics = PerChannelStatistics(128)

        # Upsample config: (spatial_factor, temporal_factor) per DepthToSpaceUpsample
        # up_blocks indices 1, 3, 5, 7
        self._upsample_config: list[tuple[int, int]] = [
            (2, 2),  # block 1: 4096 / (2*2*2) = 512
            (2, 2),  # block 3: 4096 / (2*2*2) = 512
            (1, 2),  # block 5: 512 / (1*1*2) = 256
            (2, 1),  # block 7: 512 / (2*2*1) = 128
        ]

    def denormalize_latent(self, latent: mx.array) -> mx.array:
        """Reverse per-channel normalization: x * std + mean.

        Args:
            latent: (B, F, H, W, C) in MLX layout.

        Returns:
            Denormalized latent.
        """
        mean = self.per_channel_statistics.mean.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics.std.reshape(1, 1, 1, 1, -1)
        return latent * std + mean

    def decode(self, latent: mx.array) -> mx.array:
        """Decode latent to pixel frames.

        Args:
            latent: (B, C, F, H, W) latent in PyTorch layout.

        Returns:
            Pixels (B, 3, F, H, W) in [-1, 1].
        """
        # Convert BCFHW -> BFHWC for MLX convolutions
        x = latent.transpose(0, 2, 3, 4, 1)
        x = self.denormalize_latent(x)

        x = self.conv_in(x)

        upsample_idx = 0
        for i, block in enumerate(self.up_blocks):
            x = block(x)

            # Apply pixel shuffle after each DepthToSpaceUpsample (odd indices)
            if i % 2 == 1:
                sf, tf = self._upsample_config[upsample_idx]
                x = pixel_shuffle_3d(x, spatial_factor=sf, temporal_factor=tf)
                # Reference: ALWAYS remove first frame after temporal upsample
                # (unconditional on causal mode, gated on stride[0]==2 only)
                if tf > 1:
                    x = x[:, 1:, :, :, :]
                upsample_idx += 1

        # Pre-activation PixelNorm + SiLU before final conv
        x = self.conv_out(nn.silu(pixel_norm(x)))

        # Final spatial pixel shuffle: 48 -> 3 channels, 4x spatial expansion
        x = pixel_shuffle_3d(x, spatial_factor=4, temporal_factor=1)

        # BFHWC -> BCFHW
        return x.transpose(0, 4, 1, 2, 3)

    def decode_and_stream(
        self,
        latent: mx.array,
        output_path: str,
        fps: float = 24.0,
        audio_path: str | None = None,
    ) -> None:
        """Decode latent and stream frames to ffmpeg.

        Decodes one temporal chunk at a time to avoid OOM.

        Args:
            latent: (B, C, F, H, W) latent.
            output_path: Path to output video file.
            fps: Output frames per second.
            audio_path: Optional audio file to mux.
        """
        ffmpeg = find_ffmpeg()

        # Estimate output dimensions from latent
        _, _, _F_lat, H_lat, W_lat = latent.shape
        out_H = H_lat * 32
        out_W = W_lat * 32

        # Build ffmpeg command
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{out_W}x{out_H}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        if audio_path:
            cmd.extend(["-i", audio_path, "-c:a", "aac"])
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path])

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None

        try:
            # Decode full volume and stream frames
            pixels = self.decode(latent)
            mx.async_eval(pixels)

            num_frames = pixels.shape[2]
            for i in range(num_frames):
                frame = pixels[:, :, i, :, :]  # (B, 3, H, W)
                frame = mx.clip(frame, -1.0, 1.0)
                frame = ((frame + 1.0) * 127.5).astype(mx.uint8)
                # (1, 3, H, W) -> (H, W, 3)
                frame_hwc = frame[0].transpose(1, 2, 0)
                mx.eval(frame_hwc)  # must be sync — async_eval can write before data is ready
                proc.stdin.write(bytes(memoryview(frame_hwc)))
                del frame, frame_hwc
                if i % 8 == 0:
                    aggressive_cleanup()
        except BrokenPipeError:
            pass  # ffmpeg may close early with -shortest; output is still valid
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            proc.wait()
            aggressive_cleanup()


class VideoEncoder(nn.Module):
    """Video VAE Encoder.

    Encodes pixel frames (B, 3, F, H, W) to latent (B, C, F', H', W').
    Temporal 8x, spatial 32x compression with 128 latent channels.

    Reference architecture:
        patchify(4x4 spatial) -> conv_in -> down_blocks -> norm+silu -> conv_out

    down_blocks layout (from config encoder_blocks):
        0: ResStage  128,  4 blocks
        1: SpaceToDepthDownsample 128->256, stride=(1,2,2) -- spatial 2x
        2: ResStage  256,  6 blocks
        3: SpaceToDepthDownsample 256->512, stride=(2,1,1) -- temporal 2x
        4: ResStage  512,  4 blocks
        5: SpaceToDepthDownsample 512->1024, stride=(2,2,2) -- all 2x
        6: ResStage  1024, 2 blocks
        7: SpaceToDepthDownsample 1024->1024, stride=(2,2,2) -- all 2x (mult=1)
        8: ResStage  1024, 2 blocks

    Weight loading: use :func:`~ltx_2_mlx.model.video_vae.ops.remap_encoder_weight_keys`
    before calling ``load_weights`` to handle the underscore-prefixed per-channel stats keys.
    """

    def __init__(self):
        super().__init__()

        # Input convolution: 48 channels (3 RGB x 4x4 spatial patchify) -> 128
        self.conv_in = Conv3dBlock(48, 128, kernel_size=3, padding=1, causal=True)

        # Flat list of down_blocks -- indices must match weight keys exactly.
        self.down_blocks: list = [
            ResBlockStage(128, num_blocks=4, causal=True),  # 0
            SpaceToDepthDownsample(128, 256, stride=(1, 2, 2)),  # 1
            ResBlockStage(256, num_blocks=6, causal=True),  # 2
            SpaceToDepthDownsample(256, 512, stride=(2, 1, 1)),  # 3
            ResBlockStage(512, num_blocks=4, causal=True),  # 4
            SpaceToDepthDownsample(512, 1024, stride=(2, 2, 2)),  # 5
            ResBlockStage(1024, num_blocks=2, causal=True),  # 6
            SpaceToDepthDownsample(1024, 1024, stride=(2, 2, 2)),  # 7
            ResBlockStage(1024, num_blocks=2, causal=True),  # 8
        ]

        # Output convolution: 1024 -> 129 channels
        self.conv_out = Conv3dBlock(1024, 129, kernel_size=3, padding=1, causal=True)

        # Per-channel normalization statistics
        self.per_channel_statistics = EncoderPerChannelStatistics(128)

    def normalize_latent(self, latent: mx.array) -> mx.array:
        """Apply per-channel normalization: (x - mean) / std.

        Args:
            latent: (B, F, H, W, C) in MLX layout.

        Returns:
            Normalized latent.
        """
        mean = self.per_channel_statistics.mean_of_means.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics.std_of_means.reshape(1, 1, 1, 1, -1)
        return (latent - mean) / std

    def encode(self, pixels: mx.array) -> mx.array:
        """Encode pixel frames to latent.

        Args:
            pixels: (B, 3, F, H, W) in [-1, 1], PyTorch layout.

        Returns:
            Latent (B, C, F', H', W') in PyTorch layout.
        """
        # BCFHW -> BFHWC for MLX convolutions
        x = pixels.transpose(0, 2, 3, 4, 1)

        # Spatial patchification: (B, F, H, W, 3) -> (B, F, H/4, W/4, 48)
        # Reference: patchify(sample, patch_size_hw=4, patch_size_t=1)
        x = patchify_spatial(x, patch_size=4)

        x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x)

        # PixelNorm + SiLU before conv_out (reference: conv_norm_out + conv_act)
        x = self.conv_out(nn.silu(pixel_norm(x)))

        # Take first 128 channels (mean), discard the rest (log_var or dummy)
        x = x[:, :, :, :, :128]

        x = self.normalize_latent(x)

        # BFHWC -> BCFHW
        return x.transpose(0, 4, 1, 2, 3)
