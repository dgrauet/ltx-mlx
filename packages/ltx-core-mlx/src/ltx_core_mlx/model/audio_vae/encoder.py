"""Audio VAE Encoder — mel spectrogram to latent.

Ported from ltx-core audio VAE encoder. Mirrors the decoder architecture
using Conv2d with the same wrapped-conv key structure.

NOTE: No pre-converted encoder weights are shipped in the audio_vae.safetensors
file. This module exists for architectural completeness and can be used if
encoder weights become available in the future.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.audio_vae.audio_vae import (
    AudioResBlock,
    PerChannelStatistics,
    WrappedConv2d,
    pixel_norm,
)


class AudioDownsample(nn.Module):
    """2x spatial downsample with causal padding support.

    Key: ``downsample.conv.conv.{weight,bias}``
    """

    def __init__(self, channels: int, causal: bool = False):
        super().__init__()
        self._causal = causal
        if causal:
            # Causal downsample: manual pad then stride-2 conv with no padding
            self.conv = WrappedConv2d(channels, channels, 3, stride=2, padding=0, causal=False)
        else:
            self.conv = WrappedConv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        if self._causal:
            # Causal: pad (2, 0) on height, (0, 1) on width
            x = mx.pad(x, [(0, 0), (2, 0), (0, 1), (0, 0)])
        return self.conv(x)


class AudioDownBlock(nn.Module):
    """One encoder down-stage: N resblocks + optional downsample.

    Key prefix: ``down.<idx>.``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        add_downsample: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.block = [
            AudioResBlock(in_channels if i == 0 else out_channels, out_channels, causal=causal)
            for i in range(num_blocks)
        ]
        self.downsample = AudioDownsample(out_channels, causal=causal) if add_downsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for blk in self.block:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class AudioVAEEncoder(nn.Module):
    """Audio VAE encoder: mel (B, 2, T', 64) -> latent (B, 8, T, 16).

    Encodes mel spectrograms to compressed audio latents.
    Architecture mirrors the decoder in reverse.
    """

    def __init__(self):
        super().__init__()

        # Input: 2 channels (stereo mel), frequency=64
        self.conv_in = WrappedConv2d(2, 128, 3, padding=1, causal=True)

        # Down blocks (mirror of up blocks in decoder)
        self.down = [
            AudioDownBlock(128, 256, num_blocks=3, add_downsample=True, causal=True),  # down.0: freq 64→32
            AudioDownBlock(256, 512, num_blocks=3, add_downsample=True, causal=True),  # down.1: freq 32→16
            AudioDownBlock(512, 512, num_blocks=3, add_downsample=False, causal=True),  # down.2: no downsample
        ]

        # Mid
        from ltx_core_mlx.model.audio_vae.audio_vae import AudioMidBlock

        self.mid = AudioMidBlock(512, causal=True, add_attention=False)

        # Output: 8 channels (latent C1 dim)
        self.conv_out = WrappedConv2d(512, 8, 3, padding=1, causal=True)

        # Per-channel normalization stats
        self.per_channel_statistics = PerChannelStatistics(128)

    def encode(self, mel: mx.array) -> mx.array:
        """Encode mel spectrogram to audio latent.

        Args:
            mel: (B, 2, T', 64) stereo mel spectrogram.

        Returns:
            Latent (B, 8, T, 16).
        """
        B, C, T_mel, M = mel.shape  # (B, 2, T', 64)

        # Convert to NHWC: (B, T', 64, 2) — T'=height, 64=width (freq), 2=channels
        x = mel.transpose(0, 2, 3, 1)

        x = self.conv_in(x)

        for blk in self.down:
            x = blk(x)

        x = self.mid(x)

        x = pixel_norm(x)  # norm_out
        x = nn.silu(x)
        x = self.conv_out(x)  # (B, T, 16, 8) in NHWC

        # Flatten to (B, T, 128) for normalization
        B2, T, W, C_out = x.shape
        x_flat = x.reshape(B2, T, W * C_out)  # (B, T, 128)

        # Normalize using per-channel statistics
        mean = self.per_channel_statistics.mean_of_means.reshape(1, 1, -1)
        std = self.per_channel_statistics.std_of_means.reshape(1, 1, -1)
        x_flat = (x_flat - mean) / (std + 1e-8)

        # Reshape: (B, T, 128) -> (B, T, 16, 8) -> (B, 8, T, 16)
        x = x_flat.reshape(B2, T, 16, 8)
        return x.transpose(0, 3, 1, 2)
