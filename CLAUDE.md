# CLAUDE.md — ltx-2-mlx

## Project Overview

Pure MLX port of [ltx-core](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core) and [ltx-pipelines](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines) (Lightricks) for Apple Silicon. Two-package monorepo mirroring the reference structure:

- **ltx-core-mlx** (`ltx_core_mlx`) — model library: DiT, VAE, audio, text encoder, conditioning
- **ltx-pipelines-mlx** (`ltx_pipelines_mlx`) — generation pipelines: T2V, I2V, retake, extend, keyframe, two-stage

Loads pre-converted MLX weights from the [LTX-2.3 MLX collection on HuggingFace](https://huggingface.co/collections/dgrauet/ltx-23). Weight conversion is handled by [mlx-forge](https://github.com/dgrauet/mlx-forge).

---

## Tech Stack

- Python 3.11+, `uv` workspace (monorepo with `packages/*`)
- MLX (`mlx>=0.31.0`) — Apple Silicon ML framework (unified CPU/GPU memory)
- `mlx-lm>=0.31.0` — for Gemma 3 text encoder loading
- `safetensors`, `huggingface-hub`, `numpy`
- Linter/formatter: ruff

---

## Architecture

```
packages/
├── ltx-core-mlx/                          # ltx_core_mlx
│   └── src/ltx_core_mlx/
│       ├── conditioning/                   # Latent conditioning system
│       │   ├── mask_utils.py               # build/update/resolve attention masks
│       │   └── types/
│       │       ├── latent_cond.py          # LatentState, VideoConditionByLatentIndex
│       │       ├── keyframe_cond.py        # VideoConditionByKeyframeIndex
│       │       └── reference_video_cond.py # VideoConditionByReferenceLatent (IC-LoRA)
│       │
│       ├── model/
│       │   ├── audio_vae/                  # Audio VAE + vocoder + BWE
│       │   │   ├── audio_vae.py            # AudioVAEDecoder, AudioResBlock, AudioAttnBlock
│       │   │   ├── encoder.py              # AudioVAEEncoder
│       │   │   ├── vocoder.py              # BigVGANVocoder, SnakeBeta, Activation1d
│       │   │   ├── bwe.py                  # VocoderWithBWE, HannSincResampler, MelSTFT
│       │   │   └── processor.py            # AudioProcessor (STFT + mel filterbank)
│       │   │
│       │   ├── transformer/                # Diffusion Transformer (DiT)
│       │   │   ├── model.py                # LTXModel, X0Model, LTXModelConfig
│       │   │   ├── transformer.py          # BasicAVTransformerBlock (joint audio+video)
│       │   │   ├── attention.py            # Multi-head attention + RoPE + per-head gating
│       │   │   ├── feed_forward.py         # Gated MLP blocks
│       │   │   ├── rope.py                 # Rotary position embeddings (SPLIT type)
│       │   │   ├── adaln.py                # AdaLayerNormSingle (9-param)
│       │   │   └── timestep_embedding.py   # Sinusoidal + MLP timestep encoding
│       │   │
│       │   ├── upsampler/                  # Neural latent upscaler
│       │   │   └── model.py                # LatentUpsampler, SpatialRationalResampler
│       │   │
│       │   └── video_vae/                  # Video VAE
│       │       ├── video_vae.py            # VideoDecoder (streaming), VideoEncoder
│       │       ├── convolution.py          # Conv3dBlock (causal + reflect padding)
│       │       ├── resnet.py               # ResBlock3d, ResBlockStage
│       │       ├── sampling.py             # DepthToSpaceUpsample, pixel_shuffle_3d
│       │       ├── patchifier.py           # VideoLatentPatchifier, AudioPatchifier
│       │       ├── normalization.py        # pixel_norm (RMS)
│       │       └── ops.py                  # PerChannelStatistics
│       │
│       ├── text_encoders/                  # Text encoding (Gemma 3)
│       │   └── gemma/
│       │       ├── language_model.py        # Gemma 3 12B wrapper via mlx-lm
│       │       ├── embeddings_connector.py  # Embeddings1DConnector (RoPE + registers)
│       │       └── feature_extractor.py     # GemmaFeaturesExtractorV2 (video/audio projections)
│       │
│       └── utils/
│           ├── positions.py    # compute_video_positions, compute_audio_positions
│           ├── weights.py      # load_split_safetensors, apply_quantization
│           ├── memory.py       # aggressive_cleanup, get_memory_stats
│           ├── image.py        # prepare_image_for_encoding
│           └── ffmpeg.py       # find_ffmpeg, probe_video_info
│
└── ltx-pipelines-mlx/                     # ltx_pipelines_mlx
    └── src/ltx_pipelines_mlx/
        ├── text_to_video.py    # T2V: prompt → video+audio
        ├── image_to_video.py   # I2V: image + prompt → video+audio
        ├── retake.py           # Retake: regenerate a time segment
        ├── extend.py           # Extend: add frames before/after
        ├── keyframe_interp.py  # Keyframe interpolation
        ├── two_stage.py        # Two-stage: half res → upscale → refine
        ├── denoise.py          # Euler denoising loop (joint audio+video)
        ├── scheduler.py        # DISTILLED_SIGMAS, STAGE_2_SIGMAS
        └── cli.py              # CLI entry point
```

---

## LTX-2.3 Model Architecture

- **Type**: Diffusion Transformer (DiT), 19B params, joint audio+video single-pass
- **Transformer**: 48 layers × 32 heads × 128-dim = 4096-dim (video), 32 heads × 64-dim = 2048-dim (audio)
- **VAE**: Temporal 8×, Spatial 32× compression → 128-channel latent
- **Text encoder**: Gemma 3 12B → dual projections (video 4096-dim, audio 2048-dim) via Embeddings1DConnector
- **Vocoder**: BigVGAN v2 with SnakeBeta activation (log-scale alpha/beta) + anti-aliased resampling
- **BWE**: Residual bandwidth extension (base 16kHz → Hann-sinc 3× resample → causal MelSTFT → BWE generator → 48kHz)
- **Distilled**: 8 steps (predefined sigma schedule), no classifier-free guidance

### Key Shapes

| Component | Input | Output |
|-----------|-------|--------|
| Text encoder | token_ids (1, 1024) | video_embeds (1, 1024, 4096), audio_embeds (1, 1024, 2048) |
| Transformer (video) | latent (B, F×H×W, 128) | velocity (B, F×H×W, 128) |
| Transformer (audio) | latent (B, T, 128) | velocity (B, T, 128) |
| Video VAE decoder | latent (B, 128, F', H', W') | pixels (B, 3, F, H, W) |
| Audio VAE decoder | latent (B, 8, T, 16) | mel (B, 2, T', 64) |
| Vocoder | mel (B, 2, T', 64) | waveform (B, 2, T_audio) @ 16kHz |
| BWE | waveform 16kHz | waveform 48kHz |
| Upsampler | latent (B, 128, F, H, W) | latent (B, 128, F, 2H, 2W) |

### Audio Token Count

Audio tokens per video: `round(num_pixel_frames / fps * 25)` where 25 = sample_rate(16000) / hop_length(160) / downsample_factor(4).

---

## Weight Format

Weights are pre-converted by [mlx-forge](https://github.com/dgrauet/mlx-forge) and hosted on HuggingFace. This package only **loads** weights — it never converts them.

### Available Variants

| Variant | HuggingFace | Size | Notes |
|---------|-------------|------|-------|
| Distilled bf16 | [dgrauet/ltx-2.3-mlx-distilled](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled) | ~42GB | Full precision, requires 64GB+ RAM |
| Distilled int8 | [dgrauet/ltx-2.3-mlx-distilled-q8](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled-q8) | ~21GB | Recommended for 32GB+ |
| Distilled int4 | [dgrauet/ltx-2.3-mlx-distilled-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled-q4) | ~12GB | Lower quality, fits 16GB |

### MLX Layout Conventions

| Layer Type | PyTorch | MLX | Notes |
|-----------|---------|-----|-------|
| Linear | (O, I) | (O, I) | No transpose |
| Conv1d | (O, I, K) | (O, K, I) | Pre-converted by mlx-forge |
| Conv2d | (O, I, H, W) | (O, H, W, I) | Pre-converted |
| Conv3d | (O, I, D, H, W) | (O, D, H, W, I) | Pre-converted |
| ConvTranspose1d | (I, O, K) | (O, K, I) | Pre-converted by mlx-forge |
| Norm layers | (D,) | (D,) | No transpose |

**All weights must be in MLX format on disk.** If a weight file contains PyTorch-format tensors, fix it in mlx-forge — don't work around it here.

### Quantization

- Only `nn.Linear` inside `transformer_blocks` → int8 (group_size=64)
- Non-quantizable (must stay bf16): `adaln_single`, `proj_out`, `patchify_proj`, connectors, VAE, vocoder
- MLX can only quantize Linear and Embedding — never Conv layers

### Split Safetensors

| File | Prefix | Content |
|------|--------|---------|
| `transformer.safetensors` | `transformer.` | DiT blocks (quantized) |
| `connector.safetensors` | N/A | Text embeddings connectors |
| `vae_decoder.safetensors` | `vae_decoder.` | Video VAE decoder + per-channel stats |
| `vae_encoder.safetensors` | `vae_encoder.` | Video VAE encoder + per-channel stats |
| `audio_vae.safetensors` | `audio_vae.` | Audio VAE decoder + per-channel stats |
| `vocoder.safetensors` | `vocoder.` | Base vocoder + BWE generator + mel STFT |

---

## Critical Rules

### 1. Metal Memory Management (NON-NEGOTIABLE)

```python
from ltx_core_mlx.utils.memory import aggressive_cleanup
aggressive_cleanup()  # gc.collect() + mx.clear_cache()
```

Call between **every pipeline stage**. MLX Metal cache grows unbounded without explicit cleanup.

### 2. Streaming VAE Decode (NON-NEGOTIABLE)

Never decode all video frames in RAM. Stream frame-by-frame to ffmpeg:

```python
for i in range(num_frames):
    frame = decoder.decode_frame(latents[:, :, i:i+1])
    ffmpeg_proc.stdin.write(frame_to_bytes(frame))
    del frame
    if i % 8 == 0:
        aggressive_cleanup()
```

### 3. Reference Implementation is ltx-core

**ALWAYS** port from [ltx-core](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core) (Lightricks official), NOT from mlx-video.

Key reference paths:
- `packages/ltx-core/src/ltx_core/model/transformer/` — DiT architecture
- `packages/ltx-core/src/ltx_core/model/audio_vae/` — Audio VAE + vocoder + BWE
- `packages/ltx-core/src/ltx_core/model/video_vae/` — Video VAE
- `packages/ltx-core/src/ltx_core/conditioning/` — Conditioning system
- `packages/ltx-core/src/ltx_core/components/` — Schedulers, patchifiers, guiders
- `packages/ltx-pipelines/src/ltx_pipelines/` — Pipeline implementations

### 4. No Weight Conversion in This Package

Weight conversion is handled by [mlx-forge](https://github.com/dgrauet/mlx-forge). This package loads pre-converted weights only.

### 5. Positions Must Be in Pixel-Space

Video positions use pixel-space coordinates with causal fix, divided by fps:
- Temporal: `midpoint(max(0, i*8 - 7), i*8 + 1) / fps`
- Spatial: `h * 32 + 16`, `w * 32 + 16`

Audio positions use real-time seconds: `midpoint(max(0, (i-3)*4) * 0.01, max(0, (i-2)*4) * 0.01)`

Never use raw latent indices as positions.

### 6. Per-Token Timesteps for Conditioning

When conditioning (I2V, retake, extend), use per-token timesteps `sigma * denoise_mask`:
- X0Model denoising: `x0 = x_t - per_token_sigma * v` (preserved tokens get sigma=0 → x0=x_t)
- AdaLN: reshape per-token params as `(B, N, P, dim)` not `(B*N, P, dim)`

---

## Conditioning System

### Core Types
- `LatentState(latent, clean_latent, denoise_mask, positions?, attention_mask?)` — generation state
- `denoise_mask`: `1.0` = denoise (generate), `0.0` = preserve (keep clean)
- `positions`: (B, N, num_axes) pixel-space positions for RoPE
- `attention_mask`: (B, N, N) optional self-attention mask [0,1]

### Conditioning Items
- `VideoConditionByLatentIndex(frame_indices, clean_latent, strength)` — replace tokens at frame index (I2V)
- `VideoConditionByKeyframeIndex(indices, latents, positions, strength)` — append tokens (interpolation)
- `VideoConditionByReferenceLatent(latent, positions, downscale_factor, strength)` — append reference (IC-LoRA)
- `TemporalRegionMask(start_frame, end_frame)` — time-range masking (retake)

### Attention Mask System
- `mask_utils.build_attention_mask()` — block-structured (B, N+M, N+M) mask
- `mask_utils.update_attention_mask()` — incremental mask building for conditioning items
- Conditioning items call `update_attention_mask` when appending tokens

### Diffusion Loop
```python
# denoise_loop resolves positions/attention_mask from LatentState automatically
# Per-step: video_timesteps = sigma * denoise_mask (preserved regions get sigma=0)
# Per-step: x0 = apply_denoise_mask(x0, clean_latent, mask) → blend before Euler step
# Noising: noise_latent_state() blends clean*(1-mask) + noisy*mask
```

---

## Audio Pipeline

### Full Chain
```
Audio latent (B, 8, T, 16)
    → Audio VAE decoder (causal Conv2d + PixelNorm + AttnBlock) → mel (B, 2, T', 64)
    → BigVGAN v2 vocoder (SnakeBeta log-scale + anti-aliased) → waveform @ 16kHz
    → BWE (Hann-sinc 3× resample + causal MelSTFT + BigVGAN residual) → waveform @ 48kHz
```

### Key Implementation Details
- **SnakeBeta**: weights stored in log-scale, forward applies `exp(alpha)` and `exp(beta)`
- **Audio VAE Conv2d**: causal padding on height axis (time), reflect padding NOT used (zeros)
- **Audio VAE upsample**: drop first row after causal conv for temporal alignment
- **BWE resampler**: Hann-windowed sinc, 43 taps, rolloff=0.99 (NOT Kaiser)
- **BWE MelSTFT**: causal left-only padding (352, 0), NOT symmetric
- **BWE generator**: `apply_final_activation=False` (no tanh on residual)

---

## Conventions

- Python 3.11+
- Mandatory type hints on all functions
- Google-style docstrings
- ruff for formatting/linting
- Tests in `tests/` using pytest
- Conventional commits (feat:, fix:, docs:, refactor:)
- Package imports: `ltx_core_mlx.*` for core, `ltx_pipelines_mlx.*` for pipelines

---

## Resources

- **ltx-core**: [GitHub](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core)
- **ltx-pipelines**: [GitHub](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines)
- **MLX**: [Docs](https://ml-explore.github.io/mlx/) · [GitHub](https://github.com/ml-explore/mlx)
- **mlx-forge**: [GitHub](https://github.com/dgrauet/mlx-forge) — weight conversion
- **Pre-converted weights**: [HuggingFace collection](https://huggingface.co/collections/dgrauet/ltx-23)
