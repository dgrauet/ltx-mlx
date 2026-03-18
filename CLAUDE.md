# CLAUDE.md — ltx-mlx

## Project Overview

Pure MLX port of [ltx-core](https://github.com/Lightricks/LTX-2) (Lightricks) for Apple Silicon. Provides the complete LTX-2.3 inference pipeline — transformer, VAE, audio, vocoder, conditioning — as a standalone Python package.

Loads pre-converted MLX weights from the [LTX-2.3 MLX collection on HuggingFace](https://huggingface.co/collections/dgrauet/ltx-23). Weight conversion is handled by [mlx-forge](https://github.com/dgrauet/mlx-forge).

---

## Tech Stack

- Python 3.11+, `uv` package manager
- MLX (`mlx>=0.31.0`) — Apple Silicon ML framework (unified CPU/GPU memory)
- `mlx-lm>=0.31.0` — for Gemma 3 text encoder loading
- `safetensors`, `huggingface-hub`, `numpy`
- Linter/formatter: ruff

---

## Architecture

```
src/ltx_mlx/
├── model/                    # Diffusion Transformer (DiT)
│   ├── model.py              # LTXModel, X0Model, Modality dataclass
│   ├── transformer.py        # BasicAVTransformerBlock (joint audio+video)
│   ├── attention.py          # Multi-head attention + RoPE + per-head gating
│   ├── feed_forward.py       # Gated MLP blocks
│   ├── rope.py               # Rotary position embeddings (SPLIT type)
│   └── timestep_embedding.py # AdaLayerNormSingle (6-param V1, 9-param V2)
│
├── vae/                      # Video VAE
│   ├── decoder.py            # VideoDecoder — streaming frame-by-frame to ffmpeg
│   ├── encoder.py            # VideoEncoder — image/video → latent
│   └── patchifier.py         # VideoLatentPatchifier, AudioPatchifier, shape utils
│
├── audio/                    # Audio pipeline
│   ├── decoder.py            # Audio VAE decoder: latent (B,8,T,16) → mel (B,2,T',64)
│   ├── encoder.py            # Audio VAE encoder: mel → latent
│   ├── processor.py          # AudioProcessor: STFT + mel filterbank
│   ├── vocoder.py            # BigVGAN v2: mel → waveform 16kHz
│   └── bwe.py                # Bandwidth Extension: 16kHz → 48kHz (VocoderWithBWE)
│
├── text_encoder/             # Text encoding (Gemma 3)
│   ├── language_model.py     # Gemma 3 12B wrapper via mlx-lm
│   ├── connector.py          # Embeddings1DConnector — RoPE refinement, learnable registers
│   └── feature_extractor.py  # GemmaFeaturesExtractorV2 — separate video/audio projections
│
├── conditioning/             # Latent conditioning system
│   ├── latent.py             # LatentState, VideoConditionByLatentIndex, core functions
│   ├── keyframe.py           # VideoConditionByKeyframeIndex (appended tokens + attn mask)
│   └── reference.py          # VideoConditionByReferenceLatent (IC-LoRA)
│
├── pipeline/                 # Generation pipelines
│   ├── denoise.py            # Euler denoising loop (joint audio+video)
│   ├── generate.py           # High-level generate() API
│   └── scheduler.py          # Sigma schedules (DISTILLED_SIGMAS, STAGE_2_SIGMAS)
│
├── upsampler/                # Neural latent upscaler
│   └── upsampler.py          # LatentUpsampler (Conv3d ResBlocks + 2× spatial)
│
└── utils/                    # Shared utilities
    ├── ffmpeg.py             # find_ffmpeg, find_ffprobe, probe_video_info, has_audio_stream
    ├── image.py              # prepare_image_for_encoding (PIL → tensor)
    ├── weights.py            # load_split_safetensors, apply_quantization
    └── memory.py             # aggressive_cleanup, get_memory_stats
```

---

## LTX-2.3 Model Architecture

- **Type**: Diffusion Transformer (DiT), 19B params, joint audio+video single-pass
- **Transformer**: 48 layers × 32 heads × 128-dim = 4096-dim (video), 32 heads × 64-dim = 2048-dim (audio)
- **VAE**: Temporal 8×, Spatial 32× compression → 128-channel latent
- **Text encoder**: Gemma 3 12B → dual projections (video 4096-dim, audio 2048-dim) via Embeddings1DConnector
- **Vocoder**: BigVGAN v2 with SnakeBeta activation + anti-aliased resampling
- **BWE**: Residual bandwidth extension (base 16kHz → upsample 3× → mel → BWE generator → 48kHz)
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
import gc
import mlx.core as mx

def aggressive_cleanup():
    gc.collect()
    mx.clear_cache()
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
- `ltx-core/src/ltx_core/model/transformer/` — DiT architecture
- `ltx-core/src/ltx_core/model/audio_vae/` — Audio VAE + vocoder + BWE
- `ltx-core/src/ltx_core/model/video_vae/` — Video VAE
- `ltx-core/src/ltx_core/conditioning/` — Conditioning system
- `ltx-core/src/ltx_core/components/` — Schedulers, patchifiers, guiders
- `ltx-pipelines/src/ltx_pipelines/` — Pipeline implementations

### 4. No Weight Conversion in This Package

Weight conversion is handled by [mlx-forge](https://github.com/dgrauet/mlx-forge). This package loads pre-converted weights only.

---

## Audio Pipeline

### Full Chain
```
Audio latent (B, 8, T, 16)
    → Audio VAE decoder → mel (B, 2, T', 64)
    → BigVGAN v2 vocoder → waveform @ 16kHz
    → BWE → waveform @ 48kHz
```

### Vocoder (Base, 16kHz)
- Upsample ratios: `[5, 2, 2, 2, 2, 2]` → 160× (hop_length=160 at 16kHz)
- Kernel sizes: `[11, 4, 4, 4, 4, 4]`
- Activation: SnakeBeta + anti-aliased resampling (Activation1d)
- 3 × AMPBlock1 per upsample stage

### BWE (16kHz → 48kHz)
- Resampler: 3× upsample (kaiser-sinc)
- BWE generator: separate BigVGAN with ratios `[6, 5, 2, 2, 2]` (240×)
- MelSTFT: `mel_basis (64, 257)`, STFT basis `(514, 1, 512)`
- Output: `clamp(resampled_base + bwe_residual, -1, 1)`

---

## Conditioning System

### Core Types
- `LatentState(latent, clean_latent, denoise_mask)` — generation state
- `denoise_mask`: `1.0` = denoise (generate), `0.0` = preserve (keep clean)

### Conditioning Items
- `VideoConditionByLatentIndex` — replace tokens at frame index (I2V)
- `VideoConditionByKeyframeIndex` — append tokens with attention mask (interpolation)
- `VideoConditionByReferenceLatent` — append reference with scaled positions (IC-LoRA)
- `TemporalRegionMask` — time-range masking (retake)

### Diffusion Loop
```python
# Per-step: timesteps = sigma × denoise_mask (preserved regions get sigma=0)
# Per-step: x0 = apply_denoise_mask(x0, clean_latent, mask) → blend before Euler step
```

---

## Hardware Notes

- **Minimum**: 32GB Apple Silicon
- **Model weights**: ~21GB int8 (dominant cost)
- Text encoder (Gemma 3 12B, ~8.7GB peak) and transformer (~12.6GB peak) cannot coexist on 32GB — callers should load/unload sequentially

---

## Conventions

- Python 3.11+
- Mandatory type hints on all functions
- Google-style docstrings
- ruff for formatting/linting
- Tests in `tests/` using pytest
- Conventional commits (feat:, fix:, docs:, refactor:)

---

## Resources

- **ltx-core**: [GitHub](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core)
- **ltx-pipelines**: [GitHub](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines)
- **MLX**: [Docs](https://ml-explore.github.io/mlx/) · [GitHub](https://github.com/ml-explore/mlx)
- **mlx-forge**: [GitHub](https://github.com/dgrauet/mlx-forge) — weight conversion
- **Pre-converted weights**: [HuggingFace](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled-q8)
