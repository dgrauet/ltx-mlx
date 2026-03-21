# ltx-2-mlx

Pure MLX port of [LTX-2.3](https://github.com/Lightricks/LTX-2) for Apple Silicon. Complete inference stack — 19B Diffusion Transformer, Video/Audio VAE, Gemma 3 text encoder, and generation pipelines — as a standalone Python package.

## Features

- **Text-to-Video** — generate video + stereo audio from a text prompt
- **Image-to-Video** — animate a reference image
- **Retake / Extend / Keyframe** — edit existing videos
- **Two-stage generation** — half-res → neural upscale → refine
- **3 model variants** — bf16, int8, int4 (fits 16GB–64GB Macs)
- **3 upsamplers** — spatial 2×, spatial 1.5×, temporal 2×

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 32GB+ RAM recommended (int8 model), 16GB minimum (int4)
- ffmpeg (for video encoding)

## Installation

```bash
pip install ltx-2-mlx
```

Or for development:

```bash
git clone https://github.com/dgrauet/ltx-2-mlx.git
cd ltx-2-mlx
uv pip install -e ".[dev]"
```

## Quick Start

### CLI

```bash
# Text-to-Video
ltx-2-mlx generate --prompt "A sunset over the ocean with waves" --output sunset.mp4

# Image-to-Video
ltx-2-mlx generate --prompt "Animate this scene" --image photo.jpg --output animated.mp4

# Custom resolution and duration
ltx-2-mlx generate -p "A cat walking" -o cat.mp4 --height 320 --width 512 --frames 41

# Use the lightweight int4 model (fits 16GB)
ltx-2-mlx generate -p "A cat walking" -o cat.mp4 --model dgrauet/ltx-2.3-mlx-distilled-q4

# Show model info
ltx-2-mlx info --model dgrauet/ltx-2.3-mlx-distilled-q8
```

Also works via `python -m ltx_2_mlx`:

```bash
python -m ltx_2_mlx generate -p "A sunset" -o sunset.mp4
```

### Python API

```python
from ltx_2_mlx.pipeline import TextToVideoPipeline

pipe = TextToVideoPipeline(model_dir="dgrauet/ltx-2.3-mlx-distilled-q8")
pipe.generate_and_save(
    prompt="A sunset over the ocean with waves crashing",
    output_path="sunset.mp4",
    height=480,
    width=704,
    num_frames=97,
    seed=42,
)
```

Image-to-Video:

```python
from ltx_2_mlx.pipeline import ImageToVideoPipeline

pipe = ImageToVideoPipeline(model_dir="dgrauet/ltx-2.3-mlx-distilled-q8")
pipe.generate_and_save(
    prompt="Animate this scene with gentle motion",
    output_path="animated.mp4",
    image="photo.jpg",
)
```

## CLI Reference

```
ltx-2-mlx generate [options]
  --prompt, -p    Text prompt (required)
  --output, -o    Output .mp4 path (required)
  --image, -i     Reference image for I2V (optional)
  --model, -m     Model weights — HF repo or local path
                  (default: dgrauet/ltx-2.3-mlx-distilled-q8)
  --gemma         Gemma text encoder model
                  (default: mlx-community/gemma-3-12b-it-4bit)
  --height, -H    Video height in pixels (default: 480)
  --width, -W     Video width in pixels (default: 704)
  --frames, -f    Number of frames (default: 97)
  --seed, -s      Random seed (default: 42)
  --steps         Denoising steps (default: 8)
  --no-audio      Skip audio generation
  --quiet, -q     Suppress progress output

ltx-2-mlx info [options]
  --model, -m     Model weights to inspect
```

## Pre-converted Weights

| Variant | HuggingFace | Size | RAM |
|---------|-------------|------|-----|
| bf16 | [dgrauet/ltx-2.3-mlx-distilled](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled) | ~42 GB | 64 GB+ |
| int8 | [dgrauet/ltx-2.3-mlx-distilled-q8](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled-q8) | ~21 GB | 32 GB+ |
| int4 | [dgrauet/ltx-2.3-mlx-distilled-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled-q4) | ~12 GB | 16 GB+ |

Weights are pre-converted to MLX format by [mlx-forge](https://github.com/dgrauet/mlx-forge). Each variant includes the full stack: transformer, VAE encoder/decoder, audio VAE, vocoder + BWE, text connector, and three latent upsamplers.

## Architecture

```
src/ltx_2_mlx/
├── model/          # 19B DiT — 48 transformer blocks, joint audio+video
├── vae/            # Video VAE encoder/decoder (8× temporal, 32× spatial)
├── audio/          # Audio VAE + BigVGAN vocoder + BWE (48kHz stereo)
├── text_encoder/   # Gemma 3 12B → dual connector (video 4096-d, audio 2048-d)
├── conditioning/   # I2V, retake, extend, keyframe conditioning
├── pipeline/       # T2V, I2V, retake, extend, keyframe, two-stage pipelines
├── upsampler/      # Neural latent upsamplers (spatial 2×/1.5×, temporal 2×)
├── utils/          # FFmpeg, weight loading, memory management
└── cli.py          # Command-line interface
```

## Resources

- [LTX-2](https://github.com/Lightricks/LTX-2) — Lightricks original (ltx-core + ltx-pipelines)
- [mlx-forge](https://github.com/dgrauet/mlx-forge) — weight conversion tool
- [Pre-converted weights](https://huggingface.co/collections/dgrauet/ltx-23) — HuggingFace collection
- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework

## License

MIT
