"""Command-line interface for ltx-2-mlx.

Usage:
    ltx-2-mlx generate --prompt "a cat walking" --output out.mp4
    ltx-2-mlx generate --prompt "animate this" --image photo.jpg --output out.mp4
    ltx-2-mlx info --model dgrauet/ltx-2.3-mlx-distilled-q8
"""

from __future__ import annotations

import argparse
import sys
import time

DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-distilled-q8"
DEFAULT_GEMMA = "mlx-community/gemma-3-12b-it-4bit"


def main() -> None:
    """Entry point for the ltx-2-mlx CLI."""
    parser = argparse.ArgumentParser(
        prog="ltx-2-mlx",
        description="LTX-2.3 video generation on Apple Silicon (MLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  ltx-2-mlx generate --prompt "a sunset over the ocean" --output sunset.mp4
  ltx-2-mlx generate --prompt "animate" --image photo.jpg -o anim.mp4
  ltx-2-mlx generate --prompt "a cat" -o cat.mp4 --height 320 --width 512 --frames 25
  ltx-2-mlx info --model dgrauet/ltx-2.3-mlx-distilled-q4
""",
    )
    sub = parser.add_subparsers(dest="command")

    # --- generate ---
    gen = sub.add_parser("generate", help="Generate video from text (T2V) or image (I2V)")
    gen.add_argument("--prompt", "-p", required=True, help="Text prompt")
    gen.add_argument("--output", "-o", required=True, help="Output video path (.mp4)")
    gen.add_argument("--image", "-i", default=None, help="Reference image for I2V (optional)")
    gen.add_argument(
        "--model", "-m", default=DEFAULT_MODEL, help=f"Model weights (HF repo or path, default: {DEFAULT_MODEL})"
    )
    gen.add_argument("--gemma", default=DEFAULT_GEMMA, help=f"Gemma model for text encoding (default: {DEFAULT_GEMMA})")
    gen.add_argument("--height", "-H", type=int, default=480, help="Video height (default: 480)")
    gen.add_argument("--width", "-W", type=int, default=704, help="Video width (default: 704)")
    gen.add_argument("--frames", "-f", type=int, default=97, help="Number of frames (default: 97)")
    gen.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    gen.add_argument("--steps", type=int, default=None, help="Denoising steps (default: 8)")
    gen.add_argument("--no-audio", action="store_true", help="Skip audio generation")
    gen.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    # --- info ---
    info = sub.add_parser("info", help="Show model info and memory estimate")
    info.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model weights (HF repo or path)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "info":
        _cmd_info(args)


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate a video from a text prompt (and optionally a reference image)."""
    t0 = time.time()

    if args.image:
        from ltx_pipelines_mlx.image_to_video import ImageToVideoPipeline

        if not args.quiet:
            print("Mode: Image-to-Video")
            print(f"Image: {args.image}")

        pipe = ImageToVideoPipeline(model_dir=args.model, gemma_model_id=args.gemma)
        output = pipe.generate_and_save(
            prompt=args.prompt,
            output_path=args.output,
            image=args.image,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            seed=args.seed,
            num_steps=args.steps,
        )
    else:
        from ltx_pipelines_mlx.text_to_video import TextToVideoPipeline

        if not args.quiet:
            print("Mode: Text-to-Video")

        pipe = TextToVideoPipeline(model_dir=args.model, gemma_model_id=args.gemma)
        output = pipe.generate_and_save(
            prompt=args.prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            seed=args.seed,
            num_steps=args.steps,
        )

    elapsed = time.time() - t0

    if not args.quiet:
        print(f"\nSaved to: {output}")
        print(f"Time: {elapsed:.1f}s")


def _cmd_info(args: argparse.Namespace) -> None:
    """Show model info and memory estimate."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    model_dir = Path(args.model)
    if not model_dir.exists():
        try:
            model_dir = Path(snapshot_download(args.model))
        except Exception as e:
            print(f"Could not find or download model: {args.model}")
            print(f"  {e}")
            sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Path:  {model_dir}")
    print()

    # Weight files
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        print("  No .safetensors files found.")
        return

    total_bytes = 0
    for f in safetensor_files:
        size = f.stat().st_size
        total_bytes += size
        print(f"  {f.name:<45s} {size / 1024**2:>8.1f} MB")

    total_mb = total_bytes / 1024**2
    total_gb = total_mb / 1024
    print(f"  {'─' * 55}")
    print(f"  {'Total':<45s} {total_mb:>8.1f} MB ({total_gb:.1f} GB)")
    print(f"  Estimated RAM: ~{total_gb * 1.3:.0f} GB (model + inference overhead)")

    # Config files
    json_files = sorted(model_dir.glob("*.json"))
    if json_files:
        print(f"\n  Config files: {', '.join(f.name for f in json_files)}")

    # Upsampler info
    upsampler_files = [f for f in safetensor_files if "upscaler" in f.name or "upsampler" in f.name]
    if upsampler_files:
        print(f"\n  Upsamplers: {', '.join(f.stem for f in upsampler_files)}")


if __name__ == "__main__":
    main()
