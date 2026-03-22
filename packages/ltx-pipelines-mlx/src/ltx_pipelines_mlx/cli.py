"""Command-line interface for ltx-2-mlx.

Usage:
    ltx-2-mlx generate --prompt "a cat walking" --output out.mp4
    ltx-2-mlx generate --prompt "animate this" --image photo.jpg --output out.mp4
    ltx-2-mlx generate --prompt "a scene" --two-stage --output hires.mp4
    ltx-2-mlx generate --prompt "a scene" --hq --output hq.mp4
    ltx-2-mlx enhance --prompt "a cat" --mode t2v
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
  ltx-2-mlx generate --prompt "a scene" --two-stage -o hires.mp4
  ltx-2-mlx generate --prompt "a scene" --hq --stage1-steps 20 -o hq.mp4
  ltx-2-mlx enhance --prompt "a cat walking" --mode t2v
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
    gen.add_argument("--two-stage", action="store_true", help="Use two-stage pipeline (half-res + upscale + refine)")
    gen.add_argument("--hq", action="store_true", help="Use HQ pipeline (res_2s sampler + upscale + refine)")
    gen.add_argument("--stage1-steps", type=int, default=None, help="Stage 1 steps for two-stage/HQ mode")
    gen.add_argument("--stage2-steps", type=int, default=None, help="Stage 2 steps for two-stage/HQ mode")
    gen.add_argument("--enhance-prompt", action="store_true", help="Enhance prompt using Gemma before generation")

    # --- enhance ---
    enh = sub.add_parser("enhance", help="Enhance a prompt using Gemma (no video generation)")
    enh.add_argument("--prompt", "-p", required=True, help="Prompt to enhance")
    enh.add_argument("--mode", choices=["t2v", "i2v"], default="t2v", help="Prompt mode (default: t2v)")
    enh.add_argument("--gemma", default=DEFAULT_GEMMA, help=f"Gemma model (default: {DEFAULT_GEMMA})")
    enh.add_argument("--seed", "-s", type=int, default=10, help="Random seed (default: 10)")

    # --- info ---
    info = sub.add_parser("info", help="Show model info and memory estimate")
    info.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model weights (HF repo or path)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "enhance":
        _cmd_enhance(args)
    elif args.command == "info":
        _cmd_info(args)


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate a video from a text prompt (and optionally a reference image)."""
    t0 = time.time()

    # Enhance prompt if requested
    prompt = args.prompt
    if args.enhance_prompt:
        from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

        if not args.quiet:
            print("Enhancing prompt...")
        gemma = GemmaLanguageModel()
        gemma.load(args.gemma)
        if args.image:
            prompt = gemma.enhance_i2v(prompt, seed=args.seed)
        else:
            prompt = gemma.enhance_t2v(prompt, seed=args.seed)
        if not args.quiet:
            print(f"Enhanced: {prompt[:200]}...")
        del gemma
        from ltx_core_mlx.utils.memory import aggressive_cleanup

        aggressive_cleanup()

    if args.hq:
        from ltx_pipelines_mlx.ti2vid_two_stages_hq import TwoStageHQPipeline

        if not args.quiet:
            print("Mode: HQ Two-Stage (res_2s)")

        pipe = TwoStageHQPipeline(model_dir=args.model, low_memory=True)
        video_latent, audio_latent = pipe.generate_hq(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            seed=args.seed,
            stage1_steps=args.stage1_steps or 20,
            stage2_steps=args.stage2_steps,
            image=args.image,
        )
        _decode_and_save(pipe, video_latent, audio_latent, args)

    elif args.two_stage:
        from ltx_pipelines_mlx.ti2vid_two_stages import TwoStagePipeline

        if not args.quiet:
            print("Mode: Two-Stage")

        pipe = TwoStagePipeline(model_dir=args.model, low_memory=True)
        video_latent, audio_latent = pipe.generate_two_stage(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            seed=args.seed,
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
        )
        _decode_and_save(pipe, video_latent, audio_latent, args)

    elif args.image:
        from ltx_pipelines_mlx.ti2vid_one_stage import ImageToVideoPipeline

        if not args.quiet:
            print("Mode: Image-to-Video")
            print(f"Image: {args.image}")

        pipe = ImageToVideoPipeline(model_dir=args.model, gemma_model_id=args.gemma)
        output = pipe.generate_and_save(
            prompt=prompt,
            output_path=args.output,
            image=args.image,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            seed=args.seed,
            num_steps=args.steps,
        )
        _print_result(output, t0, args.quiet)
        return
    else:
        from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline

        if not args.quiet:
            print("Mode: Text-to-Video")

        pipe = TextToVideoPipeline(model_dir=args.model, gemma_model_id=args.gemma)
        output = pipe.generate_and_save(
            prompt=prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            seed=args.seed,
            num_steps=args.steps,
        )
        _print_result(output, t0, args.quiet)
        return

    _print_result(args.output, t0, args.quiet)


def _decode_and_save(
    pipe: object,
    video_latent: object,
    audio_latent: object,
    args: argparse.Namespace,
) -> None:
    """Decode latents and save to file (shared by two-stage pipelines)."""
    import tempfile
    from pathlib import Path

    from ltx_core_mlx.utils.memory import aggressive_cleanup

    # Free transformer to make room for VAE decode
    if hasattr(pipe, "low_memory") and pipe.low_memory:
        pipe.dit = None
        pipe.text_encoder = None
        pipe.feature_extractor = None
        pipe._loaded = False
        aggressive_cleanup()

    assert pipe.audio_decoder is not None
    assert pipe.vocoder is not None
    mel = pipe.audio_decoder.decode(audio_latent)
    waveform = pipe.vocoder(mel)
    aggressive_cleanup()

    audio_path = tempfile.mktemp(suffix=".wav")
    pipe._save_waveform(waveform, audio_path, sample_rate=48000)

    assert pipe.vae_decoder is not None
    pipe.vae_decoder.decode_and_stream(video_latent, args.output, fps=24.0, audio_path=audio_path)
    Path(audio_path).unlink(missing_ok=True)
    aggressive_cleanup()


def _print_result(output: str, t0: float, quiet: bool) -> None:
    """Print generation result."""
    elapsed = time.time() - t0
    if not quiet:
        print(f"\nSaved to: {output}")
        print(f"Time: {elapsed:.1f}s")


def _cmd_enhance(args: argparse.Namespace) -> None:
    """Enhance a prompt using Gemma."""
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

    print("Loading Gemma...")
    gemma = GemmaLanguageModel()
    gemma.load(args.gemma)

    if args.mode == "t2v":
        enhanced = gemma.enhance_t2v(args.prompt, seed=args.seed)
    else:
        enhanced = gemma.enhance_i2v(args.prompt, seed=args.seed)

    print(f"\nOriginal: {args.prompt}")
    print(f"\nEnhanced: {enhanced}")


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
