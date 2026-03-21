"""Weight loading utilities for pre-converted MLX safetensors."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_split_safetensors(
    path: str | Path,
    prefix: str | None = None,
) -> dict[str, mx.array]:
    """Load weights from a safetensors file, optionally stripping a prefix.

    Uses mx.load() which natively handles bfloat16 and all MLX dtypes.

    Args:
        path: Path to the .safetensors file.
        prefix: If provided, only keys starting with this prefix are loaded
            and the prefix is stripped from the key names.

    Returns:
        Dict mapping parameter names to MLX arrays.
    """
    path = Path(path)
    raw = mx.load(str(path))

    if not prefix:
        return raw

    weights: dict[str, mx.array] = {}
    for key, tensor in raw.items():
        if key.startswith(prefix):
            weights[key[len(prefix) :]] = tensor

    return weights


def apply_quantization(
    model: nn.Module,
    weights: dict[str, mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> None:
    """Apply quantization to Linear layers that have quantized weights.

    Detects quantized layers by checking for 'scales' and 'biases' keys
    in the weight dict and calls nn.quantize on matching layers.

    Args:
        model: The nn.Module to quantize.
        weights: Weight dict (may contain scales/biases for quantized layers).
        group_size: Quantization group size.
        bits: Quantization bit width.
    """
    quantized_layers: set[str] = set()

    for key in weights:
        if key.endswith(".scales"):
            layer_name = key.rsplit(".scales", 1)[0]
            quantized_layers.add(layer_name)

    if not quantized_layers:
        return

    # Build class predicate: only quantize layers that have scales in the weights
    def _should_quantize(path: str, _module: nn.Module) -> bool:
        return path in quantized_layers and isinstance(_module, nn.Linear)

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=_should_quantize)


def remap_audio_vae_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap underscore-prefixed per-channel stats keys for audio VAE.

    MLX treats ``_``-prefixed attributes as private, so safetensors keys
    ``_mean_of_means`` / ``_std_of_means`` must be loaded as
    ``mean_of_means`` / ``std_of_means``.
    """
    return {
        k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
        for k, v in weights.items()
    }
