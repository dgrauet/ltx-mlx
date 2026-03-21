"""Gemma 3 language model wrapper via mlx-lm.

Ported from ltx-core/src/ltx_core/text_encoders/gemma/language_model.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class GemmaLanguageModel(nn.Module):
    """Wrapper around Gemma 3 12B loaded via mlx-lm.

    Uses mlx_lm.load() for native MLX loading.
    Extracts hidden states from ALL layers for multi-layer feature extraction.

    Gemma 3 12B has 48 transformer layers + embedding layer = 49 total
    hidden states (embedding output + 48 layer outputs), each of dim 3840.

    Args:
        model_path: Path to the Gemma 3 MLX weights directory.
    """

    def __init__(self, model_path: str | Path | None = None):
        super().__init__()
        self._model = None
        self._tokenizer = None
        self._model_path = str(model_path) if model_path else None

    def load(self, model_path: str | None = None) -> None:
        """Load the Gemma model via mlx-lm.

        Args:
            model_path: Path or HuggingFace repo ID.
        """
        from mlx_lm import load as mlx_lm_load

        path = model_path or self._model_path
        if path is None:
            raise ValueError("model_path must be provided")

        self._model, self._tokenizer = mlx_lm_load(path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def tokenize(self, text: str, max_length: int = 1024) -> tuple[mx.array, mx.array]:
        """Tokenize a text string with left-padding to max_length.

        Reference: LTXVGemmaTokenizer pads to max_length with padding_side="left".
        Returns both token_ids and attention_mask.

        Args:
            text: Input text.
            max_length: Sequence length (padded to this length).

        Returns:
            Tuple of (token_ids, attention_mask), each shape (1, max_length).
            attention_mask: 1 for valid tokens, 0 for padding.
        """
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokens = self._tokenizer.encode(text.strip())
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]  # Keep last tokens (left-pad = truncate from left)

        # Left-pad to max_length using the native pad token
        pad_length = max_length - len(tokens)
        pad_token = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        padded_tokens = [pad_token] * pad_length + tokens
        attention_mask = [0] * pad_length + [1] * len(tokens)

        return mx.array([padded_tokens]), mx.array([attention_mask])

    def get_all_hidden_states(
        self,
        token_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> list[mx.array]:
        """Extract hidden states from ALL layers of the language model.

        Collects the embedding output plus all transformer layer outputs,
        yielding 49 hidden states total for Gemma 3 12B.

        Args:
            token_ids: (B, seq_len) token IDs.
            attention_mask: (B, seq_len) binary mask (1=valid, 0=padding).
                Used to build a causal attention mask that also masks padding.

        Returns:
            List of (B, seq_len, hidden_dim) tensors, one per layer.
            Length = num_transformer_layers + 1 (embedding + layers).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Navigate to the inner model with embed_tokens and layers.
        inner = self._model
        for attr in ("model", "language_model", "model"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
            if hasattr(inner, "embed_tokens"):
                break

        if not hasattr(inner, "embed_tokens"):
            raise RuntimeError("Cannot find embed_tokens in the model hierarchy")

        all_hidden_states: list[mx.array] = []

        # Embeddings with Gemma 3 scaling (sqrt(hidden_size))
        h = inner.embed_tokens(token_ids)
        hidden_size = h.shape[-1]
        h = h * mx.array(hidden_size**0.5, dtype=mx.bfloat16).astype(h.dtype)
        all_hidden_states.append(h)

        # Build combined causal + padding mask.
        # Causal mask prevents attending to future tokens; padding mask
        # prevents attending to padding tokens (left-padded input).
        # Must be bfloat16 to match mlx-lm's scaled_dot_product_attention output type.
        T = token_ids.shape[1]
        causal_mask = mx.triu(mx.full((T, T), -1e9, dtype=mx.bfloat16), k=1)
        if attention_mask is not None:
            # attention_mask: (B, T) with 1=valid, 0=padding
            pad_mask = (1 - attention_mask[:, None, None, :].astype(mx.bfloat16)) * -1e9
            combined_mask = causal_mask[None, None, :, :] + pad_mask  # (B, 1, T, T)
        else:
            combined_mask = causal_mask[None, None, :, :]  # (1, 1, T, T)

        # Run layers with combined mask
        for layer in inner.layers:
            h = layer(h, mask=combined_mask, cache=None)
            if isinstance(h, tuple):
                h = h[0]
            all_hidden_states.append(h)

        return all_hidden_states

    def get_hidden_states(self, token_ids: mx.array) -> mx.array:
        """Extract final hidden states from the language model.

        Args:
            token_ids: (B, seq_len) token IDs.

        Returns:
            Hidden states of shape (B, seq_len, hidden_dim) from the last layer.
        """
        all_states = self.get_all_hidden_states(token_ids)
        return all_states[-1]

    def encode(self, text: str, max_length: int = 1024) -> mx.array:
        """Tokenize and extract final hidden states in one call.

        Args:
            text: Input text.
            max_length: Padded sequence length.

        Returns:
            Hidden states of shape (1, max_length, hidden_dim).
        """
        token_ids, attention_mask = self.tokenize(text, max_length)
        all_states = self.get_all_hidden_states(token_ids, attention_mask=attention_mask)
        return all_states[-1]

    def encode_all_layers(self, text: str, max_length: int = 1024) -> tuple[list[mx.array], mx.array]:
        """Tokenize and extract ALL layer hidden states.

        Args:
            text: Input text.
            max_length: Padded sequence length.

        Returns:
            Tuple of (hidden_states, attention_mask):
            - hidden_states: list of (1, max_length, hidden_dim) tensors (49 total)
            - attention_mask: (1, max_length) binary mask
        """
        token_ids, attention_mask = self.tokenize(text, max_length)
        hidden_states = self.get_all_hidden_states(token_ids, attention_mask=attention_mask)
        return hidden_states, attention_mask
