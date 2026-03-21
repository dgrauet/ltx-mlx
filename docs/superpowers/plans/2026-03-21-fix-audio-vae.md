# Fix Audio VAE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 11 bugs in the audio VAE pipeline (decoder, encoder, vocoder, BWE) by aligning with the ltx-core reference implementation.

**Architecture:** The audio VAE uses Conv2d (not Conv1d). Latent shape is (B, 8, T, 16), treated as NHWC (B, T, 16, 8) in MLX. The reference uses causal Conv2d (asymmetric padding along height=time), PixelNorm normalization, and attention blocks — all currently missing. The vocoder uses SnakeBeta activations with log-scale alpha/beta, and the BWE uses Hann-windowed resampling with causal STFT.

**Tech Stack:** Python 3.11+, MLX, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/audio_vae.py` | Modify | Add PixelNorm, norm_out, AttnBlock, causal padding to all Conv2d |
| `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/encoder.py` | Modify | Same causal + norm fixes mirrored |
| `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/vocoder.py` | Modify | Fix SnakeBeta exp(), fix DownSample1d padding, add apply_final_activation flag |
| `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/bwe.py` | Modify | Fix resampler window, fix MelSTFT padding, disable tanh on BWE generator |
| `tests/test_audio_shapes.py` | Modify | Update tests for new components |

---

### Task 1: Fix SnakeBeta — add exp() for log-scale weights

Bug #1: Reference stores alpha/beta in log-scale and applies `exp()` during forward. Our port uses raw values with no `exp()`.

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/vocoder.py:38-54`
- Test: `tests/test_audio_shapes.py`

- [ ] **Step 1: Write failing test**

```python
# In tests/test_audio_shapes.py, add:
def test_snakebeta_logscale(self):
    """SnakeBeta with log-scale alpha=0 should give exp(0)=1, same as alpha=1."""
    snake = SnakeBeta(4)
    # Set weights to log-scale zeros (exp(0)=1.0)
    snake.alpha = mx.zeros((4,))
    snake.beta = mx.zeros((4,))
    x = mx.ones((1, 5, 4))
    result = snake(x)
    # With alpha=exp(0)=1.0, beta=exp(0)=1.0: x + (1/1) * sin^2(1*x) = 1 + sin^2(1)
    expected_val = 1.0 + (1.0 / 1.0) * (mx.sin(mx.array(1.0)) ** 2)
    assert mx.allclose(result[0, 0, 0], expected_val, atol=1e-5).item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_audio_shapes.py::TestSnakeBeta::test_snakebeta_logscale -v`
Expected: FAIL — without exp(), alpha=0 gives sin(0*x)=0, result=x

- [ ] **Step 3: Fix SnakeBeta**

```python
class SnakeBeta(nn.Module):
    """SnakeBeta activation: x + (1/b) * sin^2(a * x).

    Weights are stored in LOG-SCALE. Forward applies exp() to get actual values.
    Weight keys: ``act.alpha``, ``act.beta``.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Initialized to zeros (exp(0)=1.0) to match reference default
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C)"""
        alpha = mx.exp(self.alpha).reshape(1, 1, -1)
        beta = mx.exp(self.beta).reshape(1, 1, -1)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(alpha * x), 2)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_audio_shapes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/vocoder.py tests/test_audio_shapes.py
git commit -m "fix(audio_vae): SnakeBeta applies exp() to log-scale alpha/beta weights"
```

---

### Task 2: Add PixelNorm and norm_out to AudioResBlock and AudioVAEDecoder

Bug #2: AudioResBlock omits normalization layers (PixelNorm).
Bug #3: AudioVAEDecoder/Encoder missing `norm_out` before `conv_out`.

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/audio_vae.py`
- Test: `tests/test_audio_shapes.py`

- [ ] **Step 1: Write failing tests**

```python
def test_decoder_has_norm_out(self):
    """Decoder should have norm_out attribute."""
    dec = AudioVAEDecoder()
    assert hasattr(dec, 'norm_out')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_audio_shapes.py::TestAudioVAEDecoder::test_decoder_has_norm_out -v`
Expected: FAIL

- [ ] **Step 3: Add PixelNorm to AudioResBlock and norm_out to decoder**

Add a `pixel_norm` function at the top of `audio_vae.py`:
```python
def pixel_norm(x: mx.array, eps: float = 1e-8) -> mx.array:
    """RMS normalization over the channel dimension (PixelNorm)."""
    return mx.fast.rms_norm(x, weight=None, eps=eps)
```

Update `AudioResBlock.__call__`:
```python
def __call__(self, x: mx.array) -> mx.array:
    """x: (B, H, W, C) in MLX NHWC layout."""
    residual = x
    x = pixel_norm(x)
    x = nn.silu(x)
    x = self.conv1(x)
    x = pixel_norm(x)
    x = nn.silu(x)
    x = self.conv2(x)
    if self.nin_shortcut is not None:
        residual = self.nin_shortcut(residual)
    return x + residual
```

Add `norm_out` to `AudioVAEDecoder.__init__` (as a flag — PixelNorm has no learnable params):
No new module needed since PixelNorm is a stateless function. Just add the call.

Update `AudioVAEDecoder.decode` — add `pixel_norm` before the final `nn.silu`:
```python
    # Up blocks run in reverse index order
    for i in reversed(range(len(self.up))):
        x = self.up[i](x)

    x = pixel_norm(x)  # norm_out
    x = nn.silu(x)
    x = self.conv_out(x)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_audio_shapes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/audio_vae.py tests/test_audio_shapes.py
git commit -m "fix(audio_vae): add PixelNorm to AudioResBlock and norm_out to decoder"
```

---

### Task 3: Add AttnBlock to AudioMidBlock and AudioUpBlock

Bug #4: Attention layers completely omitted from mid-block and up/down blocks.

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/audio_vae.py`
- Test: `tests/test_audio_shapes.py`

- [ ] **Step 1: Write failing test**

```python
def test_mid_block_has_attention(self):
    """MidBlock should have attn_1 between block_1 and block_2."""
    mid = AudioMidBlock(512)
    assert hasattr(mid, 'attn_1')
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement AttnBlock and add to MidBlock/UpBlock**

Add `AudioAttnBlock` class to `audio_vae.py`:
```python
class AudioAttnBlock(nn.Module):
    """Self-attention block for audio VAE.

    Weight keys: norm.{weight,bias}, q.conv.{w,b}, k.conv.{w,b}, v.conv.{w,b}, proj_out.conv.{w,b}
    Uses Conv2d 1x1 for Q/K/V projections (matching reference AttnBlock).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = WrappedConv2d(channels, channels, 1, padding=0)
        self.k = WrappedConv2d(channels, channels, 1, padding=0)
        self.v = WrappedConv2d(channels, channels, 1, padding=0)
        self.proj_out = WrappedConv2d(channels, channels, 1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        residual = x
        h = self.norm(x)

        q = self.q(h).reshape(B, H * W, C)
        k = self.k(h).reshape(B, H * W, C)
        v = self.v(h).reshape(B, H * W, C)

        scale = C ** -0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).reshape(B, H, W, C)
        out = self.proj_out(out)
        return residual + out
```

Update `AudioMidBlock` to include attention:
```python
class AudioMidBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block_1 = AudioResBlock(channels)
        self.attn_1 = AudioAttnBlock(channels)
        self.block_2 = AudioResBlock(channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.block_1(x)
        x = self.attn_1(x)
        x = self.block_2(x)
        return x
```

Add optional attention to `AudioUpBlock`:
```python
class AudioUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, add_upsample=False, add_attention=False):
        super().__init__()
        self.block = [AudioResBlock(in_channels if i == 0 else out_channels, out_channels) for i in range(num_blocks)]
        if add_attention:
            self.attn = [AudioAttnBlock(out_channels) for _ in range(num_blocks)]
        else:
            self.attn = None
        self.upsample = AudioUpsample(out_channels) if add_upsample else None

    def __call__(self, x):
        for i, blk in enumerate(self.block):
            x = blk(x)
            if self.attn is not None:
                x = self.attn[i](x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
```

Update decoder `self.up` list — the reference config has `attn_resolutions={8, 16, 32}`. At frequency dims 16 and 32, we need attention. Up.2 operates at freq=16 (input), up.1 at freq=32 (after upsample of up.2):
```python
self.up = [
    AudioUpBlock(256, 128, num_blocks=3, add_upsample=False, add_attention=False),
    AudioUpBlock(512, 256, num_blocks=3, add_upsample=True, add_attention=True),
    AudioUpBlock(512, 512, num_blocks=3, add_upsample=True, add_attention=True),
]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_audio_shapes.py -v`

- [ ] **Step 5: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/audio_vae.py tests/test_audio_shapes.py
git commit -m "fix(audio_vae): add AttnBlock to AudioMidBlock and AudioUpBlock"
```

---

### Task 4: Implement causal padding for Conv2d in audio VAE

Bug #5: All Conv2d use symmetric padding instead of causal (asymmetric top-heavy).
Bug #6: AudioUpsample missing first-row drop after causal conv.
Bug #7: AudioDownsample missing causal padding.

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/audio_vae.py`
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/encoder.py`
- Test: `tests/test_audio_shapes.py`

- [ ] **Step 1: Write failing test**

```python
def test_causal_upsample_drops_first_row(self):
    """Causal upsample should drop first row, reducing height by 1."""
    up = AudioUpsample(4, causal=True)
    x = mx.zeros((1, 4, 4, 4))  # (B, H=4, W=4, C=4)
    result = up(x)
    # 2x upsample (H=4→8) minus 1 (drop first row) = 7
    assert result.shape[1] == 7
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement causal Conv2d wrapping**

Update `WrappedConv2d` to support causal mode:
```python
class WrappedConv2d(nn.Module):
    """Conv2d with optional causal (height-axis) padding."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, causal=False):
        super().__init__()
        self._causal = causal
        if causal and kernel_size > 1:
            # Causal: no padding in conv, we pad manually
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
            ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            self._pad_h = ks - 1  # full causal pad on top
            sp = padding if isinstance(padding, int) else padding[1]
            self._pad_w = sp
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self._pad_h = 0
            self._pad_w = 0

    def __call__(self, x: mx.array) -> mx.array:
        if self._causal and self._pad_h > 0:
            # NHWC: pad height (axis 1) asymmetrically — all on top, none on bottom
            # Pad width (axis 2) symmetrically
            x = mx.pad(x, [(0, 0), (self._pad_h, 0), (self._pad_w, self._pad_w), (0, 0)])
        return self.conv(x)
```

Update `AudioUpsample` to support causal mode with first-row drop:
```python
class AudioUpsample(nn.Module):
    def __init__(self, channels: int, causal: bool = False):
        super().__init__()
        self.conv = WrappedConv2d(channels, channels, 3, padding=1, causal=causal)
        self._causal = causal

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        x = self.conv(x)
        if self._causal:
            x = x[:, 1:, :, :]  # Drop first row for causal alignment
        return x
```

Update `AudioDownsample` in encoder.py for causal padding:
```python
class AudioDownsample(nn.Module):
    def __init__(self, channels: int, causal: bool = False):
        super().__init__()
        self._causal = causal
        if causal:
            # Causal downsample: pad (2, 0) on height, (0, 1) on width
            self.conv = WrappedConv2d(channels, channels, 3, stride=2, padding=0, causal=False)
        else:
            self.conv = WrappedConv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        if self._causal:
            x = mx.pad(x, [(0, 0), (2, 0), (0, 1), (0, 0)])
        return self.conv(x)
```

Update all constructors to pass `causal=True`:
- `AudioVAEDecoder.__init__`: pass `causal=True` to `WrappedConv2d` for conv_in and conv_out
- `AudioUpBlock`: pass causal to resblocks and upsample
- `AudioMidBlock`: pass causal to resblocks
- `AudioVAEEncoder`: same mirror

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_audio_shapes.py -v`

- [ ] **Step 5: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/audio_vae.py packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/encoder.py tests/test_audio_shapes.py
git commit -m "fix(audio_vae): implement causal padding for all Conv2d, drop first row on upsample"
```

---

### Task 5: Fix DownSample1d padding order in vocoder

Bug #8: Pad amounts (6,5) instead of reference (5,6).

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/vocoder.py:85-101`

- [ ] **Step 1: Write failing test**

```python
def test_downsample1d_pad_order(self):
    """DownSample1d should pad (5, 6) for kernel_size=12, not (6, 5)."""
    ds = DownSample1d(kernel_size=12)
    # With correct padding, specific input should give deterministic output length
    x = mx.ones((1, 20, 1))
    ds.lowpass.filter = mx.ones((1, 12, 1)) / 12.0
    result = ds(x)
    assert result.shape == (1, 10, 1)  # 20 // 2 = 10
```

- [ ] **Step 2: Fix padding in DownSample1d**

```python
def __call__(self, x: mx.array) -> mx.array:
    B, T, C = x.shape
    x = x.transpose(0, 2, 1).reshape(B * C, T, 1)
    K = self.lowpass.filter.shape[1]
    # Reference: pad_left = K // 2 - int(even), pad_right = K // 2
    even = 1 if K % 2 == 0 else 0
    pad_left = K // 2 - even
    pad_right = K // 2
    x = mx.pad(x, [(0, 0), (pad_left, pad_right), (0, 0)])
    x = mx.conv1d(x, self.lowpass.filter, stride=2)
    T_out = x.shape[1]
    return x.reshape(B, C, T_out).transpose(0, 2, 1)
```

For K=12: pad_left = 6 - 1 = 5, pad_right = 6. Matches reference.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_audio_shapes.py -v`

- [ ] **Step 4: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/vocoder.py tests/test_audio_shapes.py
git commit -m "fix(vocoder): DownSample1d pad order (5,6) matches reference"
```

---

### Task 6: Fix BWE resampler — use Hann window and correct kernel size

Bug #9: Uses Kaiser window (beta=6.0, 129 taps) instead of Hann (43 taps, rolloff=0.99).

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/bwe.py:28-73`

- [ ] **Step 1: Fix KaiserSincResampler to use Hann window**

Rename to `HannSincResampler` and fix the kernel:
```python
class HannSincResampler:
    """3x upsampler using Hann-windowed sinc interpolation.

    Matches reference UpSample1d(ratio=3, window_type="hann").
    """

    def __init__(self, upsample_factor: int = 3):
        self.upsample_factor = upsample_factor
        kernel = self._build_kernel(upsample_factor)
        self.kernel = mx.array(kernel[:, None])  # (K, 1)

    @staticmethod
    def _build_kernel(ratio: int) -> np.ndarray:
        """Build Hann-windowed sinc filter matching reference."""
        rolloff = 0.99
        lowpass_filter_width = 6
        width = int(np.ceil(lowpass_filter_width / rolloff))  # 7
        kernel_size = 2 * width * ratio + 1  # 43
        idx = np.arange(-width * ratio, width * ratio + 1, dtype=np.float64)
        t = idx / ratio
        sinc = np.sinc(t * rolloff)
        window = np.cos(t * np.pi / (2 * width)) ** 2  # Hann window
        kernel = (sinc * window).astype(np.float32)
        kernel = kernel / kernel.sum() * ratio
        return kernel

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample: (B, T) -> (B, T * factor)."""
        B, T = x.shape
        upsampled = mx.zeros((B, T * self.upsample_factor))
        upsampled = upsampled.at[:, :: self.upsample_factor].add(x)
        upsampled = upsampled[:, :, None]
        pad_len = self.kernel.shape[0] // 2
        upsampled = mx.pad(upsampled, [(0, 0), (pad_len, pad_len), (0, 0)])
        filt = self.kernel[None, :, :]  # (1, K, 1)
        result = mx.conv1d(upsampled, filt, padding=0)
        return result.squeeze(-1)[:, : T * self.upsample_factor]
```

Update `VocoderWithBWE.__init__`:
```python
self._resampler = HannSincResampler(upsample_factor=3)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_audio_shapes.py -v`

- [ ] **Step 3: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/bwe.py
git commit -m "fix(bwe): resampler uses Hann window with correct kernel size (43 taps)"
```

---

### Task 7: Fix BWE MelSTFT padding — causal instead of symmetric

Bug #10: Symmetric padding (256, 256) instead of causal (352, 0).

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/bwe.py:124-163`

- [ ] **Step 1: Fix MelSTFT padding**

In `MelSTFT.__call__`, change:
```python
    # Causal padding: left-only, matching reference _STFTFn
    left_pad = max(0, self.n_fft - self.hop_length)  # 512 - 160 = 352
    x = mx.pad(x, [(0, 0), (left_pad, 0), (0, 0)])
```

Replace the old symmetric padding lines.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_audio_shapes.py -v`

- [ ] **Step 3: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/bwe.py
git commit -m "fix(bwe): MelSTFT uses causal left-only padding (352,0) matching reference"
```

---

### Task 8: Disable tanh on BWE generator output

Bug #11: BWE generator always applies tanh, but reference disables it for BWE residual.

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/vocoder.py` (BigVGANVocoder)
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/bwe.py` (VocoderWithBWE)

- [ ] **Step 1: Add apply_final_activation flag to BigVGANVocoder**

In `BigVGANVocoder.__init__`, add parameter:
```python
def __init__(self, ..., apply_final_activation: bool = True):
    ...
    self._apply_final_activation = apply_final_activation
```

In `BigVGANVocoder.__call__`, change:
```python
    x = self.conv_post(x)
    if self._apply_final_activation:
        x = mx.tanh(x)
```

- [ ] **Step 2: Set apply_final_activation=False for BWE generator**

In `VocoderWithBWE.__init__`:
```python
self.bwe_generator = BigVGANVocoder(
    ...,
    apply_final_activation=False,
)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_audio_shapes.py -v`

- [ ] **Step 4: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/vocoder.py packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/bwe.py
git commit -m "fix(bwe): disable tanh on BWE generator output (residual should be unbounded)"
```

---

### Task 9: Update encoder with matching fixes

Mirror decoder fixes (PixelNorm, norm_out, causal, attention) in the encoder.

**Files:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/encoder.py`

- [ ] **Step 1: Update encoder**

- Import `pixel_norm` and `AudioAttnBlock` from `audio_vae.py`
- Add `pixel_norm` + `nn.silu` before `self.conv_out` (norm_out)
- Pass `causal=True` to all WrappedConv2d and AudioDownBlock
- Add attention to AudioDownBlock (mirror of AudioUpBlock)

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_audio_shapes.py -v`
Then: `uv run pytest tests/ -m "not slow" -q`

- [ ] **Step 3: Commit**

```bash
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/audio_vae/encoder.py
git commit -m "fix(audio_vae): mirror decoder fixes in encoder (PixelNorm, norm_out, causal, attention)"
```
