# Known Issues

Remaining issues and areas not yet verified against the PyTorch reference.

---

## Not Yet Verified

### Text Encoder Pipeline

The full text encoding pipeline has not been compared with the PyTorch reference:

- **Gemma 3 12B**: loaded via mlx-lm; hidden state extraction not compared.
- **Embeddings1DConnector**: RoPE refinement (SPLIT mode) and learnable register replacement not compared.
- **GemmaFeaturesExtractorV2**: separate video/audio projections not compared.

These components are functional but their numerical output has not been validated against ltx-core.

### Conditioning System

- `VideoConditionByLatentIndex`, `VideoConditionByKeyframeIndex`, `VideoConditionByReferenceLatent` have not been compared with the reference for correct token placement and mask construction.

---

## Output Quality

- The model produces spatially varied, colorful content from the VAE decoder.
- However, full end-to-end generation output appears **blocky and noisy**, not producing coherent images/video.

### Possible Causes (Under Investigation)

1. **Text connector register replacement**: the connector appends learnable registers to the sequence; the replacement logic may not match the reference exactly.
2. **Attention mask conversion**: cross-attention masks between text and latent tokens may have incorrect boolean/additive conversion.
3. **Per-layer normalization details**: subtle differences in norm application order or epsilon values between MLX and PyTorch RMSNorm.
4. **Text embedding quality**: if hidden states from Gemma via mlx-lm differ from the PyTorch extraction, all downstream conditioning would be affected.

---

## Verified Working

For contrast, these components have been individually verified correct:

- Timestep embedding (max diff 5e-05)
- RoPE frequency grid and cos/sin (max diff 4e-05)
- Full transformer block 0 (identical output)
- Self-attention (identical output)
- VAE decoder (correct shapes, varied output)
- All 3 upsampler variants (correct shapes)
- Audio vocoder format and layout
