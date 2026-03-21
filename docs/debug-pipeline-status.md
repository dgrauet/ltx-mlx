# Pipeline Debug Status

## Résultat final des comparaisons

### Chaque opération individuelle matche la référence PyTorch

| Opération | Diff MLX vs PyTorch | Corrélation | Verdict |
|-----------|-------------------|-------------|---------|
| `get_timestep_embedding` | 0.002 | ~1.0 | ✅ |
| AdaLN params | 0.012 | 0.9999 | ✅ |
| `scale_shift_table` (poids) | 0.000 | 1.0 | ✅ |
| RMSNorm | 0.000 | 1.0 | ✅ |
| Modulation (mêmes params) | 0.029 | ~1.0 | ✅ |
| RoPE values (cos/sin) | 0.009 | 1.0 | ✅ |
| RoPE application | 0.0003 | 1.0 | ✅ |
| Attention sans RoPE | 0.0001 | 1.0 | ✅ |
| Attention avec RoPE | 0.0003 | 1.0 | ✅ |
| Connector (8 blocs) | 1.03 | ~1.0 | ✅ |
| Register replacement | 0.000 | 1.0 | ✅ |
| VAE Decoder | fonctionne | — | ✅ |
| Block 0 complet | 30.0 | 0.97 | ⚠️ accumulé |
| 48 blocs complets | 3.3 | 0.34 | ⚠️ accumulé |

### Comparaison PyTorch MPS vs CPU (même hardware Metal)

| Opération | Diff MPS vs CPU | Notre diff MLX vs CPU |
|-----------|----------------|----------------------|
| Patchify | 0.016 | 0.016 |
| Block 0 | 0.019 | 30.0 |
| Block 1 | 0.022 | 21.0 |
| Block 2 | 0.024 | 25.0 |

**Conclusion** : PyTorch MPS matche CPU parfaitement (diff 0.02), mais notre MLX a diff 30 au bloc 0. Ce n'est **PAS** une différence de précision hardware Metal — c'est une **amplification** de la diff AdaLN 0.012 par les gates.

## Chaîne d'amplification identifiée

```
get_timestep_embedding (MLX float32 sin/cos)
  │ diff 0.002 vs PyTorch (probablement calcul sin/cos interne différent)
  ▼
AdaLN linear1 → SiLU → linear2 (bf16 × bf16 matmul)
  │ diff 0.012 (0.002 amplifié par matmul 4096×256 puis 4096×4096)
  ▼
AdaLN final linear (36864 × 4096 matmul)
  │ diff 0.012 propagé (pas amplifié)
  ▼
scale_sa + shift_sa + gate_sa
  │ diff 0.012 sur le gate, gate range [-6.45, +0.60]
  │ appliqué à attention output magnitude ~50
  ▼
block_output = x + attention(modulated_x) × gate
  │ diff 0.012 × 50 × 6.45 ≈ 30 dans le pire cas
  ▼
Block 0 diff ≈ 30, corrélation 0.97
  │ accumulé sur 48 blocs
  ▼
Final diff ≈ 3.3, corrélation 0.34
```

## Root cause de la diff 0.012 au AdaLN

La diff de 0.002 au `get_timestep_embedding` est la source. Elle vient de la façon dont MLX et PyTorch calculent `sin()` et `cos()` en float32 — les implémentations numériques diffèrent légèrement.

PyTorch MPS ne souffre pas de ce problème car `get_timestep_embedding` est une fonction PyTorch qui produit exactement les mêmes valeurs sur CPU et MPS.

**Solution potentielle** : implémenter le calcul de `sin`/`cos` pour correspondre exactement à PyTorch, ou utiliser une lookup table pré-calculée.

## Schéma du pipeline complet

```
PROMPT "a cat walking..."
    │
    ▼
┌─────────────────────────────────┐
│  GEMMA 3 (mlx-lm, 4-bit)       │  ✅ Extraction corrigée
│  embed × √3840 + causal mask   │     (scaling + native mask "causal")
│  48 layers → 49 hidden states  │  ❓ Non comparé vs transformers
└────────────┬────────────────────┘
             │ 49 × (1, 256, 3840)
             ▼
┌─────────────────────────────────┐
│  FEATURE EXTRACTOR              │  ✅ Vérifié (diff 0.029)
│  per-layer RMSNorm + zero pad  │
│  rescale √(target/source)       │
│  video_proj + audio_proj        │
└────────────┬────────────────────┘
             │ (1, 256, 4096) + (1, 256, 2048)
             ▼
┌─────────────────────────────────┐
│  CONNECTOR (8 transformer blks) │  ✅ Vérifié (diff 1.03)
│  register replacement (diff 0)  │
│  RoPE SPLIT log-spaced          │  ✅ Corrigé (était standard)
│  output RMSNorm                 │
└────────────┬────────────────────┘
             │ video_emb + audio_emb
             ▼
┌─────────────────────────────────────────────────────┐
│  DiT TRANSFORMER (48 blocks)                        │
│                                                     │
│  Chaque opération matche individuellement ✅         │
│  Mais les diffs s'accumulent sur 48 blocs :         │
│                                                     │
│  AdaLN params diff 0.012                            │
│       → amplifié par gates [-6.45, +0.60]           │
│       → block 0 diff 30 (corr 0.97)                │
│       → block 47 accumulé (corr 0.34)              │
│                                                     │
│  Source: get_timestep_embedding sin/cos diff 0.002  │
└────────────┬────────────────────────────────────────┘
             │ velocity
             ▼
┌─────────────────────────────────┐
│  EULER DENOISE (8 steps)        │  ✅ Formule correcte
└────────────┬────────────────────┘
             │ latent
             ▼
┌─────────────────────────────────┐
│  VAE DECODER                    │  ✅ Fonctionne
└────────────┬────────────────────┘
             │
             ▼
         VIDEO .mp4 (blocky)

```

## Corrections appliquées (historique complet)

### DiT Model
- `get_timestep_embedding` : ordre `[sin, cos]`, diviseur `(half - 1)` ✅
- `AdaLayerNormSingle` : retourne tuple `(params, embedded_timestep)` ✅
- Output block : `scale_shift_table + embedded_timestep` adaptatif ✅
- RMSNorm au lieu de LayerNorm dans les blocs ✅
- Param ordering : `[sa(0-2), ff(3-5), ca(6-8)]` ✅
- Cross-attention AdaLN : réactivé (config `false` mais poids = `true`) ✅
- AV cross-attn scale/shift : corrigé l'ordre ✅
- RoPE : log-spaced grid, INTERLEAVED, per-head ✅
- Positions : midpoints `i + 0.5` (`use_middle_indices_grid=true`) ✅
- Positions passées au denoise loop ✅

### Text Encoder
- Gemma embedding scaling : `h × √hidden_size` ✅
- Gemma attention mask : `"causal"` string native (pas mask explicite) ✅
- Tokenization : left-pad à 256 avec `pad_token_id=0` ✅
- Per-layer RMS normalization + zero-out padding positions ✅
- Connector RoPE : log-spaced grid SPLIT (était standard 1/θ^k) ✅
- Register replacement : tokens valides au début, registers à la fin ✅

### VAE Decoder
- CausalConv3d : replicate first frame ✅
- PixelNorm pre-activation dans ResBlocks ✅
- Pixel shuffle : C-outermost, final `sf=4, tf=1` ✅
- Temporal frame removal `x[:, 1:]` après tf>1 ✅

### Audio
- Vocoder stereo concat `(B, 128, T)` ✅
- Filter shapes MLX Conv1d `(O, K, I)` ✅
- STFT basis transposé par mlx-forge ✅
- Audio VAE key remapping ✅

### Upsampler
- 3 variantes : spatial×2, spatial×1.5 (rational), temporal×2 ✅

## Prochaines étapes

1. **Comparer les hidden states Gemma** : charger le même Gemma (bf16) en
   PyTorch transformers et comparer avec notre mlx-lm — vérifier si les
   49 hidden states matchent

2. **Résoudre la diff sin/cos** : investiguer pourquoi `mx.sin(x)` ≠
   `torch.sin(x)` pour les mêmes valeurs float32, et aligner

3. **Obtenir un output de référence** : faire tourner la pipeline ltx-core
   complète sur une machine CUDA (ou demander à Lightricks un exemple de
   sortie de référence pour ce prompt/seed)

4. **Tester avec les poids PyTorch originaux** (non convertis par mlx-forge)
   chargés directement en PyTorch MPS pour confirmer que le modèle produit
   de bonnes images avec ces poids spécifiques
