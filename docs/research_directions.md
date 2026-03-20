# Research Directions for Sub-1.15 BPB

## The Challenge
Current best: ~1.20 BPB (estimated on 8xH100 with our hyperparameter tweaks)
Target: < 1.15 BPB
Gap to close: ~0.05 BPB — requires architectural innovation beyond just tuning

## The 4-Hour Ceiling
The unlimited-compute baseline (329K steps, same architecture) achieves 1.175 BPB pre-quant, 1.207 post-quant. This means the current architecture's asymptotic limit is ~1.17 BPB. Breaking 1.15 requires changing architecture, quantization, or tokenizer.

## The BPB Equation
```
BPB = (loss / ln(2)) × (total_tokens / total_bytes)
       ─────────────     ─────────────────────────
       bits per token     tokens per byte
```
Two independent levers:
1. **Lower loss** → better model (architecture, more params, better training)
2. **Fewer tokens per byte** → better tokenizer (larger vocab)

Current decomposition (8xH100 baseline): loss=2.073 nats, tokens_per_byte=0.409, BPB=1.224

## Scaling Law Analysis
- N = 17M params, D = 7.2B training tokens → D/N ≈ 422
- Chinchilla-optimal is D/N ≈ 20. We are 20x past optimal.
- **We are parameter-limited, not compute-limited.** Techniques that increase effective params win.
- Doubling params from 17M to 34M predicts ~0.03-0.05 BPB gain from Chinchilla extrapolation.

---

## Tier 1: High-Impact Ideas (each could yield 0.01-0.05+ BPB)

### 1. Larger Vocabulary (vocab=4096 + ALBERT factorization)
**Why it matters**: BPB = bits_per_token × tokens_per_byte. Larger vocab → each token covers more bytes → dramatically better BPB even if loss increases slightly.

**The math**: At vocab=4096, tokens_per_byte drops from ~0.41 to ~0.29-0.33. Even with 0.2-0.3 nats higher loss, BPB improves by 0.05-0.15.

**ALBERT factorization**: Factor embedding as V×E + E×D where E=128. At V=4096: 590K params vs 2.1M unfactored. Nearly same cost as current V=1024 embedding (524K).

**Critical first step**: Train SentencePiece tokenizer at V=4096 and measure actual tokens_per_byte on validation set. This determines if the idea is viable.

**Expected gain**: 0.05-0.15 BPB (potentially our largest single lever)

**Risk**: Tokenizer scrutiny from judges. Must prove BPB calculation is correct.

**Key papers**: ALBERT (arXiv:1909.11942), "No Free Lunch in Tokenization" (Schmidt et al., 2024), "Scaling Laws for Vocabulary Size" (Tao et al., 2024)

### 2. Depth Recurrence (Weight Sharing Across Layers)
**Why it matters**: The 16MB constraint is on stored parameters, not effective depth. Sharing weights across layers and looping gives a deeper, wider model with fewer unique parameters.

**What to try** (ordered by expected impact):
- 2 blocks × 6 repeats, dim=870 (maximum width, most sharing)
- 3 blocks × 4 repeats, dim=810 (balanced)
- 4 blocks × 3 repeats, dim=768 (current prototype)

**Key insight**: Width matters more than depth at <100M params (Tay et al. 2022 "Scale Efficiently"). Fewer unique blocks + more repeats = wider model = likely better.

**Per-layer control parameters** (attn_scale, mlp_scale, resid_mix, q_gain) are validated by the literature as critical for weight-shared models. Consider adding per-layer RMSNorm gains (~768 params each, cheap).

**Expected gain**: 0.01-0.03 BPB

**Key papers**: Universal Transformers (arXiv:1807.03819), ALBERT (arXiv:1909.11942), Looped Transformers (Giannou et al., 2023)

### 3. Int4 QAT (Quantization-Aware Training)
**Why it matters**: Going from int8 to int4 doubles effective params in 16MB (17M → ~30M). QAT trains the model to be robust to quantization noise.

**How it works**: During training, "fake quantize" weights to int4 grid points using straight-through estimator. The model learns weights that naturally sit near int4 values. Post-training int4 quantization then causes minimal damage.

**Group quantization**: group_size=32 gives per-group scales, 4.5 bits/param effective → ~28-30M params in 16MB.

**Expected gain**: 0.02-0.04 BPB net (more params minus quant penalty)

**Key papers**: LSQ (arXiv:1902.08153), GPTQ (arXiv:2210.17323), AWQ (arXiv:2306.00978)

---

## Tier 2: Medium-Impact Ideas (0.005-0.02 BPB each)

### 4. SwiGLU Activation
Replace ReLU² MLP with SwiGLU. Gated activations decouple feature selection from feature computation. Used by every modern LLM (LLaMA, Mistral, Gemma, Qwen).

At iso-parameters: hidden_dim shrinks from 2×dim to 4×dim/3 (682 for dim=512). 3 matrices instead of 2, same total params.

**Expected gain**: 0.005-0.015 BPB
**Key paper**: Shazeer 2020 "GLU Variants Improve Transformer" (arXiv:2002.05202)

### 5. Sliding Window Evaluation with Overlap
Evaluate with overlapping windows, score only the non-overlapping suffix. Each scored token gets ≥(window−stride) context tokens. Eliminates cold-start loss.

**Implementation challenge**: Need per-position losses (model returns mean loss). Add get_logits() method for eval.

Current eval: ~1.4 seconds out of 600s budget. Even 10x slower is fine.

**Expected gain**: 0.01-0.03 BPB (free, no training change)

### 6. Differential Attention
Each head splits into two sub-heads, computes two attention patterns, takes their difference. Cancels noise, amplifies signal. Using halved head_dim variant: zero parameter overhead, zero compute overhead.

**Expected gain**: 0.002-0.008 BPB
**Key paper**: Ye et al. 2024 "Differential Transformer" (arXiv:2410.05258)

### 7. Test-Time Depth Recurrence
With weight-shared layers, run MORE passes at eval than during training. Train with 12 effective layers, eval with 16-24 by cycling control parameters.

Essentially free (eval headroom is massive). Simplest: repeat last cycle's control params.

**Expected gain**: 0.00-0.01 BPB

---

## Tier 3: Validated Not to Help / Too Risky

| Idea | Status | Reason |
|------|--------|--------|
| EMA weights (1xH100) | Tested, hurts | Model still improving at ~1600 steps |
| ROPE_BASE tuning | Tested, no impact | 50000 same as default 10000 |
| MoE | Rejected | Wrong constraint regime (param-limited, not compute-limited) |
| Mamba/SSM | Rejected | Marginal at seq=1024, dependency risk |
| Linear Attention | Rejected | 5-15% quality loss |
| DEQ models | Rejected | Training instability |
| BitNet/Ternary | Rejected | Unproven at 17M scale |
| Label smoothing | Rejected | Directly increases cross-entropy → hurts BPB |
| Knowledge distillation | Rejected | Likely against competition rules |
| Byte-level models | Rejected | 3-5x compute penalty |
| AQLM/VQ codebooks | Rejected | Implementation complexity too high |

## Key Numbers
- 16,000,000 bytes total (code + compressed model)
- Current code: ~49KB → ~15,950KB for model
- int8+zlib compression: ~0.92 bytes per parameter
- Max params at int8: ~17.3M
- Max params at int4 (group_size=32): ~28-30M
- Training budget: ~13,780 steps on 8xH100 in 10 min
- Val set: 62M tokens (fixed first 50k docs of FineWeb)
- Eval time: ~1.4 seconds out of 600s budget (massive headroom)
- Shannon entropy floor for web text: ~0.8-1.0 BPB
- 4-hour architecture ceiling: 1.175 BPB pre-quant

## Evaluation Budget Analysis
| Eval Strategy | Time | BPB Gain |
|--------------|------|----------|
| Non-overlapping, seq=1024 | ~0.7s | baseline |
| Non-overlapping, seq=2048 (NTK-RoPE) | ~1.4s | +0.048 |
| Sliding window, stride=1024, seq=2048 | ~2.8s | +0.06-0.07 (est) |
| Sliding window, stride=512, seq=2048 | ~5.6s | +0.06-0.08 (est) |
| Sliding window, stride=256, seq=2048 | ~11s | +0.06-0.08 (est) |
| + Test-time extra recurrence passes | 2-3x above | +0.00-0.01 (est) |

All well within the 600s eval budget.
