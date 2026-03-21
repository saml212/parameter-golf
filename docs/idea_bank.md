# Idea Bank — Parameter Golf Competition

## Leverage Table — Where Are the Outsized Returns?

| Lever | Current | Max | Gap | Status |
|-------|---------|-----|-----|--------|
| **Training time** | 600s | 600s | 0 | Maxed out |
| **Eval time** | ~210s | 600s | **390s idle** | Biggest untapped resource |
| **Artifact bytes** | 15.7MB | 16.0MB | **300KB** | Could fit more params with better compression |
| **Model depth** | 12L | 13L+ | **Open** | 12L just shipped. 13L with better compression? |
| **Training data ordering** | Sequential shards | Curriculum/shuffle | **Untapped** | Research says 0.001-0.005 BPP |
| **Compression scheme** | Gradient-guided int5/6/7 | Bit-packing, codebooks | **Partially explored** | Shipped. Can push further |
| **Weight averaging** | EMA (0.997) | — | Small | Near-optimal |
| **Eval methodology** | Sliding stride=64 | Ensemble, multi-stride | **Untapped** | Checkpoint ensemble possible |

## Submitted: PR #332

12L + Gradient-Guided Adaptive Quantization + Partial RoPE + LN Scale + XSA + EMA
- val_bpb: 1.1320 (3-seed mean, std 0.0002)
- Novel: gradient-guided per-tensor int5/int6/int7 allocation
- Negative finding: Late QAT hurts at 12L (throughput cost > quant benefit)

## Priority Queue — Next Experiments

### High Priority (likely 0.002+ BPP each)
- [ ] **Checkpoint ensemble at eval**: Save EMA + final weights, eval both, average logits. Uses idle eval time. Novel.
- [ ] **Bit-packed int6**: Pack 6-bit values natively (6 values per 4.5 bytes vs 6 bytes). 25% better compression. Could enable 13L or MLP=1536 at 12L.
- [ ] **FA3 instead of FA2**: PR #315 uses FA3 directly. May give ~0.001 from better attention kernel.

### Medium Priority (0.001-0.002 BPP)
- [ ] MLP=1472 at 12L (between 1408 and 1536 — test if artifact fits)
- [ ] EMA decay sweep: 0.995, 0.998, 0.999
- [ ] Partial RoPE dims sweep: 8, 24, 32
- [ ] Multi-stride eval ensemble: stride=64 AND stride=32, average per-position BPP
- [ ] Data curriculum-as-warmup: easy data first 1000 steps

### Low Priority / Speculative
- [ ] 13 layers with int5 for all weights + gradient-guided allocation
- [ ] SP4096 tokenizer at 12L
- [ ] Attention sink tokens (learned prefix)
- [ ] Asymmetric encoder/decoder (7E+5D vs 6E+6D)
- [ ] Learned non-uniform quantization codebook (k-means on weight values)

## Tried and Resolved

| Idea | Result | Verdict |
|------|--------|---------|
| **Gradient-guided quant** | Enables 12L, saves ~1MB | **SHIPPED PR #332** |
| Partial RoPE (16/64 dims) | ~0.001 improvement | Shipped |
| LN Scale (1/sqrt(layer+1)) | ~0.001 improvement | Shipped |
| XSA on last 4 layers | ~0.003 improvement | Shipped |
| EMA (0.997) replacing SWA | ~0.003 improvement | Shipped |
| Batch=524K | 22% more steps than 786K | Shipped |
| Int8 tok_emb | Saves ~250KB vs fp16, ~lossless | Shipped |
| Late QAT at 12L | 1.1361 (WORSE than 1.1321) | **Dead end at 12L** |
| Cyclic SWA | 1.1466 (worse than linear) | Dead end |
| Progressive layer freezing | 1.4321 (catastrophic) | Dead end |
| QAT int6 STE (full training) | 115ms/step, net loss | Dead end |
| Train@1024 | 1.1811 (halved context) | Dead end |
| Pre-eval TTT | Ruled illegal | Dead end |
| Causal online TTT | 1.1322 (legal but politically risky) | Shelved |
| Int5 for MLP | 0.029 quant penalty vs 0.010 for int6 | Dead end |
| Bigram=8192 | Best BPP but over 16MB | Need better compression |
| 12L MLP=1280 | Worse than 12L/1408 | Dead end |
| 11L MLP=1280 | Worse than 10L/1536 | Dead end |

## Principles

1. Outsized returns come from OUTSIDE the training loop: eval time, compression, architecture
2. Throughput is king: any technique adding >5ms/step must prove massive per-step quality gain
3. The 390s idle eval budget is the biggest untapped resource
4. Test one thing at a time, measure precisely, log immediately
5. Keep at least 10 untried ideas. If below 10, brainstorm before continuing
6. Read new PRs before every session — the competition moves fast
