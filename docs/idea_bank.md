# Idea Bank — Parameter Golf Competition

## Leverage Table — Where Are the Outsized Returns?

| Lever | Current | Max | Gap | Status |
|-------|---------|-----|-----|--------|
| **Training time** | 600s | 600s | 0 | Maxed out |
| **Eval time** | ~210s | 600s | **390s idle** | Biggest untapped resource |
| **Artifact bytes** | 15.7MB | 16.0MB | **300KB** | Could fit more params with better compression |
| **Model architecture** | 12L transformer + XSA | Novel ops | **Open** | Partial RoPE, LN Scale tested |
| **Training data ordering** | Sequential shards | Curriculum/shuffle | **Untapped** | Research says 0.001-0.005 BPP possible |
| **Compression scheme** | Gradient-guided int5/6/7 | Bit-packing, codebooks | **Partially explored** | Our novel contribution |
| **Weight averaging** | EMA (0.997) | — | Small | Near-optimal |
| **Quantization timing** | Post-training + gradient-guided | Late QAT | **Tested** | Late QAT hurts at 12L |
| **Eval methodology** | Sliding stride=64 | Ensemble, multi-stride | **Untapped** | Checkpoint ensemble possible |

## Priority Queue — Next Experiments

### High Priority (likely 0.002+ BPP each)
- [ ] **Checkpoint ensemble at eval**: Save EMA + final weights, eval both, average logits. Uses idle eval time. Novel.
- [ ] **Bit-packed int6**: Pack 6-bit values natively (6 values in 4.5 bytes vs 6 bytes in int8 containers). 25% better compression. Enables 13L or MLP=1536 at 12L.
- [ ] **Multi-stride eval ensemble**: Run stride=64 AND stride=32. Average per-position BPP. Uses idle eval time.

### Medium Priority (0.001-0.002 BPP)
- [ ] Data curriculum-as-warmup: Easy data for first 1000 steps, then random
- [ ] MLP=1472 at 12L (between 1408 and 1536)
- [ ] EMA decay sweep: 0.995, 0.998, 0.999
- [ ] Partial RoPE dims sweep: 8, 16, 24, 32
- [ ] LN Scale exponent sweep: 1/sqrt(i+1) vs 1/(i+1)^0.3

### Low Priority / Speculative
- [ ] Data shard shuffling (research says 0.001 BPP max)
- [ ] Attention sink tokens (learned prefix tokens)
- [ ] Multi-head value mixing matrix
- [ ] SP4096 tokenizer angle at 12L

## Tried and Resolved

| Idea | Result | Verdict |
|------|--------|---------|
| Gradient-guided quant | Enables 12L, saves ~1MB | **SHIPPED in current PR** |
| Partial RoPE (16/64 dims) | ~0.001 improvement | Shipped |
| LN Scale (1/sqrt(layer+1)) | ~0.001 improvement | Shipped |
| Late QAT at 12L | 1.1361 (WORSE than 1.1321) | **Dead end at 12L** — throughput cost |
| Cyclic SWA | 1.1466 (worse than linear) | Dead end |
| Progressive layer freezing | 1.4321 (catastrophic) | Dead end |
| QAT int6 STE (full training) | 115ms/step, net loss | Dead end |
| Train@1024 | 1.1811 (halved context kills quality) | Dead end |
| Pre-eval TTT | Ruled illegal by competition | Dead end (legal causal TTT exists) |
| Causal online TTT | 1.1322 (legal, stride=128) | Works but politically risky — shelved |
| Bigram=8192 | Best BPP but over 16MB | Need better compression first |
| Int5 for MLP | Worse than int6-all (0.029 vs 0.010 quant penalty) | Dead end |
| Batch=786K | Fewer steps than 524K | 524K is optimal |

## Principles

1. Outsized returns come from OUTSIDE the training loop: eval time, compression, architecture
2. Throughput is king: any technique adding >5ms/step must prove massive per-step quality gain
3. The 390s idle eval budget is the biggest untapped resource in the competition
4. Test one thing at a time, measure precisely, log immediately
5. Keep at least 10 untried ideas. If below 10, brainstorm before continuing.
