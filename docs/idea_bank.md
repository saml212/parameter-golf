# Idea Bank — Parameter Golf Competition

## Leverage Table — Where Are the Outsized Returns?

| Lever | Current | Max | Gap | Status |
|-------|---------|-----|-----|--------|
| **Training time** | 600s | 600s | 0 | Maxed out |
| **Eval time** | ~200s | 600s | **400s idle** | KV cache reuse, PPM-C, stride=1 |
| **Artifact bytes** | 15.7MB | 16.0MB | **300KB** | Entropy coding could free 1-2MB |
| **Model architecture** | 12L Transformer + XSA | Novel ops | **Open** | Value residuals, gated attention, memory tokens |
| **Quantization quality** | Per-row int6 | Hadamard+int6 | **Untested** | Rotation before quant, nobody has tried |
| **Post-quant calibration** | None | Temp scaling | **Untested** | Single scalar, seconds to calibrate |
| **Optimizer** | Muon + AdamW | Mousse | **Untested** | 3% overhead, 12% better sample efficiency |
| **Training data ordering** | Sequential shards | Curriculum/shuffle | **Untapped** | Blocked by SWA incompatibility, but EMA fixes that |
| **Compression scheme** | zstd-22 | Entropy coding | **Untested** | Could free 1-2MB for more params |
| **Layer count** | 12L (74ms) vs 11L (85ms) | — | **Strategic** | 11L + Late QAT beats 12L. Revert? |

**Strategy**: The 11L vs 12L decision is the most important. After that: zero-cost architectural additions (value residuals, memory tokens), then quantization improvements (Hadamard rotation, temp scaling), then eval-time exploitation.

## NOVELTY CHECK — Before Every Novel Idea

```bash
gh pr list --repo openai/parameter-golf --state open --limit 50 --json number,title,body | grep -i "<keyword>"
```
If someone submitted it, read their PR, learn from it, figure out how to do it BETTER.

Tags:
- [NOVEL] — nobody has tried this (verified via PR search)
- [ADAPTED] — someone tried similar, we're improving it
- [COMMODITY] — table stakes, everyone does this
- [DEAD] — proven not to work, do not re-test

## Priority 1: Close the 0.007 Gap to #315 (immediate)

### Path A: Match #315's stack on 11L
- [ ] [ADAPTED] Revert to 11L, drop gradient-guided quant, use uniform int6 + Late QAT
- [ ] [ADAPTED] Late QAT at 11L (STE fake-quant last 4% of training) — works at 11L, proven by #315
- [ ] [ADAPTED] Test RoPE base=50K — zero cost, used by multiple top submissions

### Path B: Add zero-cost wins to beat #315
- [ ] [NOVEL] **Value Residual Learning** — cache layer 0 V, add to all subsequent layers. Zero params. ACL 2025. arXiv:2410.17897. Nobody in competition uses this.
- [ ] [NOVEL] **Hadamard rotation before int6 quantization** — flatten weight distribution pre-quant. Zero inference cost (fuses into weights). QuaRot/SpinQuant literature. arXiv:2512.24124
- [ ] [NOVEL] **Temperature scaling post-quantization** — single scalar T, calibrate in seconds. arXiv:2409.19817
- [ ] [NOVEL] **Memory Tokens (PR #352)** — 64 learned prefix embeddings as global scratchpad. -0.014 BPP on 10L base. ~8K params. Untested on 11L XSA+EMA.
- [ ] [NOVEL] **Backout Connection (PR #339)** — learned scalar subtracts mid-layer h from output. 1 param, -0.007 BPP in controlled test. Interaction with XSA unknown.

## Priority 2: Optimizer and Training Improvements

- [ ] [NOVEL] **Mousse optimizer** — curvature-aware Muon. Shampoo preconditioning before orthogonalization. ~3% overhead, 12% better sample efficiency. arXiv:2603.09697. Drop-in Muon replacement.
- [ ] [NOVEL] **Batch size warmup** — start 128-256K, ramp to 524K. More gradient updates early when LR high. Zero per-step cost. Nobody has tried. arXiv:2505.23971
- [ ] [NOVEL] **Cautious Weight Decay** — WD only where sign-aligned with gradient. 1 line change. arXiv:2510.12402
- [ ] [ADAPTED] **Warmdown shape** — test sqrt decay instead of linear. ICML 2025 says shape matters. Zero cost. arXiv:2601.09000
- [ ] [NOVEL] **Gated Attention** — per-head sigmoid gate. ~50K params, <2% overhead. NeurIPS 2025 Best Paper (Qwen). arXiv:2505.06708

## Priority 3: Eval-Time Exploitation (400s of idle eval time)

- [ ] [NOVEL] **KV cache reuse across sliding windows** — reuse 97% of KV cache between stride-64 windows. Enable stride=1 within eval budget. PR #318 concept, never tested. FA3 supports seqlen_k > seqlen_q.
- [ ] [NOVEL] **PPM-C classical compression mixer** — blend byte-level PPM with neural logprobs at eval. ~0.003-0.008 BPP on weak bases. PR #283 showed it works. Zero artifact cost.
- [ ] [ADAPTED] **Eval stride=32** — ~0.003 BPP over stride=64. 2x eval time but budget permits.
- [ ] [NOVEL] **Multi-checkpoint logit ensemble** — average logits from 3-4 training checkpoints at eval. Different from SWA (weight avg). Eval budget can fit 3-4 forward passes.

## Priority 4: Compression and Artifact

- [ ] [NOVEL] **Entropy-coded weights** — replace zstd with ANS/Huffman exploiting quantized weight distribution. Free 1-2MB for more params. Decoder must fit in artifact. arXiv:2505.02380
- [ ] [NOVEL] **Entropy-regularized QAT** — add compression penalty that clusters quantized weights. Saves 0.5-1.5MB. arXiv:2505.18758
- [ ] [ADAPTED] **Magnitude pruning 5-8%** — zero smallest weights before zstd. PR #349 uses it. Zero training cost.
- [ ] [ADAPTED] **zstd dictionary training** — train zstd dict on weight tensors. Better compression for small files.
- [ ] [NOVEL] **OptRot learned rotations** — data-free learned rotation matrices. Better than random Hadamard. arXiv:2512.24124

## Priority 5: Higher Risk / Higher Reward

- [ ] [NOVEL] **Differential Attention** — difference of two softmax maps. 5-10% overhead (risky). ICLR 2025. arXiv:2410.05258
- [ ] [NOVEL] **WaveletGPT** — multi-scale Haar wavelet on half of embedding dims. Zero params, 40-60% faster convergence claimed. arXiv:2409.12924. Overhead unclear.
- [ ] [NOVEL] **Partial weight sharing + 14L** — share middle-layer pairs with LoRA adapters. Saves ~3-5MB for more depth.
- [ ] [NOVEL] **Knowledge distillation** — train teacher 7 min, distill to student 3 min. High complexity.

## Tried (Record Results Here)

| Idea | Result | Takeaway | Tag |
|------|--------|----------|-----|
| 10L int5-MLP (cold cache) | 1.1758 | Cold cache 111ms/step killed it | COMMODITY |
| SmearGate + BigramHash | 1.1644 | +0.011 improvement on 10L | COMMODITY |
| QAT int6 STE (full training) | 1.1755 | NET LOSS — 115ms/step overhead | DEAD |
| Int6-all (not int5-MLP) | 1.1465 | Quant penalty 0.010 vs 0.029. Better | ADAPTED |
| Batch=524K (from 786K) | 1.1465 | 63ms/step, 9400 steps. Throughput win | ADAPTED |
| Batched sliding eval stride=64 | ~0.004 gain | 32 windows at once, 172s | ADAPTED |
| 11L MLP=1280 | 1.1480 | Too narrow — worse than 10L/1536 | COMMODITY |
| **11L MLP=1408** | **1.1444 (3-seed)** | Sweet spot depth x width | COMMODITY |
| FlashAttention 2.8.3 | 1.1429 | 66ms vs 68ms, ~200 more steps | ADAPTED |
| XSA + EMA (11L) | 1.1350 | +0.005 over SWA base | ADAPTED |
| 12L + GradQuant + PartialRoPE | 1.1320 | **PR #332.** But 11L+LateQAT beats it | ADAPTED |
| Late QAT at 12L | 1.1361 | NET LOSS — 7ms/step costs 770 steps | DEAD |
| EMA without XSA | worse | EMA needs XSA to work (PR #201) | DEAD |
| TTT on XSA+EMA base | worse | +0.016 BPP worse (PR #303) | DEAD |
| Cosine warmdown | 1.1704 | Multiple teams, doesn't beat linear | DEAD |

## Principles
1. Outsized returns come from OUTSIDE the training loop: eval time, compression, architecture
2. If something works, try MORE of it until it stops working
3. Step throughput is king — anything >10% overhead per step is a net loss
4. Zero-cost techniques first: value residuals, Hadamard rotation, temp scaling
5. Before declaring an idea dead, check if cold cache was the problem
6. Read Issue #140 before every session — it tracks what's been tried community-wide
7. The leader wins by finding zero-cost wins from papers, not by grinding hyperparameters

---
Maintain at least 10 untried ideas at all times. If below 10, brainstorm before continuing.
Every 10 experiments, refresh novelty tags by checking latest PRs.
