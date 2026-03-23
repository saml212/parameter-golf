# Idea Bank — Parameter Golf Competition

## Leverage Table — Where Are the Outsized Returns?

| Lever | Current | Max | Gap | Status |
|-------|---------|-----|-----|--------|
| **Quant quality** | GPTQ-lite (clip search) | +Hadamard rotation | **Untested** | Nobody has tried rotation. Could stack with GPTQ-lite. |
| **Architecture** | 11L + XSA + EMA | +Value Residual +Catalytic Res +Gated Attn | **Partially tested** | -0.015, -0.024, -0.003 in ablations. Not on SOTA stack. |
| **Post-quant calibration** | None | Temperature scaling | **Untested** | Single scalar, seconds to calibrate |
| **Optimizer** | Muon + AdamW | Mousse | **Untested** | 3% overhead, 12% better sample efficiency |
| **Eval time** | ~200s | 600s | **400s idle** | KV cache reuse could enable stride=1 |
| **Training schedule** | Linear warmdown 3500 | +Batch warmup | **Untested** | Start small batch, ramp up. Zero per-step cost. |
| **Compression** | int6 + zstd-22 | Entropy coding | **Untested** | Could free 1-2MB for more params |
| **Legal TTT** | Not used | Score-first chunked | **~0.002 BPP** | Save for last. PR #473 showed small gain on #414. |

## Phase 1: Parity (target 1.123)
- [ ] Reproduce PR #414 result on our hardware
- [ ] Verify warm cache performance

## Phase 2: Zero-Cost Architecture Wins (target 1.120)
- [ ] [NOVEL on SOTA] **Value Residual Learning** — -0.015 BPP ablation. 18 params. arXiv:2410.17897
- [ ] [NOVEL on SOTA] **Catalytic Residuals** — -0.024 BPP ablation. ~11K params. PR #450
- [ ] [NOVEL on SOTA] **Gated Attention** — -0.003 BPP ablation. ~37K params. arXiv:2505.06708
- [ ] [ADAPTED] **Backout Connection** — -0.003 BPP. 1 param. PR #339, #394

## Phase 3: Novel Quant (target 1.118)
- [ ] [NOVEL] **Hadamard rotation + GPTQ-lite** — nobody has tried. Orthogonal angles of attack.
- [ ] [NOVEL] **Temperature scaling** — single scalar post-quant calibration. arXiv:2409.19817

## Phase 4: Novel Training (target 1.115)
- [ ] [NOVEL] **Mousse optimizer** — drop-in Muon replacement. arXiv:2603.09697
- [ ] [NOVEL] **Batch size warmup** — zero cost. arXiv:2505.23971
- [ ] [ADAPTED] **Weight Entropy Regularization** — PR #459. Needs 8xH100 validation.

## Phase 5: Legal TTT (last resort)
- [ ] [ADAPTED] **Score-first chunked TTT** — PR #461 recipe. SGD+momentum, freeze blocks 0-1. ~0.002 BPP.

## Tried and Dead
| Idea | Result | Tag |
|------|--------|-----|
| PPM-C mixer | +0.0018 (NEGATIVE) on SmearGate | DEAD |
| Full-val TTT on XSA+EMA | Invalid + neutral/negative | DEAD |
| SwiGLU | Confirmed dead by PR #340, #344 | DEAD |
| Depth recurrence | Quant error amplifies ~900x | DEAD |
| 12L at seq2048 | Loses to 11L (Late QAT can't be used) | DEAD |
| Late QAT at 12L | -770 steps, net negative | DEAD |
| Cosine warmdown | Multiple teams, doesn't beat linear | DEAD |

## Principles
1. Outsized returns come from OUTSIDE the training loop: eval time, compression, architecture
2. Step throughput is king — anything >10% overhead per step is a net loss
3. Zero-cost techniques first: value residuals, Hadamard rotation, temp scaling
4. The leader wins by finding zero-cost wins from papers, not by grinding hyperparameters
5. Read Issue #140 before every session — it tracks what's been tried community-wide
6. Start from the leader's code, not our old code. Build forward, not sideways.
