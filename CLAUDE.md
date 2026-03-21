# Parameter Golf Competition - Project Guide

## What This Is
OpenAI Parameter Golf: train the best LM in 16MB, 10 min on 8xH100. Metric: val_bpb (lower = better).
Baseline: 1.2244 BPB. Current leader: PR #315 at 1.1250. Our best: PR #332 at 1.1320.

## Current Best Config (verified, 3-seed) — PR #332
```
NUM_LAYERS=12 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 MLP_HIDDEN=1408
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3
EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32
+ Gradient-Guided Adaptive Quantization (int7/int6/int5 by gradient sensitivity)
+ FP16 tied embedding + Late-K passthrough (last 2 layers c_k)
+ SmearGate (per-dim learned gate, 512 params)
+ BigramHash (2048 buckets, dim=128)
+ OrthoInit + MuonWD=0.04
+ EMA (decay=0.997) replacing SWA
+ XSA on last 4 layers (zero params)
+ Partial RoPE (16/64 head dims)
+ LN Scale (1/sqrt(layer_idx+1))
+ zstd-22 compression
+ Batched sliding window eval (32 windows at once)
```
On 8xH100: **1.1320 BPB** (3-seed mean, std=0.0002). 27.6M params in 15.7MB. 8,060 steps at 74ms/step.
Seeds: 1337→1.1321, 1338→1.1321, 1339→1.1318.

## Gap to Leader: 0.007 BPB
PR #315 (@jfprincz): 1.1250 BPB with 11L + Partial RoPE + LN Scale + Late QAT + XSA4 + EMA.
Key difference: he uses 11L with Late QAT. Our 12L gets more steps but Late QAT hurts at 12L (~770 steps lost).

## Competition Landscape (2026-03-21)
| PR | BPB | Author | Approach |
|----|-----|--------|----------|
| **#315** | **1.1250** | jfprincz | 11L + Partial RoPE + LN Scale + Late QAT + XSA4 + EMA |
| #338 | 1.1256 | alertcat | #315 stack + TTT (neutral on this base) |
| #287 | 1.1280 | jfprincz | 11L + XSA4 + EMA + SmearGate + BigramHash |
| #254 | 1.1313 | timowhite88 | 11L + TTT + RoPE50K + SWA (pre-eval TTT ruled invalid) |
| **#332** | **1.1320** | **us** | 12L + GradQuant + Partial RoPE + LN Scale + XSA4 + EMA |
| #198 | 1.1326 | jfprincz | 11L + SmearGate + BigramHash + WD 0.04 + SWA |
| #236 | 1.1400 | us | 11L + SmearGate + BigramHash + SWA |
| #180 | 1.1428 | thwu1 | 10L Int5-MLP + BigramHash(10240) + SWA (official SOTA) |

## Our PRs
- #61: 1.2154 BPB (warmdown-quantization discovery)
- #96: 1.1764 BPB (sliding window + long-context training)
- #114: 1.1574 BPB (int6 + MLP 3x + selective precision)
- #236: 1.1400 BPB (11L + SmearGate + BigramHash + SWA)
- #332: 1.1320 BPB (12L + GradQuant + Partial RoPE + XSA4 + EMA)

## Next Experiment Priorities (see docs/idea_bank.md for full list)

### Tier 1: Highest EV, zero/near-zero cost
1. **Revert to 11L + Late QAT** — #315's stack beats our 12L. Drop gradient-guided quant, use uniform int6.
2. **Value Residual Learning (ResFormer)** — cache layer 0 V vectors, add to all layers. Zero params. ACL 2025. Nobody in competition uses it. arXiv:2410.17897
3. **Hadamard rotation before int6 quantization** — flatten weight distribution, reduce outlier quant error. Post-training, zero cost. QuaRot/SpinQuant proven 30-50% quant error reduction.
4. **Temperature scaling post-quantization** — single learned scalar T. Seconds to calibrate. Free BPB.
5. **Eval stride=32** — ~0.003 BPP over stride=64 if eval budget permits.

### Tier 2: Worth testing, low cost
6. **Mousse optimizer** — drop-in Muon replacement, ~3% overhead, 12% better sample efficiency. arXiv:2603.09697
7. **Batch size warmup** — start at 128-256K, ramp to 524K. More early gradient updates. Zero per-step cost.
8. **Memory Tokens (PR #352)** — 64 learned prefix embeddings. -0.014 BPP on 10L base. ~8K params. Untested on 11L XSA+EMA.
9. **Gated Attention** — per-head sigmoid gate. ~50K params. NeurIPS 2025 Best Paper. arXiv:2505.06708
10. **Backout Connection (PR #339)** — learned scalar subtracts mid-layer hidden state. 1 param, -0.007 BPP controlled.

### Tier 3: Moonshots
11. **KV cache reuse across sliding windows** — enable stride=1 eval within budget.
12. **PPM-C classical compression mixer** — blend classical predictor with neural logprobs at eval.
13. **Entropy-coded weights** — replace zstd with ANS/Huffman for int6 weights. Free 1-2MB.

## Proven Wins (ranked by impact, 8xH100)
1. Sliding window eval stride=64 (+0.034 BPB)
2. TRAIN_SEQ_LEN=2048 with LR=0.02, momentum=0.99 (+0.020 vs stock 1024)
3. Int6 quantization + MLP 3x (hidden=1536, 21.8M params in 16MB)
4. XSA on last 4 layers (zero params, zero cost)
5. Partial RoPE 16/64 dims (zero params, zero cost)
6. LN Scale 1/sqrt(layer+1) (zero params, zero cost)
7. EMA decay=0.997 (replaces SWA on XSA+EMA base)
8. SmearGate + BigramHash + OrthoInit (+0.011 combined)
9. MuonWD=0.04 (+0.005 and improves quant robustness)
10. GRAD_CLIP_NORM=0.3 (+0.002, specifically helps long-sequence training)
11. FP16 tied embedding + Late-K passthrough (last 2 layers c_k in fp16)
12. Batch=524K with 11L (22% more steps than 786K)

## Proven Losses (DO NOT RE-TEST)
- ALBERT embedding factorization (torch.compile overhead kills 32% of steps)
- NorMuon optimizer (110ms/step, throughput death on our hardware)
- MTP / multi-token prediction (86ms/step, aux head too expensive)
- SwiGLU at iso-params (narrower hidden doesn't beat ReLU²)
- WD=20000 with low LR (doesn't stack — low LR already smooths weights)
- NUM_KV_HEADS=2 (capacity loss not worth param savings)
- Int6 step=4 rounding with LR=0.06 (catastrophic — needs proper per-row ±31)
- Depth recurrence (quant error amplifies ~900x over 3 cycles)
- eval@2048 on well-trained 8xH100 models (NTK distortion)
- train@4096 vs train@2048 (identical with sliding window, 2048 gets more steps)
- Batch=1M (too few steps)
- WD=30000 (decays too fast)
- QK_GAIN_INIT=1.7 (no improvement over 1.5)
- Late QAT at 12L (overhead costs ~770 steps, net negative)
- Cosine warmdown (multiple teams tried, doesn't beat linear)
- TTT on XSA+EMA base (PR #303: +0.016 worse. XSA and TTT extract same signal)
- Block-wise weight sharing (quant error amplifies, no artifact savings)
- cuDNN SDP (40% faster but worse BPB — precision issues)
- Step-based LR schedule (catastrophic — can't adapt to hardware-dependent wallclock)
- INT4 all (0.06 BPP quant gap, strictly worse than INT6 with fewer params)

## Key Insight: Step Throughput Is King
On a 10-min budget, per-step overhead >10% is a net loss. Only MLP 3x has ever compensated. Everything else that traded steps for per-step quality failed.

## Key Interaction Effects
- WD=20000 helps with high LR (0.06) but hurts with low LR (0.02)
- EMA needs XSA to work (EMA alone hurts on non-XSA base)
- SmearGate needs OrthoInit (hurts BPB by 0.003 without it)
- TTT helps weak bases, neutral on frontier, hurts XSA+EMA bases
- Late QAT works at 11L but hurts at 12L (step budget dependent)
- Int5-MLP tradeoff is layer-dependent (works at 10L to fund BigramHash, loses at 11L)

## Cloud Compute
- RunPod pods with H100 SXM GPUs
- **Official template**: Python 3.12 + PyTorch 2.9.1+cu128
- SP1024 data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/`
- SP4096 data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp4096/`
- DISK: Pod has 50GB. Clean /tmp/torchinductor_root periodically.
- **torch.compile cache**: First run ~100ms/step. After 2-3 warmup runs: ~67-82ms/step. ALWAYS warmup first.

## Git Workflow
- `sam/base` — consolidated working branch with all docs, records, scripts
- For new PRs: branch off `sam/base`, make submission, PR to main
- Keep `sam/base` up to date after each experiment session
