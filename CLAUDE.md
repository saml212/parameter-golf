# Parameter Golf Competition - Project Guide

## What This Is
OpenAI Parameter Golf: train the best LM in 16MB, 10 min on 8xH100. Metric: val_bpb (lower = better).
Baseline: 1.2244 BPB. Current valid leader: PR #414 at 1.1233. Our best: PR #332 at 1.1320. Gap: 0.009.

## CRITICAL CONTEXT: What Happened Since PR #332
Our last experiment session ended at PR #332 (1.1320, 12L). Since then the competition moved fast. Here is what you need to know:

### The Valid Leaderboard (2026-03-22 evening)
| PR | BPB | Author | Approach | Valid? |
|----|-----|--------|----------|--------|
| #473 | 1.1213 | abaybektursun | #414 stack + legal score-first TTT | VALID (1 seed only) |
| **#414** | **1.1233** | signalrush | 11L + GPTQ-lite + EMA + Tight SWA + QAT@0.15 + U-Net skips | **VALID (3-seed)** |
| #394 | 1.1247 | greqone | #315 + Backout connection | VALID |
| #315 | 1.1250 | jfprincz | 11L + Partial RoPE + LN Scale + XSA4 + EMA | VALID |
| **#332** | **1.1320** | **us (saml212)** | 12L + GradQuant + Partial RoPE + XSA4 + EMA | VALID |

NOTE: PRs #442 (1.1027), #462 (1.0672), #398 (1.1221) all use full-val-set TTT (train on val tokens before scoring them). This was ruled invalid by @0hq in PR #152. Only backward-looking "score-first" TTT is legal. The valid frontier is PR #414 at 1.1233.

### New Techniques That Emerged (adopt these)
1. **GPTQ-lite (PR #379, #414)**: Instead of row-max for int6 scale, try 5 clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) per row and pick the one minimizing reconstruction MSE. Zero training cost. -0.0006 BPP. In the #414 stack.
2. **Backout Connection (PR #339, #394)**: Learned scalar lambda (init=0.2) subtracts mid-layer hidden state from final output. 1 parameter. -0.003 BPP on #315 base. PR #394 got 1.1247 with it.
3. **U-Net Skip Connections (PR #289, #295, #414)**: Skip connections across layers (like U-Net encoder-decoder). In the #414 stack.
4. **Tight SWA + EMA combined**: #414 uses both SWA and EMA together, not EMA replacing SWA.
5. **Value Residual Learning (PR #413)**: Cache layer 0 V vectors, add to all subsequent layers via learned scalars. 18 params. **-0.015 BPP in ablation on small base.** Stacks with gated attention. arXiv:2410.17897 (ACL 2025). NOT YET TESTED on 11L SOTA stack — someone is running it now. This is the biggest untested opportunity.
6. **Gated Attention (PR #413)**: Per-head sigmoid gate after SDPA. ~37K params. -0.003 BPP. Stacks with value residual for -0.017 combined. arXiv:2505.06708 (NeurIPS 2025 Best Paper).
7. **Catalytic Residuals (PR #450)**: Replace `x + f(x)` with `x + c*f(x)` where c is learned per-dimension. ~11K params. -0.024 BPP in ablation. Zero compute cost.
8. **Late QAT threshold 0.15** (not 0.10): #414 uses 0.15. Only enable STE fake-quant when lr_scale < 0.15 instead of 0.10.
9. **Warmdown=3500** (not 3000): #414 uses 3500.
10. **Weight Entropy Regularization (PR #459)**: `loss += lambda * weight_entropy` during warmdown improves SWA averaging. +0.028 BPP. Only tested on 1xH100. Unvalidated on 8xH100.

### Techniques Nobody Has Tried (our novel opportunities)
- **Hadamard rotation before int6 quantization** — flatten weight distribution pre-quant. Zero inference cost (fuses into weights). QuaRot/SpinQuant literature proven 30-50% quant error reduction. Could stack on top of GPTQ-lite. arXiv:2512.24124
- **Temperature scaling post-quantization** — single learned scalar T to recalibrate logits after int6. Seconds to calibrate. arXiv:2409.19817
- **Mousse optimizer** — curvature-aware Muon. ~3% overhead, claims 12% better sample efficiency. arXiv:2603.09697
- **Batch size warmup** — start at 128-256K, ramp to 524K. More gradient updates early when LR is high. Zero per-step cost.

### Confirmed Dead Ends (from community since #332)
- **PPM-C classical compression mixer**: +0.0018 BPP (NEGATIVE) on SmearGate models. Confirmed by PR #413. Do not use.
- **Full-val-set TTT on XSA+EMA base with SGD**: neutral or negative (PR #303, #338). AdamW TTT works but is invalid.
- **Legal score-first TTT**: gives only ~0.002 BPP on top of #414 stack (PR #473). Save for last.
- **Depth recurrence**: quant error amplifies ~900x over 3 recurrence cycles (PR #363). Dead.
- **SwiGLU**: confirmed negative by PR #340, #344 independently. Dead.
- **12L at seq2048**: our own finding. 12L loses to 11L despite more steps because Late QAT can't be used.

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

## Experiment Roadmap

### Phase 1: Get to Parity with #414 (target: 1.123)
Start from PR #414's code or the closest community stack. Do NOT build from our #332 code — start from the leader's stack and verify it reproduces.
1. Reproduce #414's result (1.1233) on our hardware. Expect ~1.124 with cold cache.
2. Verify warm cache gives ~1.123.
3. This is our new baseline. All further experiments branch from here.

### Phase 2: Stack Proven Zero-Cost Wins (target: 1.120)
Add one at a time, measure each independently:
4. **Value Residual Learning** — cache layer 0 V, add to all layers. 18 params. -0.015 BPP in ablation. HIGHEST PRIORITY.
5. **Gated Attention** — per-head sigmoid gate. ~37K params. -0.003 BPP. Stacks with VRL.
6. **Catalytic Residuals** — `x + c*f(x)`. ~11K params. -0.024 BPP in ablation.
7. **Backout Connection** — if not already in #414 stack. 1 param. -0.003 BPP.

### Phase 3: Novel Quant Improvements (target: 1.118)
8. **Hadamard rotation before GPTQ-lite** — nobody has tried this. Flatten weight distribution before clip percentile search. The two techniques attack quant error from orthogonal angles. If they stack, this is a genuine finding.
9. **Temperature scaling post-quant** — single scalar T. Seconds to calibrate.

### Phase 4: Novel Training Improvements (target: 1.115)
10. **Mousse optimizer** — drop-in Muon replacement. 3% overhead, potentially 12% better samples.
11. **Batch size warmup** — start 128-256K, ramp to 524K.
12. **Weight Entropy Regularization** — if using SWA component.

### Phase 5: Legal TTT (last resort, target: 1.113)
13. Only after all above are exhausted. Score-first chunked TTT with SGD+momentum, freeze early blocks. Expected gain: ~0.002 BPP. Use PR #461's recipe.

### Phase 6: Moonshots
14. **KV cache reuse across sliding windows** — carry forward KV cache for 50K+ token context. Nobody has built this. Could enable stride=1.
15. **Entropy-coded weights** — replace zstd with ANS/Huffman. Free 1-2MB for more params.

## Our PRs
- #61: 1.2154 BPB (warmdown-quantization discovery)
- #96: 1.1764 BPB (sliding window + long-context training)
- #114: 1.1574 BPB (int6 + MLP 3x + selective precision)
- #236: 1.1400 BPB (11L + SmearGate + BigramHash + SWA)
- #332: 1.1320 BPB (12L + GradQuant + Partial RoPE + XSA4 + EMA)

## Proven Losses (DO NOT RE-TEST)
- ALBERT embedding factorization (torch.compile overhead kills 32% of steps)
- NorMuon optimizer (110ms/step, throughput death)
- MTP / multi-token prediction (86ms/step, aux head too expensive)
- SwiGLU at iso-params (confirmed dead by PR #340, #344)
- Depth recurrence (quant error amplifies ~900x)
- Full-val TTT on XSA+EMA base (invalid + doesn't help)
- Late QAT at 12L (overhead costs ~770 steps)
- Cosine warmdown (multiple teams, doesn't beat linear)
- PPM-C mixer on SmearGate models (negative, PR #413)
- INT4 all (0.06 BPP quant gap)
- cuDNN SDP (worse BPB despite faster)
- Step-based LR schedule (catastrophic)
- Block-wise weight sharing (quant error amplifies)
- eval@2048 on well-trained models (NTK distortion)
- Batch=1M (too few steps)

## Key Interaction Effects
- EMA needs XSA to work (EMA alone hurts on non-XSA base)
- SmearGate needs OrthoInit (hurts by 0.003 without it)
- Late QAT works at 11L but hurts at 12L (step budget dependent)
- Value Residual + Gated Attention stack additively (-0.017 combined, PR #413)
- GPTQ-lite and Hadamard rotation attack quant error from different angles — may stack

## Key Insight: Step Throughput Is King
On a 10-min budget, per-step overhead >10% is a net loss. Only MLP 3x has ever compensated.

## Cloud Compute
- RunPod pods with H100 SXM GPUs
- **Official template**: Python 3.12 + PyTorch 2.9.1+cu128
- SP1024 data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/`
- DISK: Pod has 50GB. Clean /tmp/torchinductor_root periodically.
- **torch.compile cache**: First run ~100ms/step. After 2-3 warmup runs: ~67-82ms/step. ALWAYS warmup first.

## Git Workflow
- `sam/base` — consolidated working branch with all docs, records, scripts
- For new PRs: branch off `sam/base`, make submission, PR to main
- Keep `sam/base` up to date after each experiment session
- NEVER sign commits with AI attribution. No Co-Authored-By, no AI fingerprints.
