# Submission Checklist

## Required Files (in records/track_10min_16mb/<date>_<name>/)

- [ ] `train_gpt.py` — self-contained script that reproduces the run
- [ ] `README.md` — explains the approach (see template below)
- [ ] `submission.json` — metadata (see template below)
- [ ] `train.log` — exact training log from the 8xH100 run

## Pre-Submission Verification

- [ ] Script runs clean on 8xH100: `torchrun --standalone --nproc_per_node=8 train_gpt.py`
- [ ] Completes in under 10 minutes (MAX_WALLCLOCK_SECONDS=600)
- [ ] Total submission size (code + int8+zlib model) < 16,000,000 bytes
- [ ] val_bpb beats current SOTA by at least 0.005 nats
- [ ] Multiple runs to show p < 0.01 significance (at least 3-5 seeds)
- [ ] No external downloads or network calls during training/eval
- [ ] Script under 1500 lines

## submission.json Template
```json
{
  "author": "Samuel Larson",
  "github_id": "samuellarson",
  "name": "<descriptive name>",
  "blurb": "<1-2 sentence description of approach>",
  "date": "<YYYY-MM-DD>",
  "val_loss": <float>,
  "val_bpb": <float>,
  "bytes_total": <int>,
  "bytes_code": <int>
}
```

## README.md Template
```markdown
# <Submission Name>

<1-paragraph summary of approach and key innovations>

## Configuration
- Layout: `VOCAB_SIZE=X NUM_UNIQUE_BLOCKS=X RECURRENCE_DEPTH=X MODEL_DIM=X ...`
- Key hyperparameters: `MATRIX_LR=X GRAD_CLIP_NORM=X WARMDOWN_ITERS=X ...`
- Eval: `EVAL_SEQ_LEN=X`

## Command
\```bash
RUN_ID=<name> \
<env vars> \
torchrun --standalone --nproc_per_node=8 train_gpt.py
\```

## Key Metrics
- Steps completed: X/Y
- Pre-quant: val_loss=X, val_bpb=X
- Post-quant: val_loss=X, val_bpb=X
- Model size (int8+zlib): X bytes
- Code size: X bytes
- Total: X bytes
- Train time: Xms, step_avg=Xms
- Peak memory: X MiB

## Approach Details
<Explain what's different from the baseline and why it helps>

## Reproduction
<Exact steps to reproduce from a fresh RunPod pod>
```

## PR Process
1. Fork the repo (or push branch)
2. Create PR that ONLY adds new folder to `records/track_10min_16mb/`
3. PR title: short, descriptive
4. PR body: summary + test plan
5. Wait for review — top entries get verified by OpenAI
