# Environment Setup Guide

## RunPod Setup (1xH100 for iteration)

### Create Pod
1. Go to https://console.runpod.io → Pods → Deploy
2. Select **H100 SXM** ($2.69/hr Community, $3.29/hr Secure)
3. GPU count: **1** for iteration, **8** for final submission
4. Template: "Runpod Pytorch 2.4.0" (default) — we upgrade PyTorch after
5. Enable: SSH terminal access + Jupyter notebook
6. Storage: 20GB container disk, network volume recommended for persistence

### SSH Key Setup
Add your public key to RunPod Settings → SSH Public Keys:
```
cat ~/.ssh/id_ed25519.pub
```

### First-Time Pod Setup
```bash
# Upgrade PyTorch (required — default 2.4.0 lacks enable_gqa for GQA)
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124

# Clone repo
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Install dependencies
pip install -q sentencepiece huggingface-hub datasets tqdm

# Download data (10 shards for iteration, 80 for full training)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

### Running Experiments
```bash
# Our best config (replace RUN_ID for each experiment)
RUN_ID=experiment_name \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=500 \
VAL_LOSS_EVERY=2000 \
WARMDOWN_ITERS=2400 \
GRAD_CLIP_NORM=1.0 \
MATRIX_LR=0.06 \
EVAL_SEQ_LEN=2048 \
torchrun --standalone --nproc_per_node=1 train_gpt_evalonly.py
```

For 8xH100: change `--nproc_per_node=8`

### Checking Results
```bash
# Final BPB is the last line
tail -3 logs/<RUN_ID>.txt

# Monitor live
tail -f logs/<RUN_ID>.txt
```

### SSH from Local Machine
```bash
# Direct TCP (faster, use the IP/port shown in pod Connect tab)
ssh -T root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 "<command>"

# Copy files to pod
scp -P <PORT> -i ~/.ssh/id_ed25519 local_file.py root@<IP>:/workspace/parameter-golf/
```

## Alternative Compute Providers
If RunPod H100s are unavailable:
- **Lambda Labs**: https://lambdalabs.com — 8xH100 clusters
- **Vast.ai**: https://vast.ai — marketplace model, check availability
- **CoreWeave**: https://coreweave.com — enterprise, fast onboarding
- **OpenAI Compute Grant**: Apply at https://openai.com/index/parameter-golf/#credit-form

## Cost Estimates
| Config | $/hr | 10-min run | 1 hour iteration |
|--------|------|-----------|------------------|
| 1xH100 | $2.69 | $0.45 | $2.69 |
| 8xH100 | $21.52 | $3.59 | $21.52 |
