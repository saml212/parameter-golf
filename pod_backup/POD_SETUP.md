# Pod Reconstruction Guide

## RunPod Configuration
- **Pod name**: son of slammy
- **GPU**: 8x H100 SXM 80GB
- **Template**: Official Parameter Golf template (use "Launch Template" from the repo README)
- **Disk**: 50GB (default)
- **SSH**: root@<NEW_IP> -p <NEW_PORT> -i ~/.ssh/id_ed25519

## After Pod Starts

### 1. Clone repo and download data
```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### 2. Install extra packages
```bash
pip install zstandard --break-system-packages
pip install flash-attn --no-build-isolation --break-system-packages
```

### 3. Upload training scripts
From local machine:
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 train_gpt_v11_novel.py root@<IP>:/workspace/parameter-golf/
scp -P <PORT> -i ~/.ssh/id_ed25519 train_gpt_v8_xsa_ema.py root@<IP>:/workspace/parameter-golf/
```

### 4. Warm torch.compile cache
Run ANY training script once (it'll be slow ~100ms/step). Second run stabilizes at ~70ms/step.
```bash
cd /workspace/parameter-golf
SEED=1337 NUM_LAYERS=11 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
torchrun --standalone --nproc_per_node=8 train_gpt_v11_novel.py
```

## Environment Versions (verified working)
- Python 3.12.3
- PyTorch 2.9.1+cu128
- flash_attn 2.8.3
- numpy 2.4.3
- sentencepiece 0.2.1
- zstandard 0.25.0

## Data Paths
- SP1024 data: /workspace/parameter-golf/data/datasets/fineweb10B_sp1024/
- Tokenizer: /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model

## Best Config (12L Gradient-Guided Quant)
```bash
SEED=1338 NUM_LAYERS=12 MLP_HIDDEN=1408 BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MUON_WD=0.04 ADAM_WD=0.04 \
SWA_ENABLED=0 EMA_ENABLED=1 EMA_DECAY=0.997 \
XSA_LAST_N=4 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 GRAD_QUANT=1 \
torchrun --standalone --nproc_per_node=8 train_gpt_v11_novel.py
```
Result: 1.1320 BPB (3-seed mean)
