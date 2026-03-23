#!/bin/bash
# Pod setup script for parameter-golf competition
# Run on a fresh RunPod 8xH100 pod with Python 3.12 + PyTorch 2.9.1+cu128

set -e

echo "=== Installing base packages ==="
pip install --break-system-packages zstandard sentencepiece

echo "=== Installing Flash Attention 2 (fallback) ==="
pip install --break-system-packages flash-attn --no-build-isolation

echo "=== Building Flash Attention 3 (Hopper-specific, ~15 min) ==="
cd /tmp
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
pip install --break-system-packages -e . --no-build-isolation
# Verify FA3
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 Hopper: OK')"

echo "=== Installing liger-kernel (fused CUDA ops) ==="
pip install --break-system-packages liger-kernel

echo "=== Verifying ==="
python3 -c "
import torch; print(f'torch {torch.__version__}')
import zstandard; print('zstandard OK')
import sentencepiece; print('sentencepiece OK')
from flash_attn.flash_attn_interface import flash_attn_func; print('FA2 OK')
try:
    from flash_attn_interface import flash_attn_func; print('FA3 Hopper OK')
except: print('FA3 Hopper: MISSING')
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; print('liger-kernel OK')
except: print('liger-kernel: MISSING')
"

echo "=== Done ==="
