#!/bin/bash
# Pod setup script for parameter-golf competition
# Run on a fresh RunPod 8xH100 pod with Python 3.12 + PyTorch 2.9.1+cu128
# Usage: bash pod_requirements.sh

set -e

echo "=== Installing base packages ==="
pip install --break-system-packages zstandard sentencepiece

echo "=== Installing Flash Attention 2 ==="
pip install --break-system-packages flash-attn --no-build-isolation

echo "=== Installing liger-kernel (fused RMSNorm, CrossEntropy, RoPE, SwiGLU) ==="
pip install --break-system-packages liger-kernel

echo "=== Verifying ==="
python3 -c "
import torch; print(f'torch {torch.__version__}')
import zstandard; print('zstandard OK')
import sentencepiece; print('sentencepiece OK')
from flash_attn.flash_attn_interface import flash_attn_func; print('flash-attn OK')
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; print('liger-kernel OK')
except: print('liger-kernel: MISSING')
print('GPU:', torch.cuda.get_device_name(0))
print('cuDNN:', torch.backends.cudnn.version())
"

echo "=== Optional: FA3 Hopper (takes 15-30 min, only if needed) ==="
echo "To build FA3 Hopper kernels (only hdim64 for our model):"
echo "  cd /tmp && git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git"
echo "  cd flash-attention/hopper"
echo "  FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM128=TRUE \\"
echo "  FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE \\"
echo "  FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \\"
echo "  NVCC_THREADS=4 MAX_JOBS=40 pip install --break-system-packages -e . --no-build-isolation"
echo ""
echo "NOTE: FA3 build compiles 134 CUDA files serially. Each takes ~90s."
echo "Total: ~3 hours. Only worth it if you can run overnight."
echo "FA2 is already 0.73ms/call — FA3 saves ~8ms/step max."

echo "=== Done ==="
