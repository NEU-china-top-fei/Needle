"""Profile only warmed-up optimized Transformer forwards with Nsight Systems."""
import ctypes
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "build/release/python"))
import numpy as np
import needle as ndl

np.random.seed(0)
device = ndl.cuda()
model = ndl.nn.TransformerLayer(256, 8, 32, 1024, dropout=0, device=device,
                                use_flash_attn=True, use_layernorm=True)
model.eval()
x = ndl.Tensor(np.random.randn(8, 64, 256).astype(np.float32), device=device, requires_grad=False)
runtime = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
with ndl.gemm_mode("tensorcore"):
    for _ in range(5):
        model(x)
    device.synchronize()
    if runtime.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed")
    for _ in range(10):
        model(x)
        device.synchronize()
    if runtime.cudaProfilerStop() != 0:
        raise RuntimeError("cudaProfilerStop failed")
