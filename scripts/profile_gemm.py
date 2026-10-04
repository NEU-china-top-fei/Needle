"""Single prepared GEMM launch for Nsight Compute; timing comes from the profiler."""
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", default="build/release")
parser.add_argument("--size", type=int, default=2048)
parser.add_argument("--mode", choices=["wmma", "tensorcore"], default="tensorcore")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / args.build / "python"))
import numpy as np
import needle as ndl
from needle.backend_ndarray import NDArray

rng = np.random.default_rng(0)
dev = ndl.cuda()
a = NDArray(rng.normal(size=(args.size, args.size)).astype(np.float32), device=dev)
b = NDArray(rng.normal(size=(args.size, args.size)).astype(np.float32), device=dev)
out = NDArray.make((args.size, args.size), device=dev)
packed = dev.PreparedGemm(a._handle, b._handle, args.size, args.size, args.size)
dev.synchronize()
packed.run(out._handle, args.mode == "tensorcore")
dev.synchronize()
