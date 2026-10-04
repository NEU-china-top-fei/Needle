"""Reproduce the published GEMM and Transformer forward benchmark suite."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build/release")
    parser.add_argument("--output", default="results/rtx4060")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root / args.build / "python"))
    for run in range(args.repeats):
        subprocess.run([sys.executable, "benchmarks/gemm.py", "--shapes",
            "128x128x128", "512x512x512", "1024x1024x1024", "2048x2048x2048",
            "512x256x1024", "127x255x129", "--warmup", "10", "--iterations", "50",
            "--output", str(output / f"gemm_{run}.json")], cwd=root, env=env, check=True)
        for name, batch, seq, dim, heads in [("small", 2, 32, 128, 4), ("medium", 8, 64, 256, 8)]:
            subprocess.run([sys.executable, "benchmarks/transformer_ablation.py",
                "--batch", str(batch), "--seq", str(seq), "--dim", str(dim), "--heads", str(heads),
                "--warmup", "5", "--iterations", "30", "--output",
                str(output / f"transformer_{name}_{run}.json")], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
