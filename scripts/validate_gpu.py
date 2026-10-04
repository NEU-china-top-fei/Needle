"""Run GPU acceptance checks and save logs; fails if GPU tests cannot execute."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build/release")
    parser.add_argument("--output", default="results/validation")
    parser.add_argument("--skip-sanitizer", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root / args.build / "python"))
    checks = [("pytest", [sys.executable, "-m", "pytest", "-q", "tests", "--require-cuda"])]
    if not args.skip_sanitizer:
        sanitizer = shutil.which("compute-sanitizer") or "/usr/local/cuda/bin/compute-sanitizer"
        for tool in ("memcheck", "racecheck", "synccheck"):
            checks.append((tool, [sanitizer, "--tool", tool, "--error-exitcode", "1",
                sys.executable, "-m", "pytest", "-q", "tests", "--require-cuda", "-k", "cuda"]))
    for name, command in checks:
        print(f"Running {name}...", flush=True)
        with (output / f"{name}.txt").open("w") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        print((output / f"{name}.txt").read_text()[-3500:], flush=True)
        if result.returncode:
            raise SystemExit(f"{name} failed ({result.returncode}); see {output / (name + '.txt')}")


if __name__ == "__main__":
    main()
