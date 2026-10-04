"""Shared timing/report helpers. No GPU timing through an unrelated framework."""
import json
import platform
import subprocess
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def timed(fn, sync, warmup, iterations):
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(samples)),
            "p90_ms": float(np.percentile(samples, 90)), "samples_ms": samples}


def errors(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    delta = actual - expected
    return {"max_abs": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-12))}


def metadata(device, args):
    def git(*command):
        try:
            return subprocess.check_output(["git", *command], text=True,
                                           cwd=Path(__file__).resolve().parents[1]).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    import needle
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("src", "python", "benchmarks"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".cc"):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return {"python": platform.python_version(), "numpy": np.__version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": digest.hexdigest(),
            "needle_import": str(Path(needle.__file__).resolve().relative_to(root))
                if Path(needle.__file__).resolve().is_relative_to(root) else str(needle.__file__),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "device": device.gemm_device_info() if device.name == "cuda" else str(device),
            "arguments": vars(args)}


def save_report(path, report):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Report: {target}")
