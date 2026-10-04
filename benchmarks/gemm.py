"""GEMM ablation: framework wall time and separate prepared Tensor Core time."""
import argparse

import numpy as np
import needle as ndl
from needle.backend_ndarray import NDArray

from _utils import errors, metadata, save_report, timed


def shape(value):
    try:
        result = tuple(map(int, value.lower().split("x")))
        if len(result) != 3 or min(result) <= 0:
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError("Expected positive MxKxN, e.g. 256x512x128")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--shapes", type=shape, nargs="+",
                   default=[(128, 128, 128), (512, 512, 512), (1024, 1024, 1024),
                            (512, 256, 1024), (127, 255, 129)])
    p.add_argument("--modes", nargs="+", choices=["baseline", "simt", "wmma", "tensorcore"],
                   default=["baseline", "simt", "wmma", "tensorcore"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/gemm.json")
    args = p.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        p.error("iterations must be positive and warmup nonnegative")
    if args.device == "cpu" and args.modes != ["baseline"]:
        p.error("CPU smoke runs require --modes baseline; CUDA modes would be CPU fallbacks")
    device = ndl.cuda() if args.device == "cuda" else ndl.cpu()
    if not device.enabled():
        p.error("Backend not built; follow README build instructions")
    sync = device.synchronize if args.device == "cuda" else lambda: None
    # This fails clearly on a missing driver instead of reporting skipped CUDA as success.
    info = metadata(device, args)
    rng = np.random.default_rng(args.seed)
    rows, library_rows = [], []
    for m, k, n in args.shapes:
        a = rng.normal(size=(m, k)).astype(np.float32)
        b = rng.normal(size=(k, n)).astype(np.float32)
        lhs, rhs = NDArray(a, device=device), NDArray(b, device=device)
        exact = a.astype(np.float64) @ b.astype(np.float64)
        rounded = a.astype(np.float16).astype(np.float64) @ b.astype(np.float16).astype(np.float64)
        packed = None
        for mode in args.modes:
            is_tc = mode in ("wmma", "tensorcore")
            with ndl.gemm_mode(mode):
                result = (lhs @ rhs).numpy()
                ref = rounded if is_tc else exact
                np.testing.assert_allclose(result, ref, rtol=1e-3, atol=2e-3)
                row = {"shape_mkn": [m, k, n], "mode": mode,
                       "execution": ("cpu_fp32" if args.device == "cpu" else
                                     "baseline_fallback" if mode == "simt" and
                                     (m % 128 or n % 128 or k % 8) else mode),
                       "input_storage": "fp32", "multiplication_inputs": "fp16" if is_tc else "fp32",
                       "accumulation_output": "fp32", "correctness_passed": True,
                       "error_vs_fp32_inputs": errors(result, exact),
                       "error_vs_compute_inputs": errors(result, ref),
                       "framework": timed(lambda: lhs @ rhs, sync, args.warmup, args.iterations)}
            row["framework"]["effective_tflops"] = 2 * m * k * n / (row["framework"]["median_ms"] * 1e9)
            if is_tc:
                if packed is None:
                    packed = device.PreparedGemm(lhs._handle, rhs._handle, m, k, n)
                # Check the prepared path too; it is the actual kernel benchmark path.
                output = NDArray.make((m, n), device=device)
                packed.run(output._handle, mode == "tensorcore")
                np.testing.assert_allclose(output.numpy(), rounded, rtol=1e-3, atol=2e-3)
                ms = packed.benchmark(args.iterations, args.warmup, mode == "tensorcore")
                pm, pk, pn = (m + 63) // 64 * 64, (k + 31) // 32 * 32, (n + 63) // 64 * 64
                row["prepared_kernel"] = {"mean_ms": ms, "padded_shape_mkn": [pm, pk, pn],
                    "effective_tflops": 2 * m * k * n / (ms * 1e9),
                    "executed_tflops": 2 * pm * pk * pn / (ms * 1e9)}
            rows.append(row)
            print(f"{m}x{k}x{n} {mode:10s} {row['framework']['median_ms']:.4f} ms (framework)")
        if packed is not None:
            packed.run_cublas(output._handle)
            library_result = output.numpy()
            np.testing.assert_allclose(library_result, rounded, rtol=1e-3, atol=2e-3)
            ms = packed.benchmark_cublas(args.iterations, args.warmup)
            library_rows.append({"shape_mkn": [m, k, n], "library": "cublasGemmEx",
                "input_dtype": "fp16", "compute_output_dtype": "fp32",
                "padded_shape_mkn": [pm, pk, pn], "mean_ms": ms,
                "effective_tflops": 2 * m * k * n / (ms * 1e9),
                "error_vs_compute_inputs": errors(library_result, rounded),
                "correctness_passed": True})
            print(f"{m}x{k}x{n} cuBLAS     {ms:.4f} ms (prepared kernel)")
    save_report(args.output, {"metadata": info, "timing_contract": {
        "framework": "synchronized wall time; dispatch, allocations, packing/padding, GEMM, crop, temporary frees",
        "prepared_kernel": "CUDA events; padded GEMM only; excludes packing, allocation, crop and Python",
        "precision": "TC modes use rounded FP16 inputs/FP32 accumulation; baseline/SIMT retain FP32 inputs"},
        "results": rows, "cublas_prepared_reference": library_rows})


if __name__ == "__main__":
    main()
