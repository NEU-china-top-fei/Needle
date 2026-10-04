"""Same-weight Needle TransformerLayer forward ablation (no training claims)."""
import argparse

import numpy as np
import needle as ndl

from _utils import errors, metadata, save_report, timed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/transformer_ablation.json")
    args = p.parse_args()
    if min(args.batch, args.seq, args.dim, args.heads, args.iterations) <= 0 or args.warmup < 0:
        p.error("Dimensions/iterations must be positive and warmup nonnegative")
    if args.dim % args.heads:
        p.error("dim must be divisible by heads")
    if not hasattr(ndl.nn, "TransformerLayer"):
        p.error("Rebuild the project to include Transformer modules")
    device = ndl.cuda() if args.device == "cuda" else ndl.cpu()
    sync = device.synchronize if args.device == "cuda" else lambda: None
    info = metadata(device, args)
    np.random.seed(args.seed)
    model = ndl.nn.TransformerLayer(args.dim, args.heads, args.dim // args.heads,
                                    4 * args.dim, dropout=0, device=device,
                                    use_flash_attn=False, use_layernorm=False)
    model.eval()
    x = ndl.Tensor(np.random.randn(args.batch, args.seq, args.dim).astype(np.float32),
                   device=device, requires_grad=False)
    # One model, one set of weights, one input. Only implementation flags change.
    stages = [("baseline", "baseline", False, False)]
    if args.device == "cuda":
        stages += [("simt", "simt", False, False),
                   ("wmma", "wmma", False, False),
                   ("tensorcore", "tensorcore", False, False),
                   ("tensorcore_attention", "tensorcore", True, False),
                   ("tensorcore_attention_layernorm", "tensorcore", True, True)]
    results, baseline = [], None
    for name, mode, attention, norm in stages:
        for module in model._children():
            if isinstance(module, ndl.nn.MultiHeadAttention):
                module.use_flash_attn = attention
            if isinstance(module, ndl.nn.LayerNorm1d):
                module.use_layernorm = norm
        with ndl.gemm_mode(mode):
            actual = model(x).numpy()
            if baseline is None:
                baseline = actual.copy()
            error = errors(actual, baseline)
            if not np.isfinite(actual).all() or error["relative_l2"] > 0.02:
                raise AssertionError(f"{name} exceeds the reported 2% relative L2 budget: {error}")
            timing = timed(lambda: model(x), sync, args.warmup, args.iterations)
        results.append({"stage": name, "gemm_mode": mode, "fused_attention": attention,
                        "fused_layernorm": norm, "error_vs_baseline": error,
                        "correctness_passed": True, "framework": timing,
                        "speedup_vs_baseline": results[0]["framework"]["median_ms"] / timing["median_ms"]
                        if results else 1.0})
        print(f"{name:34s} {timing['median_ms']:.4f} ms; rel L2={error['relative_l2']:.3g}")
    save_report(args.output, {"metadata": info,
        "contract": "same weights/input, eval, dropout=0; synchronized forward wall time including framework graph construction",
        "error_budget_relative_l2": 0.02, "results": results})


if __name__ == "__main__":
    main()
