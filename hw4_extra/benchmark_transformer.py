#!/usr/bin/env python3
"""
Benchmark: needle Transformer vs PyTorch Transformer (cProfile 方式)

对比项:
  1. MultiHeadAttention   — needle orig vs needle flash_attn vs torch
  2. TransformerLayer     — needle orig vs needle flash_attn vs torch
  3. Transformer (full)   — needle orig vs needle flash_attn vs torch
  4. Scaling             — 不同 seq_len 的耗时曲线
  5. cProfile 深度分析    — 函数级调用统计

用法:
    python benchmark_transformer.py [--cpu-only] [--profile]
"""

import sys
sys.path.insert(0, './python')
sys.path.insert(0, './apps')

import argparse
import cProfile
import pstats
import io
import time
import warnings
import numpy as np
import torch
import torch.nn as tnn
import needle as ndl
import needle.nn as nn

warnings.filterwarnings("ignore", message="enable_nested_tensor")

# ── helpers ───────────────────────────────────────────────────────────

def fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.4f} s"


def fmt_summary(name, needle_orig, needle_flash, torch_t):
    """3-way comparison: needle_orig | needle_flash | torch | speedup"""
    orig_vs_torch = needle_orig / torch_t if torch_t > 0 else float('inf')
    flash_vs_torch = needle_flash / torch_t if torch_t > 0 else float('inf')
    speedup = needle_orig / needle_flash if needle_flash > 0 else float('inf')
    return (f"  {name:<22s}  orig={fmt_time(needle_orig):>10s}  "
            f"flash={fmt_time(needle_flash):>10s}  "
            f"torch={fmt_time(torch_t):>10s}  "
            f"flash_vs_orig={speedup:.1f}x  "
            f"orig_vs_torch={orig_vs_torch:.1f}x  "
            f"flash_vs_torch={flash_vs_torch:.1f}x")


def clear_gpu_memory():
    torch.cuda.empty_cache()


def sync_device(device_str):
    if device_str == "cuda":
        torch.cuda.synchronize()


def warmup(fn, *args, n=3, device_str="cpu", **kwargs):
    for _ in range(n):
        fn(*args, **kwargs)
    sync_device(device_str)


def time_forward(fn, *args, n_iters=50, device_str="cpu", **kwargs):
    warmup(fn, *args, n=5, device_str=device_str, **kwargs)
    sync_device(device_str)
    start = time.perf_counter()
    for _ in range(n_iters):
        fn(*args, **kwargs)
    sync_device(device_str)
    elapsed = (time.perf_counter() - start) / n_iters
    return elapsed


def run_cprofile(func, *args, n_warmup=10, **kwargs):
    for _ in range(n_warmup):
        func(*args, **kwargs)
    pr = cProfile.Profile()
    pr.enable()
    func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    return s.getvalue()


# ── Configs ───────────────────────────────────────────────────────────
# 为保证 FLOPs 可比: d_model = num_head * dim_head

CONFIGS = [
    ("tiny",    2,  16,  64,  4,  16,   256,  2),
    ("small",   8,  64,  256, 8,  32,   1024, 3),
    ("medium",  16, 128, 512, 8,  64,   2048, 4),
    ("large",   4,  128, 512, 8,  64,   2048, 4),
]

CPU_CONFIGS = [
    ("tiny",    2,  16,  64,  4,  16,   256,  2),
    ("small",   4,  32,  256, 8,  32,   1024, 2),
]


def get_configs(device):
    return CPU_CONFIGS if device == "cpu" else CONFIGS


# ── helper to run both variants of a needle module ────────────────────

def bench_needle_both(run_fn_orig, run_fn_flash, device_str, n_iters):
    """Time both original and flash_attn variants. Returns (t_orig, t_flash)."""
    t_orig = time_forward(run_fn_orig, n_iters=n_iters, device_str=device_str)
    clear_gpu_memory()
    t_flash = time_forward(run_fn_flash, n_iters=n_iters, device_str=device_str)
    clear_gpu_memory()
    return t_orig, t_flash


# ── 1. MultiHeadAttention ─────────────────────────────────────────────

def bench_attention(args):
    device_ndl = ndl.cpu() if args.cpu_only else ndl.cuda()
    device_str = "cpu" if args.cpu_only else "cuda"

    print("=" * 110)
    print(f"1. MultiHeadAttention  forward  ({device_str.upper()})")
    print("=" * 110)

    for name, bs, seq_len, _d_model, num_head, dim_head, _, _ in get_configs(device_str):
        try:
            ndl_qkv = np.random.randn(bs, num_head, seq_len, dim_head).astype(np.float32)

            # --- needle original ---
            ndl_attn_orig = nn.MultiHeadAttention(
                dropout=0., causal=True, device=device_ndl, use_flash_attn=False)
            ndl_attn_orig.eval()
            ndl_q_orig = ndl.Tensor(ndl_qkv.copy(), device=device_ndl)

            # --- needle flash ---
            ndl_attn_flash = nn.MultiHeadAttention(
                dropout=0., causal=True, device=device_ndl, use_flash_attn=True)
            ndl_attn_flash.eval()
            ndl_q_flash = ndl.Tensor(ndl_qkv.copy(), device=device_ndl)

            t_orig, t_flash = bench_needle_both(
                lambda: ndl_attn_orig(ndl_q_orig, ndl_q_orig, ndl_q_orig),
                lambda: ndl_attn_flash(ndl_q_flash, ndl_q_flash, ndl_q_flash),
                device_str, n_iters=30)

            # --- torch ---
            embed_dim = num_head * dim_head
            torch_qkv = torch.from_numpy(
                ndl_qkv.transpose(0, 2, 1, 3).reshape(bs, seq_len, embed_dim)
            ).to(device_str)

            torch_attn = tnn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_head, dropout=0.0,
                bias=False, batch_first=True, device=device_str)
            torch_attn.eval()
            attn_mask = tnn.Transformer.generate_square_subsequent_mask(
                seq_len, device=torch.device(device_str))

            def run_torch():
                return torch_attn(torch_qkv, torch_qkv, torch_qkv,
                                  attn_mask=attn_mask, need_weights=False)

            t_torch = time_forward(run_torch, n_iters=30, device_str=device_str)

            print(fmt_summary(f"MHA {name:<10s}", t_orig, t_flash, t_torch))
        except RuntimeError as e:
            print(f"  MHA {name:<10s}  SKIPPED (OOM: {str(e)[:80]}...)")
        finally:
            clear_gpu_memory()

    # Verify correctness: flash vs orig
    print("\n  [correctness] Checking flash_attn output vs original (tiny config):")
    bs, nh, sq, dh = 2, 4, 16, 16
    qkv = np.random.randn(bs, nh, sq, dh).astype(np.float32)
    attn_orig = nn.MultiHeadAttention(dropout=0., causal=True, device=device_ndl, use_flash_attn=False)
    attn_flash = nn.MultiHeadAttention(dropout=0., causal=True, device=device_ndl, use_flash_attn=True)
    attn_orig.eval(); attn_flash.eval()
    r_orig, _ = attn_orig(ndl.Tensor(qkv.copy(), device=device_ndl), ndl.Tensor(qkv.copy(), device=device_ndl), ndl.Tensor(qkv.copy(), device=device_ndl))
    r_flash, _ = attn_flash(ndl.Tensor(qkv.copy(), device=device_ndl), ndl.Tensor(qkv.copy(), device=device_ndl), ndl.Tensor(qkv.copy(), device=device_ndl))
    diff = np.abs(r_orig.numpy() - r_flash.numpy()).max()
    print(f"    max absolute difference: {diff:.6e}" + ("  ✓ PASS" if diff < 1e-4 else "  ✗ FAIL"))


# ── 2. TransformerLayer ───────────────────────────────────────────────

def bench_transformer_layer(args):
    device_ndl = ndl.cpu() if args.cpu_only else ndl.cuda()
    device_str = "cpu" if args.cpu_only else "cuda"

    print("\n" + "=" * 110)
    print(f"2. TransformerLayer  forward  ({device_str.upper()})")
    print("=" * 110)

    for name, bs, seq_len, d_model, num_head, dim_head, hidden_size, _ in get_configs(device_str):
        try:
            x = np.random.randn(bs, seq_len, d_model).astype(np.float32)

            ndl_layer_orig = nn.TransformerLayer(
                d_model, num_head, dim_head, hidden_size,
                dropout=0., causal=True, device=device_ndl, use_flash_attn=False)
            ndl_layer_orig.eval()
            ndl_x_orig = ndl.Tensor(x.copy(), device=device_ndl)

            ndl_layer_flash = nn.TransformerLayer(
                d_model, num_head, dim_head, hidden_size,
                dropout=0., causal=True, device=device_ndl, use_flash_attn=True)
            ndl_layer_flash.eval()
            ndl_x_flash = ndl.Tensor(x.copy(), device=device_ndl)

            t_orig, t_flash = bench_needle_both(
                lambda: ndl_layer_orig(ndl_x_orig),
                lambda: ndl_layer_flash(ndl_x_flash),
                device_str, n_iters=20)

            torch_layer = tnn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_head, dim_feedforward=hidden_size,
                dropout=0.0, activation='relu', batch_first=True,
                bias=False, device=device_str)
            torch_layer.eval()
            torch_x = torch.from_numpy(x).to(device_str)
            src_mask = tnn.Transformer.generate_square_subsequent_mask(
                seq_len, device=torch.device(device_str))
            t_torch = time_forward(lambda: torch_layer(torch_x, src_mask=src_mask),
                                   n_iters=20, device_str=device_str)

            print(fmt_summary(f"TL {name:<10s}", t_orig, t_flash, t_torch))
        except RuntimeError as e:
            print(f"  TransformerLayer {name:<10s}  SKIPPED (OOM: {str(e)[:80]}...)")
        finally:
            clear_gpu_memory()


# ── 3. Full Transformer ───────────────────────────────────────────────

def bench_transformer(args):
    device_ndl = ndl.cpu() if args.cpu_only else ndl.cuda()
    device_str = "cpu" if args.cpu_only else "cuda"

    print("\n" + "=" * 110)
    print(f"3. Transformer (full model)  forward  ({device_str.upper()})")
    print("=" * 110)

    for name, bs, seq_len, d_model, num_head, dim_head, hidden_size, num_layers in get_configs(device_str):
        try:
            x = np.random.randn(bs, seq_len, d_model).astype(np.float32)

            ndl_model_orig = nn.Transformer(
                d_model, hidden_size, num_layers,
                num_head=num_head, dim_head=dim_head,
                dropout=0., causal=True,
                device=device_ndl, batch_first=True, sequence_len=seq_len,
                use_flash_attn=False)
            ndl_model_orig.eval()
            ndl_x_orig = ndl.Tensor(x.copy(), device=device_ndl)

            ndl_model_flash = nn.Transformer(
                d_model, hidden_size, num_layers,
                num_head=num_head, dim_head=dim_head,
                dropout=0., causal=True,
                device=device_ndl, batch_first=True, sequence_len=seq_len,
                use_flash_attn=True)
            ndl_model_flash.eval()
            ndl_x_flash = ndl.Tensor(x.copy(), device=device_ndl)

            t_orig, t_flash = bench_needle_both(
                lambda: ndl_model_orig(ndl_x_orig),
                lambda: ndl_model_flash(ndl_x_flash),
                device_str, n_iters=10)

            encoder_layer = tnn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_head, dim_feedforward=hidden_size,
                dropout=0.0, activation='relu', batch_first=True,
                bias=False, device=device_str)
            torch_model = tnn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            torch_model.eval()
            torch_x = torch.from_numpy(x).to(device_str)
            src_mask = tnn.Transformer.generate_square_subsequent_mask(
                seq_len, device=torch.device(device_str))
            t_torch = time_forward(lambda: torch_model(torch_x, mask=src_mask),
                                   n_iters=10, device_str=device_str)

            print(fmt_summary(f"TF {name:<10s}", t_orig, t_flash, t_torch))
        except RuntimeError as e:
            print(f"  Transformer {name:<10s}  SKIPPED (OOM: {str(e)[:80]}...)")
        finally:
            clear_gpu_memory()


# ── 4. Scaling with sequence length ───────────────────────────────────

def bench_by_seq_len(args):
    device_ndl = ndl.cpu() if args.cpu_only else ndl.cuda()
    device_str = "cpu" if args.cpu_only else "cuda"

    print("\n" + "=" * 110)
    print(f"4. TransformerLayer scaling over seq_len  ({device_str.upper()})")
    print("=" * 110)

    bs, d_model, num_head, dim_head, hidden_size = 4, 256, 8, 32, 1024
    seq_lens = [16, 32, 64, 128, 256, 512] if device_str == "cuda" else [16, 32, 64, 128]

    for seq_len in seq_lens:
        try:
            x = np.random.randn(bs, seq_len, d_model).astype(np.float32)

            ndl_layer_orig = nn.TransformerLayer(
                d_model, num_head, dim_head, hidden_size,
                dropout=0., causal=True, device=device_ndl, use_flash_attn=False)
            ndl_layer_orig.eval()
            ndl_x_orig = ndl.Tensor(x.copy(), device=device_ndl)

            ndl_layer_flash = nn.TransformerLayer(
                d_model, num_head, dim_head, hidden_size,
                dropout=0., causal=True, device=device_ndl, use_flash_attn=True)
            ndl_layer_flash.eval()
            ndl_x_flash = ndl.Tensor(x.copy(), device=device_ndl)

            t_orig, t_flash = bench_needle_both(
                lambda: ndl_layer_orig(ndl_x_orig),
                lambda: ndl_layer_flash(ndl_x_flash),
                device_str, n_iters=15)

            torch_layer = tnn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_head, dim_feedforward=hidden_size,
                dropout=0.0, activation='relu', batch_first=True,
                bias=False, device=device_str)
            torch_layer.eval()
            torch_x = torch.from_numpy(x).to(device_str)
            src_mask = tnn.Transformer.generate_square_subsequent_mask(
                seq_len, device=torch.device(device_str))
            t_torch = time_forward(lambda: torch_layer(torch_x, src_mask=src_mask),
                                   n_iters=15, device_str=device_str)

            print(fmt_summary(f"seq_len={seq_len:<5d}", t_orig, t_flash, t_torch))
        except RuntimeError as e:
            print(f"  seq_len={seq_len:<5d}  SKIPPED (OOM: {str(e)[:80]}...)")
        finally:
            clear_gpu_memory()


# ── 5. cProfile depth analysis ────────────────────────────────────────

def bench_profile(args):
    device_ndl = ndl.cpu() if args.cpu_only else ndl.cuda()
    device_str = "cpu" if args.cpu_only else "cuda"

    print("\n" + "=" * 90)
    print("5. cProfile 深度分析")
    print("=" * 90)

    bs, seq_len, d_model, num_head, dim_head, hidden_size = 4, 32, 256, 8, 32, 1024
    x = np.random.randn(bs, seq_len, d_model).astype(np.float32)

    # needle TransformerLayer — flash_attn
    ndl_flash = nn.TransformerLayer(
        d_model, num_head, dim_head, hidden_size,
        dropout=0., causal=True, device=device_ndl, use_flash_attn=True)
    ndl_flash.eval()
    ndl_x = ndl.Tensor(x.copy(), device=device_ndl)

    print("\n── needle TransformerLayer (use_flash_attn=True) cProfile ──")
    profile_out = run_cprofile(lambda: ndl_flash(ndl_x), n_warmup=10)
    print(profile_out)

    # needle TransformerLayer — original
    ndl_orig = nn.TransformerLayer(
        d_model, num_head, dim_head, hidden_size,
        dropout=0., causal=True, device=device_ndl, use_flash_attn=False)
    ndl_orig.eval()
    ndl_x2 = ndl.Tensor(x.copy(), device=device_ndl)

    print("── needle TransformerLayer (use_flash_attn=False) cProfile ──")
    profile_out = run_cprofile(lambda: ndl_orig(ndl_x2), n_warmup=10)
    print(profile_out)


# ── main ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark needle Transformer vs PyTorch Transformer")
    parser.add_argument("--cpu-only", action="store_true",
                        help="仅在 CPU 上运行")
    parser.add_argument("--profile", action="store_true",
                        help="额外运行 cProfile 深度分析")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 110)
    print("needle Transformer vs PyTorch Transformer — Performance Benchmark")
    print("  columns: orig (use_flash_attn=False) | flash (use_flash_attn=True) | torch")
    print("=" * 110)

    bench_attention(args)
    bench_transformer_layer(args)
    bench_transformer(args)
    bench_by_seq_len(args)

    if args.profile:
        bench_profile(args)

    print("\n" + "=" * 110)
    print("Done.")
    print("=" * 110)


if __name__ == "__main__":
    main()
