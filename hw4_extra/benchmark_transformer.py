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
from needle import ops
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


# ── 0. LayerNorm A/B Comparison ──────────────────────────────────────

def bench_layernorm(args):
    device_ndl = ndl.cpu() if args.cpu_only else ndl.cuda()
    device_str = "cpu" if args.cpu_only else "cuda"

    print("=" * 120)
    print(f"0. LayerNorm: Fused vs Original A/B Comparison  ({device_str.upper()})")
    print("=" * 120)

    configs = [
        ("tiny",  4, 256),
        ("small", 8, 1024),
        ("medium", 16, 1024),
        ("large", 32, 2048),
    ]

    for name, N, D in configs:
        try:
            x = np.random.randn(N, D).astype(np.float32)

            ln_fused = nn.LayerNorm1d(D, device=device_ndl, use_layernorm=True)
            ln_fused.eval()
            ln_orig = nn.LayerNorm1d(D, device=device_ndl, use_layernorm=False)
            ln_orig.eval()

            ndl_x_fused = ndl.Tensor(x.copy(), device=device_ndl)
            ndl_x_orig = ndl.Tensor(x.copy(), device=device_ndl)

            t_fused, t_orig = bench_needle_both(
                lambda: ln_fused(ndl_x_fused),
                lambda: ln_orig(ndl_x_orig),
                device_str, n_iters=100)

            torch_ln = tnn.LayerNorm(D, device=device_str)
            torch_ln.eval()
            torch_x = torch.from_numpy(x).to(device_str)
            t_torch = time_forward(lambda: torch_ln(torch_x), n_iters=100, device_str=device_str)

            speedup = t_orig / t_fused if t_fused > 0 else float('inf')
            print(f"  LN {name:<10s} (N={N},D={D})  "
                  f"orig={fmt_time(t_orig):>10s}  fused={fmt_time(t_fused):>10s}  "
                  f"torch={fmt_time(t_torch):>10s}  "
                  f"fused_vs_orig={speedup:.1f}x  orig_vs_torch={t_orig/t_torch:.1f}x  "
                  f"fused_vs_torch={t_fused/t_torch:.1f}x")
        except RuntimeError as e:
            print(f"  LN {name:<10s}  SKIPPED ({str(e)[:60]}...)")
        finally:
            clear_gpu_memory()

    # Correctness check
    print("\n  [correctness] fused vs original vs torch:")
    N, D = 4, 256
    x = np.random.randn(N, D).astype(np.float32)
    ln_fused = nn.LayerNorm1d(D, device=device_ndl, use_layernorm=True); ln_fused.eval()
    ln_orig = nn.LayerNorm1d(D, device=device_ndl, use_layernorm=False); ln_orig.eval()
    out_fused = ln_fused(ndl.Tensor(x.copy(), device=device_ndl)).numpy()
    out_orig = ln_orig(ndl.Tensor(x.copy(), device=device_ndl)).numpy()
    torch_ln = tnn.LayerNorm(D, device=device_str); torch_ln.eval()
    out_torch = torch_ln(torch.from_numpy(x).to(device_str)).detach().cpu().numpy()

    d1 = np.abs(out_fused - out_orig).max()
    d2 = np.abs(out_fused - out_torch).max()
    print(f"    max diff (fused vs orig):  {d1:.6e}" + ("  ✓" if d1 < 1e-4 else "  ✗"))
    print(f"    max diff (fused vs torch): {d2:.6e}" + ("  ✓" if d2 < 1e-4 else "  ✗"))

    # ── TransformerLayer: LN fused vs orig impact ──
    print("\n  ── TransformerLayer: Fused-LN vs Original-LN ──")
    bs, sq, dm, nh, dh, hs = 8, 64, 256, 8, 32, 1024
    x = np.random.randn(bs, sq, dm).astype(np.float32)

    for name, use_ln in [("TL LN-fused", True), ("TL LN-orig ", False)]:
        try:
            ndl_tl = nn.TransformerLayer(
                dm, nh, dh, hs, dropout=0., causal=True,
                device=device_ndl, use_flash_attn=True, use_layernorm=use_ln)
            ndl_tl.eval()
            ndl_x = ndl.Tensor(x.copy(), device=device_ndl)
            t = time_forward(lambda: ndl_tl(ndl_x), n_iters=20, device_str=device_str)
            print(f"    {name:<20s}  {fmt_time(t):>12s}")
        except RuntimeError as e:
            print(f"    {name:<20s}  SKIPPED ({str(e)[:60]}...)")
        finally:
            clear_gpu_memory()

    # ── Op-level LayerNorm breakdown ──
    print("\n  ── LayerNorm Op-Level Breakdown (N=512,D=256) ──")
    N, D = 512, 256
    x = np.random.randn(N, D).astype(np.float32)

    # Fused
    ln_f = nn.LayerNorm1d(D, device=device_ndl, use_layernorm=True); ln_f.eval()
    ndl_x_f = ndl.Tensor(x.copy(), device=device_ndl)
    t_fused_op = time_forward(lambda: ln_f(ndl_x_f), n_iters=200, device_str=device_str)

    print(f"    fused layernorm  (1 op):   {fmt_time(t_fused_op):>12s}")

    # Original: profile each sub-op individually
    print(f"    original layernorm (19 ops total):")
    ndl_x_o = ndl.Tensor(x.copy(), device=device_ndl)

    fn_val = float(D)

    t_sum_mean = time_forward(
        lambda: ndl_x_o.sum(axes=(1,)), n_iters=200, device_str=device_str)
    mean = ndl_x_o.sum(axes=(1,))

    t_div_mean = time_forward(
        lambda: mean / fn_val, n_iters=200, device_str=device_str)
    mean_norm = mean / fn_val

    t_reshape1 = time_forward(
        lambda: ops.reshape(mean_norm, (N, 1)), n_iters=200, device_str=device_str)
    exp1 = ops.reshape(mean_norm, (N, 1))

    t_broadcast1 = time_forward(
        lambda: ops.broadcast_to(exp1, (N, D)), n_iters=200, device_str=device_str)
    exp1_bc = ops.broadcast_to(exp1, (N, D))

    t_sub = time_forward(
        lambda: ndl_x_o - exp1_bc, n_iters=200, device_str=device_str)
    delta = ndl_x_o - exp1_bc

    t_square = time_forward(
        lambda: delta * delta, n_iters=200, device_str=device_str)
    sq = delta * delta

    t_sum_var = time_forward(
        lambda: sq.sum(axes=(1,)), n_iters=200, device_str=device_str)
    var_sum = sq.sum(axes=(1,))

    t_div_var = time_forward(
        lambda: var_sum / fn_val, n_iters=200, device_str=device_str)
    var_norm = var_sum / fn_val

    t_reshape2 = time_forward(
        lambda: ops.reshape(var_norm, (N, 1)), n_iters=200, device_str=device_str)
    var_rs = ops.reshape(var_norm, (N, 1))

    t_add_eps = time_forward(
        lambda: ops.add_scalar(var_rs, 1e-5), n_iters=200, device_str=device_str)
    var_eps = ops.add_scalar(var_rs, 1e-5)

    t_sqrt = time_forward(
        lambda: ops.power_scalar(var_eps, 0.5), n_iters=200, device_str=device_str)
    std = ops.power_scalar(var_eps, 0.5)

    t_broadcast_std = time_forward(
        lambda: ops.broadcast_to(std, (N, D)), n_iters=200, device_str=device_str)
    std_bc = ops.broadcast_to(std, (N, D))

    # Total original time
    t_orig_total = (t_sum_mean + t_div_mean + t_reshape1 + t_broadcast1 +
                    t_sub + t_square + t_sum_var + t_div_var +
                    t_reshape2 + t_add_eps + t_sqrt + t_broadcast_std)
    print(f"      sum(mean):        {fmt_time(t_sum_mean):>12s}")
    print(f"      div(mean/D):      {fmt_time(t_div_mean):>12s}")
    print(f"      reshape(mean):    {fmt_time(t_reshape1):>12s}")
    print(f"      broadcast(mean):  {fmt_time(t_broadcast1):>12s}")
    print(f"      sub(x-mean):      {fmt_time(t_sub):>12s}")
    print(f"      square(delta^2):  {fmt_time(t_square):>12s}")
    print(f"      sum(var):         {fmt_time(t_sum_var):>12s}")
    print(f"      div(var/D):       {fmt_time(t_div_var):>12s}")
    print(f"      reshape(var):     {fmt_time(t_reshape2):>12s}")
    print(f"      add_eps:          {fmt_time(t_add_eps):>12s}")
    print(f"      sqrt:             {fmt_time(t_sqrt):>12s}")
    print(f"      broadcast(std):   {fmt_time(t_broadcast_std):>12s}")
    print(f"      {'─'*45}")
    print(f"      original sum(12):  {fmt_time(t_orig_total):>12s}")
    print(f"      fused (1 op):      {fmt_time(t_fused_op):>12s}")
    print(f"      speedup: {t_orig_total/t_fused_op:.1f}x" if t_fused_op > 0 else "      N/A")


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


# ── 5. Detailed per-component profiling ───────────────────────────────

def bench_detailed_profile(args):
    device_ndl = ndl.cpu() if args.cpu_only else ndl.cuda()
    device_str = "cpu" if args.cpu_only else "cuda"

    print("\n" + "=" * 110)
    print(f"5. Detailed Per-Component Profile — TransformerLayer ({device_str.upper()})")
    print("=" * 110)

    bs, seq_len, d_model, num_head, dim_head, hidden_size = 8, 64, 256, 8, 32, 1024
    x = np.random.randn(bs, seq_len, d_model).astype(np.float32)

    def profile_component(label, setup_fn, run_fn, n_iters=50):
        """Profile a single component: setup once, time the run_fn multiple times."""
        setup_fn()
        t = time_forward(run_fn, n_iters=n_iters, device_str=device_str)
        print(f"    {label:<42s}  {fmt_time(t):>12s}")
        return t

    # ── Sub-module profiling ──
    print("\n  ── Individual Ops ──")

    N = bs * seq_len
    ndl_x_2d = ndl.Tensor(x.reshape(N, d_model), device=device_ndl)

    # LayerNorm
    ln = nn.LayerNorm1d(d_model, device=device_ndl)
    ln.eval()
    profile_component("LayerNorm (fused)", lambda: None, lambda: ln(ndl_x_2d), n_iters=100)

    # Linear (matmul)
    linear = nn.Linear(d_model, num_head * dim_head, bias=False, device=device_ndl)
    linear.eval()
    profile_component("Linear (d->num_head*dim_head)",
                      lambda: None, lambda: linear(ndl_x_2d), n_iters=50)

    linear_ffn = nn.Linear(d_model, hidden_size, bias=False, device=device_ndl)
    linear_ffn.eval()
    profile_component("Linear FFN (d->hidden)",
                      lambda: None, lambda: linear_ffn(ndl_x_2d), n_iters=50)

    linear_out = nn.Linear(hidden_size, d_model, bias=False, device=device_ndl)
    linear_out.eval()
    x_hidden = np.random.randn(N, hidden_size).astype(np.float32)
    ndl_hidden_2d = ndl.Tensor(x_hidden, device=device_ndl)
    profile_component("Linear FFN (hidden->d)",
                      lambda: None, lambda: linear_out(ndl_hidden_2d), n_iters=50)

    # ReLU
    profile_component("ReLU", lambda: None,
                      lambda: ops.relu(ndl_hidden_2d), n_iters=100)

    # Dropout (eval mode = identity)
    drop = nn.Dropout(0.0)
    drop.eval()
    profile_component("Dropout (eval=identity)", lambda: None, lambda: drop(ndl_x_2d), n_iters=100)

    # Element-wise add (residual)
    profile_component("Element-wise Add (residual)",
                      lambda: None, lambda: ndl_x_2d + ndl_x_2d, n_iters=100)

    # Permute + reshape (common in attention layer)
    inner_dim = num_head * dim_head
    x_4d = ndl.Tensor(x.reshape(bs, seq_len, num_head, dim_head), device=device_ndl)
    profile_component("Permute (0,2,1,3) 4D tensor",
                      lambda: None, lambda: ops.permute(x_4d, (0, 2, 1, 3)), n_iters=100)
    profile_component("Reshape (N,d)->(B,N,num,dim)",
                      lambda: None, lambda: ops.reshape(ndl_x_2d,
                            (bs, seq_len, num_head, dim_head)), n_iters=100)

    # ── MHA sub-components ──
    print("\n  ── MultiHeadAttention Breakdown ──")
    ndl_qkv = ndl.Tensor(np.random.randn(bs, num_head, seq_len, dim_head).astype(np.float32),
                         device=device_ndl)

    # Custom matmul (original)
    attn = nn.MultiHeadAttention(dropout=0., causal=True, device=device_ndl, use_flash_attn=False)
    attn.eval()
    profile_component("MHA: custom matmul (5D broadcast)",
                      lambda: None,
                      lambda: attn.matmul(ndl_qkv, ndl_qkv), n_iters=20)

    # Softmax (original)
    score = attn.matmul(ndl_qkv, ndl_qkv) / np.sqrt(dim_head)
    profile_component("MHA: softmax (original, 6 ops)",
                      lambda: None,
                      lambda: attn.softmax(score), n_iters=30)

    # FlashAttention
    profile_component("MHA: FlashAttn (fused kernel)",
                      lambda: None,
                      lambda: ops.flash_attention(ndl_qkv, ndl_qkv, ndl_qkv,
                            causal=True, softmax_scale=1.0/np.sqrt(dim_head)), n_iters=50)

    # ── Full AttentionLayer breakdown ──
    print("\n  ── AttentionLayer Sub-Module Breakdown (flash mode) ──")
    attn_layer = nn.AttentionLayer(
        d_model, num_head, dim_head, dropout=0., causal=True,
        device=device_ndl, use_flash_attn=True)
    attn_layer.eval()
    ndl_x_tl = ndl.Tensor(x.copy(), device=device_ndl)

    # Profile internal projections individually
    N_total = bs * seq_len
    x_flat = ndl.Tensor(x.reshape(N_total, d_model), device=device_ndl)

    # prenorm_q
    q_normed = attn_layer.prenorm_q(x_flat)
    profile_component("  prenorm_q (LayerNorm)",
                      lambda: None, lambda: attn_layer.prenorm_q(x_flat), n_iters=100)

    # q_projection
    inner_dim_total = num_head * dim_head
    q_proj = attn_layer.q_projection
    profile_component("  q_projection (Linear)",
                      lambda: None,
                      lambda: q_proj(q_normed.reshape((N_total, d_model))),
                      n_iters=50)

    # MHA (flash)
    inner_dim_val = num_head * dim_head
    x_normed = attn_layer.prenorm_q(x_flat)
    q_proj_in = q_proj(x_normed.reshape((N_total, d_model)))
    q_4d = ops.permute(q_proj_in.reshape((bs, seq_len, num_head, dim_head)), (0, 2, 1, 3))
    k_4d = ndl.Tensor(ndl_qkv.numpy().copy(), device=device_ndl)
    v_4d = ndl.Tensor(ndl_qkv.numpy().copy(), device=device_ndl)
    profile_component("  flash_attention (MHA core)",
                      lambda: None,
                      lambda: attn_layer.attn(q_4d, k_4d, v_4d), n_iters=30)

    # out_projection
    N_out = bs * seq_len
    out_in = ndl.Tensor(np.random.randn(N_out, inner_dim_val).astype(np.float32), device=device_ndl)
    profile_component("  out_projection (Linear)",
                      lambda: None,
                      lambda: attn_layer.out_projection(out_in), n_iters=50)

    # ── Full TransformerLayer per-component breakdown ──
    print("\n  ── TransformerLayer Component Tree (flash mode, one forward) ──")

    tl_flash = nn.TransformerLayer(
        d_model, num_head, dim_head, hidden_size,
        dropout=0., causal=True, device=device_ndl, use_flash_attn=True)
    tl_flash.eval()
    ndl_x_tl = ndl.Tensor(x.copy(), device=device_ndl)

    # Profile each top-level component
    comps = []

    # multihead attention sublayer
    t_attn = profile_component("TL: self-attention sublayer (total)",
        lambda: None,
        lambda: tl_flash.multiattn(ndl_x_tl), n_iters=30)
    comps.append(("Attention sub-layer", t_attn))

    # We can't easily isolate the FFN from the residual, so let's measure them separately
    # by creating standalone modules

    # FFN: norm -> linear1 -> relu -> dropout -> linear2 -> dropout
    attn_out = tl_flash.multiattn(ndl_x_tl)
    residual1 = ndl_x_tl + tl_flash.drop(attn_out)  # temp

    t_norm = profile_component("TL: LayerNorm (after attn)",
        lambda: None,
        lambda: tl_flash.norm(residual1.reshape((N_total, d_model))), n_iters=100)
    comps.append(("LayerNorm (FFN)", t_norm))

    norm_out = tl_flash.norm(residual1.reshape((N_total, d_model)))
    t_linear1 = profile_component("TL: Linear1 (d->hidden)",
        lambda: None,
        lambda: tl_flash.linear1(norm_out), n_iters=50)
    comps.append(("Linear1 (d→hidden)", t_linear1))

    lin1_out = tl_flash.linear1(norm_out)
    t_relu = profile_component("TL: ReLU",
        lambda: None,
        lambda: tl_flash.nonlinear(lin1_out), n_iters=100)
    comps.append(("ReLU", t_relu))

    relu_out = tl_flash.nonlinear(lin1_out)
    t_drop1 = profile_component("TL: Dropout (after Linear1)",
        lambda: None,
        lambda: tl_flash.drop(relu_out), n_iters=100)
    comps.append(("Dropout1", t_drop1))

    drop1_out = tl_flash.drop(relu_out)
    t_linear2 = profile_component("TL: Linear2 (hidden->d)",
        lambda: None,
        lambda: tl_flash.linear2(drop1_out), n_iters=50)
    comps.append(("Linear2 (hidden→d)", t_linear2))

    lin2_out = tl_flash.linear2(drop1_out)
    t_drop2 = profile_component("TL: Dropout (after Linear2)",
        lambda: None,
        lambda: tl_flash.drop(lin2_out), n_iters=100)
    comps.append(("Dropout2", t_drop2))

    # ── Summary table ──
    print("\n  ── Component Time Breakdown ──")
    comp_total = sum(t for _, t in comps)
    print(f"  {'Component':<36s}  {'Time':>10s}  {'%':>6s}")
    print(f"  {'-'*54}")
    for label, t in sorted(comps, key=lambda x: -x[1]):
        pct = (t / comp_total * 100) if comp_total > 0 else 0
        print(f"  {label:<36s}  {fmt_time(t):>10s}  {pct:>5.1f}%")
    print(f"  {'-'*54}")
    print(f"  {'Sum of profiled components':<36s}  {fmt_time(comp_total):>10s}")

    # Full TL timing for comparison
    t_full = profile_component("\nTL: full forward (flash)",
        lambda: None,
        lambda: tl_flash(ndl_x_tl), n_iters=30)
    overhead = t_full - comp_total
    print(f"  {'Overhead (dispatch/alloc/reshape)':<36s}  {fmt_time(max(0, overhead)):>10s}")

    # ── cProfile for both paths ──
    print("\n\n  ══ cProfile: TransformerLayer (use_flash_attn=True) ══")
    profile_out = run_cprofile(lambda: tl_flash(ndl_x_tl), n_warmup=10)
    print(profile_out)

    tl_orig = nn.TransformerLayer(
        d_model, num_head, dim_head, hidden_size,
        dropout=0., causal=True, device=device_ndl, use_flash_attn=False)
    tl_orig.eval()

    print("  ══ cProfile: TransformerLayer (use_flash_attn=False) ══")
    profile_out = run_cprofile(lambda: tl_orig(ndl_x_tl), n_warmup=10)
    print(profile_out)

    # ── GPU memory summary ──
    if device_str == "cuda":
        print("\n  ── GPU Memory ──")
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            tl_flash(ndl_x_tl)
            torch.cuda.synchronize()
        mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024
        mem_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"    peak allocated: {mem_alloc:.1f} MB")
        print(f"    peak reserved:  {mem_reserved:.1f} MB")


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

    bench_layernorm(args)
    bench_attention(args)
    bench_transformer_layer(args)
    bench_transformer(args)
    bench_by_seq_len(args)

    if args.profile:
        bench_detailed_profile(args)

    print("\n" + "=" * 110)
    print("Done.")
    print("=" * 110)


if __name__ == "__main__":
    main()
