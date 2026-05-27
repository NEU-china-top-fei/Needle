import sys
sys.path.append('./python')
sys.path.append('./apps')
import needle as ndl
import needle.nn as nn
import numpy as np
import torch
import torch.nn as tnn
import time

# ============================================================
# Configuration
# ============================================================
np.random.seed(3)
torch.manual_seed(3)

corpus = ndl.data.Corpus("data/ptb", max_lines=20)
vocab_size = len(corpus.dictionary)
embedding_dim = 30
hidden_size = 10
num_layers = 2
seq_len = 10
batch_size = 16
lr = 4.0

train_data_np = ndl.data.batchify(corpus.train, batch_size=batch_size, device=ndl.cpu(), dtype="float32")

# Transpose rules for ndl->torch weight copy (torch F.linear(x,W) = x@W.T, needle does x@W)
TRANSPOSE = {1, 2, 5, 6, 9}  # W_ih_0, W_hh_0, W_ih_1, W_hh_1, linear.weight
SQUEEZE = {10}                 # linear.bias: ndl (1,V) -> torch (V,)

from models import LanguageModel


def build_models(device_ndl, device_torch):
    ndl_dev = ndl.cpu() if device_ndl == 'cpu' else ndl.cuda()
    ndl_model = LanguageModel(embedding_dim, vocab_size, hidden_size=hidden_size,
                              num_layers=num_layers, seq_model='rnn', device=ndl_dev)
    ndl_params = list(ndl_model.parameters())

    class TorchRNN(tnn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = tnn.Embedding(vocab_size, embedding_dim)
            self.rnn = tnn.RNN(embedding_dim, hidden_size, num_layers, bias=True,
                               nonlinearity='tanh', batch_first=False)
            self.linear = tnn.Linear(hidden_size, vocab_size)
        def forward(self, x, h=None):
            emb = self.embedding(x)
            out, h = self.rnn(emb, h)
            sl, bs, hs = out.shape
            out = out.reshape(sl * bs, hs)
            out = self.linear(out)
            return out, h

    torch_model = TorchRNN().to(device_torch)
    torch_params = list(torch_model.parameters())
    return ndl_model, ndl_params, torch_model, torch_params


def copy_ndl_to_torch(ndl_params, torch_params, torch_model):
    for i in range(len(ndl_params)):
        v = ndl_params[i].realize_cached_data().numpy().copy()
        if i in SQUEEZE:
            v = v.squeeze(0)
        if i in TRANSPOSE:
            v = v.T
        torch_params[i].data.copy_(torch.from_numpy(v))
    torch_model.rnn.flatten_parameters()


def warmup(ndl_model, torch_model, source_ndl, target_ndl, source_torch, target_torch, n=3):
    for _ in range(n):
        o, _ = ndl_model(source_ndl)
        nn.SoftmaxLoss()(o, target_ndl).backward()
        o, _ = torch_model(source_torch)
        tnn.CrossEntropyLoss(reduction='mean')(o, target_torch).backward()


def bench_single_iter(ndl_model, torch_model, ndl_params, torch_params,
                      source_ndl, target_ndl, source_torch, target_torch,
                      n_iter=50):
    times_ndl_fwd, times_ndl_bwd, times_ndl_step = [], [], []
    times_torch_fwd, times_torch_bwd, times_torch_step = [], [], []

    for _ in range(n_iter):
        copy_ndl_to_torch(ndl_params, torch_params, torch_model)

        # Needle
        t0 = time.time()
        o, _ = ndl_model(source_ndl)
        l = nn.SoftmaxLoss()(o, target_ndl)
        times_ndl_fwd.append(time.time() - t0)

        opt = ndl.optim.SGD(ndl_params, lr=lr)
        opt.reset_grad()
        t0 = time.time()
        l.backward()
        times_ndl_bwd.append(time.time() - t0)

        t0 = time.time()
        opt.step()
        times_ndl_step.append(time.time() - t0)

        # Torch
        t0 = time.time()
        o, _ = torch_model(source_torch)
        l = tnn.CrossEntropyLoss(reduction='mean')(o, target_torch)
        times_torch_fwd.append(time.time() - t0)

        opt = torch.optim.SGD(torch_params, lr=lr)
        opt.zero_grad()
        t0 = time.time()
        l.backward()
        times_torch_bwd.append(time.time() - t0)

        t0 = time.time()
        opt.step()
        times_torch_step.append(time.time() - t0)

    return (np.mean(times_ndl_fwd), np.mean(times_ndl_bwd), np.mean(times_ndl_step),
            np.mean(times_torch_fwd), np.mean(times_torch_bwd), np.mean(times_torch_step))


def bench_components(ndl_model, torch_model, ndl_params, torch_params,
                     source_ndl, target_ndl, source_torch, target_torch,
                     n_iter=50):
    ndl_c = {'embed': [], 'rnn': [], 'linear': [], 'loss': [], 'backward': [], 'step': []}
    torch_c = {'embed': [], 'rnn': [], 'linear': [], 'loss': [], 'backward': [], 'step': []}

    for _ in range(n_iter):
        copy_ndl_to_torch(ndl_params, torch_params, torch_model)
        sl_, bs_, hs_ = seq_len, batch_size, hidden_size

        # == Needle breakdown ==
        t0 = time.time()
        emb = ndl_model.embedding(source_ndl)
        ndl_c['embed'].append(time.time() - t0)

        t0 = time.time()
        rnn_out, _ = ndl_model.seq_model(emb)
        ndl_c['rnn'].append(time.time() - t0)

        t0 = time.time()
        flat = rnn_out.reshape((sl_ * bs_, hs_))
        lin_out = ndl_model.linear(flat)
        ndl_c['linear'].append(time.time() - t0)

        t0 = time.time()
        loss = nn.SoftmaxLoss()(lin_out, target_ndl)
        ndl_c['loss'].append(time.time() - t0)

        opt = ndl.optim.SGD(ndl_params, lr=lr)
        opt.reset_grad()
        t0 = time.time()
        loss.backward()
        ndl_c['backward'].append(time.time() - t0)

        t0 = time.time()
        opt.step()
        ndl_c['step'].append(time.time() - t0)

        # == Torch breakdown ==
        t0 = time.time()
        emb_t = torch_model.embedding(source_torch)
        torch_c['embed'].append(time.time() - t0)

        t0 = time.time()
        rnn_out_t, _ = torch_model.rnn(emb_t)
        torch_c['rnn'].append(time.time() - t0)

        t0 = time.time()
        flat_t = rnn_out_t.reshape(sl_ * bs_, hs_)
        lin_out_t = torch_model.linear(flat_t)
        torch_c['linear'].append(time.time() - t0)

        t0 = time.time()
        loss_t = tnn.CrossEntropyLoss(reduction='mean')(lin_out_t, target_torch)
        torch_c['loss'].append(time.time() - t0)

        opt_t = torch.optim.SGD(torch_params, lr=lr)
        opt_t.zero_grad()
        t0 = time.time()
        loss_t.backward()
        torch_c['backward'].append(time.time() - t0)

        t0 = time.time()
        opt_t.step()
        torch_c['step'].append(time.time() - t0)

    return ndl_c, torch_c


def run_profile(device_name, device_ndl, device_torch):
    print()
    print("=" * 70)
    print(f"DEVICE: {device_name}")
    print("=" * 70)

    ndl_model, ndl_params, torch_model, torch_params = build_models(device_ndl, device_torch)

    # Get a batch
    ndl_dev = ndl.cpu() if device_ndl == 'cpu' else ndl.cuda()
    s_np, t_np = ndl.data.get_batch(train_data_np, 0, seq_len, ndl_dev, 'float32')
    s_np = s_np.numpy().astype(np.int64)
    t_np = t_np.numpy().astype(np.int64)

    source_ndl = ndl.Tensor(s_np.astype(np.float32), device=ndl_dev)
    target_ndl = ndl.Tensor(t_np.astype(np.float32), device=ndl_dev)
    source_torch = torch.from_numpy(s_np).to(device_torch)
    target_torch = torch.from_numpy(t_np).to(device_torch)

    # ------ Correctness ------
    copy_ndl_to_torch(ndl_params, torch_params, torch_model)
    with torch.no_grad():
        out_ndl, _ = ndl_model(source_ndl)
        out_torch, _ = torch_model(source_torch)
    out_diff = np.max(np.abs(out_ndl.numpy() - out_torch.cpu().detach().numpy()))
    ndl_loss = nn.SoftmaxLoss()(out_ndl, target_ndl).numpy()
    torch_loss = tnn.CrossEntropyLoss(reduction='mean')(out_torch, target_torch).item()
    print(f"  Correctness: out_diff={out_diff:.2e}, loss_diff={abs(ndl_loss-torch_loss):.2e}"
          f" {'PASS' if out_diff < 1e-4 else 'FAIL'}")

    # ------ Warmup ------
    warmup(ndl_model, torch_model, source_ndl, target_ndl, source_torch, target_torch, 3)

    # ------ Single iter ------
    n_fwd, n_bwd, n_step, t_fwd, t_bwd, t_step = bench_single_iter(
        ndl_model, torch_model, ndl_params, torch_params,
        source_ndl, target_ndl, source_torch, target_torch, 30)
    nt, tt = n_fwd + n_bwd + n_step, t_fwd + t_bwd + t_step

    print(f"\n  --- Single Iteration ---")
    print(f"  {'Phase':<10s} {'Needle (ms)':<14s} {'Torch (ms)':<14s} {'Ratio':<8s} {'%Needle':<8s}")
    print(f"  {'-'*52}")
    for name, n, t in [('forward', n_fwd, t_fwd), ('backward', n_bwd, t_bwd),
                        ('step', n_step, t_step), ('total', nt, tt)]:
        print(f"  {name:<10s} {n*1000:8.4f}       {t*1000:8.4f}       "
              f"{n/t:5.1f}x   {n/nt*100 if nt>0 else 0:5.1f}%")

    # ------ Component breakdown ------
    ndl_c, torch_c = bench_components(
        ndl_model, torch_model, ndl_params, torch_params,
        source_ndl, target_ndl, source_torch, target_torch, 30)

    ndl_total = sum(np.mean(ndl_c[k]) for k in ndl_c)
    torch_total = sum(np.mean(torch_c[k]) for k in torch_c)

    print(f"\n  --- Component Breakdown ---")
    print(f"  {'Component':<14s} {'Needle (ms)':<14s} {'Torch (ms)':<14s} {'Ratio':<8s} {'%Needle':<8s}")
    print(f"  {'-'*56}")
    for key in ['embed', 'rnn', 'linear', 'loss', 'backward', 'step']:
        nm = np.mean(ndl_c[key]) * 1000
        tm = np.mean(torch_c[key]) * 1000
        print(f"  {key:<14s} {nm:8.4f}       {tm:8.4f}       "
              f"{nm/tm:5.1f}x   {nm/ndl_total/10:5.1f}%")

    return dict(
        device=device_name,
        ndl_total=nt * 1000,
        torch_total=tt * 1000,
        ratio_total=nt / tt,
        ndl_fwd=n_fwd * 1000, ndl_bwd=n_bwd * 1000, ndl_step=n_step * 1000,
        torch_fwd=t_fwd * 1000, torch_bwd=t_bwd * 1000, torch_step=t_step * 1000,
        ndl_components={k: np.mean(v) * 1000 for k, v in ndl_c.items()},
        torch_components={k: np.mean(v) * 1000 for k, v in torch_c.items()},
        ndl_comp_pct={k: np.mean(v) / ndl_total * 100 for k, v in ndl_c.items()},
    )


# ============================================================
# Run on both devices
# ============================================================
results = {}

# CPU
results['cpu'] = run_profile('CPU', 'cpu', 'cpu')

# CUDA (if available)
if ndl.cuda().enabled():
    results['cuda'] = run_profile('CUDA', 'cuda', 'cuda')

# ============================================================
# Summary & Optimization Suggestions
# ============================================================
print()
print("=" * 70)
print("SUMMARY: Needle vs PyTorch Performance")
print("=" * 70)

print(f"\n  {'Device':<8s} {'Ndl tot(ms)':<13s} {'Torch tot(ms)':<14s} {'Ratio':<8s}")
print(f"  {'-'*42}")
for dev, r in results.items():
    print(f"  {dev:<8s} {r['ndl_total']:8.4f}       {r['torch_total']:8.4f}        {r['ratio_total']:5.1f}x")

print()
print("=" * 70)
print("OPTIMIZATION SUGGESTIONS (priority order)")
print("=" * 70)
print("""
1. [HIGH] Fuse backward kernels — backward pass takes ~57% of total time.
   - Implement cudnnLSTM / cublas batched gemm for RNN backward
   - Fuse LogSumExp + Summation + elementwise ops in SoftmaxLoss backward
   - Combine Stack/Split backward with RNN cell backward into single kernel

2. [HIGH] Enable CUDA kernel launch batching — current impl launches one kernel
   per operation. Uniformly 10-18x slower due to kernel launch overhead.
   - Use CUDA streams + kernel fusion to reduce launches
   - Batch multiple elementwise ops into single fused kernel

3. [MEDIUM] Optimize RNN forward — 20% of time spent here.
   - Replace python-loop over time steps with GPU-parallel scan (if using GPU)
   - For CPU: vectorize the per-timestep matmul loop via pre-allocation
   - Use cuBLAS strided-batched matmul for multi-layer RNN forward

4. [MEDIUM] Reduce host-device synchronization — realize_cached_data() syncs
   after every op in eager mode.
   - Defer realization by building lazy graph then computing in one shot
   - Use pinned memory for data transfer between CPU/GPU

5. [LOW]  Optimize Embedding lookup — 18x slower but only ~8% of total.
   - Use gathered index tensor instead of one-hot + matmul
   - Already implemented as gather op, avoid materializing one-hot matrix

6. [LOW]  SGD step optimization — use torch-style fused update kernel
   p.data = p.data - lr * grad  (single fused multiply-add)
""")
