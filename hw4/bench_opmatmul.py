import sys
sys.path.append('./python')
import needle as ndl
import needle.backend_ndarray.ndarray as ndarray
import numpy as np
import torch, time

device = ndl.cuda()
W, B = 5, 20

cases = [
    (128, 128, 128), (256, 256, 256), (512, 512, 512),
    (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
    (4096, 1024, 4096), (512, 4096, 512),
]

out = []
def L(s):
    print(s, flush=True)
    out.append(s)

L("=" * 90)
L(f"Matrix Mul Benchmark | GPU: {torch.cuda.get_device_name(0)} | warmup={W} bench={B}")
L("=" * 90)
L(f"  {'Shape':<16s} {'Reg ms':>10s} {'Opt ms':>10s} {'Torch ms':>10s} {'O/T':>8s} {'R/O':>8s} {'Err_r':>10s} {'Err_o':>10s}")
L(f"  {'-'*86}")

results = []
for M, N, P in cases:
    a_np = np.random.randn(M, N).astype(np.float32)
    b_np = np.random.randn(N, P).astype(np.float32)
    a_ndl = ndarray.NDArray(a_np, device=device)
    b_ndl = ndarray.NDArray(b_np, device=device)
    a_t = torch.from_numpy(a_np).cuda()
    b_t = torch.from_numpy(b_np).cuda()

    for _ in range(W):
        _ = a_ndl @ b_ndl; _ = a_ndl.op_matmul(b_ndl); _ = torch.matmul(a_t, b_t)
    torch.cuda.synchronize()

    tr, to, tt = [], [], []
    for _ in range(B):
        t0 = time.perf_counter(); _ = a_ndl @ b_ndl; torch.cuda.synchronize()
        tr.append(time.perf_counter() - t0)
    for _ in range(B):
        t0 = time.perf_counter(); _ = a_ndl.op_matmul(b_ndl); torch.cuda.synchronize()
        to.append(time.perf_counter() - t0)
    for _ in range(B):
        t0 = time.perf_counter(); _ = torch.matmul(a_t, b_t); torch.cuda.synchronize()
        tt.append(time.perf_counter() - t0)

    rm, om, tm = np.mean(tr)*1000, np.mean(to)*1000, np.mean(tt)*1000
    gflops = 2.0*M*N*P/1e9
    c_t = torch.matmul(a_t, b_t).cpu().numpy()
    er = np.max(np.abs((a_ndl@b_ndl).numpy() - c_t))
    eo = np.max(np.abs(a_ndl.op_matmul(b_ndl).numpy() - c_t))
    L(f"  {f'{M}x{N}x{P}':<16s} {rm:10.3f} {om:10.3f} {tm:10.3f} {om/tm:7.2f}x {rm/om:7.1f}x {er:9.2e} {eo:9.2e}")
    results.append((M,N,P,rm,om,tm,om/tm,rm/om,er,eo,gflops,gflops/(om/1000)/1000,gflops/(tm/1000)/1000))

# raw kernel timing
L("")
L("=" * 90)
L("Raw kernel timing (N=4096, pre-compact + pre-alloc, direct device.mod.op_matmul call)")
L("=" * 90)
N = 4096
a_np = np.random.randn(N, N).astype(np.float32)
b_np = np.random.randn(N, N).astype(np.float32)
a_ndl = ndarray.NDArray(a_np, device=device)
b_ndl = ndarray.NDArray(b_np, device=device)
a_t = torch.from_numpy(a_np).cuda()
b_t = torch.from_numpy(b_np).cuda()

for _ in range(W):
    ac = a_ndl.compact(); bc = b_ndl.compact()
    oc = ndarray.NDArray.make((N,N), device=device)
    device.mod.op_matmul(ac._handle, bc._handle, oc._handle, N, N, N)
    torch.matmul(a_t, b_t)
torch.cuda.synchronize()

tr2, tt2 = [], []
for _ in range(B):
    ac = a_ndl.compact(); bc = b_ndl.compact()
    oc = ndarray.NDArray.make((N,N), device=device)
    t0 = time.perf_counter()
    device.mod.op_matmul(ac._handle, bc._handle, oc._handle, N, N, N)
    torch.cuda.synchronize()
    tr2.append(time.perf_counter() - t0)
for _ in range(B):
    t0 = time.perf_counter(); torch.matmul(a_t, b_t); torch.cuda.synchronize()
    tt2.append(time.perf_counter() - t0)

rm2, tm2 = np.mean(tr2)*1000, np.mean(tt2)*1000
gf = 2.0*N*N*N/1e9
L(f"  raw needle (incl compact+alloc): {rm2:.3f}ms = {gf/(rm2/1000)/1000:.2f} TFLOPS")
L(f"  torch.matmul:                   {tm2:.3f}ms = {gf/(tm2/1000)/1000:.2f} TFLOPS")
L(f"  ratio: {rm2/tm2:.2f}x")

L("")
L("=" * 90)
L("SUMMARY")
L("=" * 90)
L(f"  {'Shape':<16s} {'Opt/Torch':>10s} {'Reg/Opt':>10s} {'Opt TF':>10s} {'Torch TF':>10s}")
L(f"  {'-'*62}")
for r in results:
    L(f"  {f'{r[0]}x{r[1]}x{r[2]}':<16s} {r[6]:9.2f}x {r[7]:9.1f}x {r[11]:9.2f} {r[12]:9.2f}")

with open("bench_opmatmul.log", "w") as f:
    f.write("\n".join(out))
print("\nLog -> bench_opmatmul.log", flush=True)
