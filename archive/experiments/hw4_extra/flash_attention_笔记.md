# FlashAttention 实现学习笔记

## 1. 问题背景：标准 Attention 为什么慢

标准 scaled dot-product attention 的数学定义：

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V$$

其中 $Q, K, V \in \mathbb{R}^{N \times D}$（单头，N = 序列长度，D = 头维度）。

**朴素实现的三步走：**

```
S = Q @ K^T         # (N, D) @ (D, N) → (N, N)     ← 写入 HBM
P = softmax(S)       # (N, N)                        ← 读 S, 写 P
O = P @ V            # (N, N) @ (N, D) → (N, D)     ← 读 P, V, 写 O
```

每一步都把中间结果写回 GPU 全局内存（HBM），然后再读回来。**核心问题是 $O(N^2)$ 的 attention 矩阵 S 和 P 被完整材料化并写入/读取 HBM**。对 N=2048，这就是 4M 个 float = 16MB；对 N=8192，即 64M 个 float = 256MB。

原来的 needle MHA 实现更糟——它把 matmul 展开成 5D broadcast → element-multiply → sum，中间张量膨胀到 `(B, H, N, N, D)` 即 5D，内存放大 D 倍。

## 2. 核心算法：Online Softmax

FlashAttention 的核心思想来自两篇论文：

- **Online normalizer calculation for softmax**（Milakov & Gimelshein, 2018）
- **FlashAttention**（Dao et al., 2022）

其关键观察是：**softmax 不需要完整的输入就能逐步计算**。

### 2.1 标准 Softmax（两遍扫描）

$$softmax(x)_j = \frac{e^{x_j - m}}{\sum_k e^{x_k - m}}, \quad m = \max(x)$$

缺点：需要第一遍找 max，第二遍算 exp 和 sum。

### 2.2 Online Softmax（一遍扫描）

思想：**维护 running max 和 running sum，逐元素更新**。

假设我们逐个处理 scores $s_1, s_2, ..., s_N$。维护三个状态变量：

| 变量 | 含义 | 初始值 |
|------|------|--------|
| $m_i$ | 当前见过的最大 score | $-\infty$ |
| $\ell_i$ | 当前所有 exp(score - m) 之和 | 0 |
| $\text{acc}[d]$ | 当前输出累加器（D 维向量） | 0 |

当遇到新 score $s$ 和对应的 $V_j$ 时（伪代码）：

```python
def online_softmax_step(m_i, l_i, acc, s, V_j):
    m_new = max(m_i, s)

    # exp(s) 的缩放：减去新的 max
    exp_val = exp(s - m_new)

    # 旧累加和也需要重新缩放
    l_new = l_i * exp(m_i - m_new) + exp_val

    # 旧输出需要按相同因子缩放
    rescale = exp(m_i - m_new)
    acc = acc * rescale + exp_val * V_j

    return m_new, l_new, acc
```

**关键性质**：处理完所有 N 个 score 后，最终输出 $O = \text{acc} / \ell_N = \text{softmax}(S) \cdot V$，与一次性计算完全等价。

### 2.3 正确性证明（缩略）

归纳法。设处理完前 k 个 score 后有状态 $(m_k, \ell_k, \text{acc}_k)$，满足：

$$\ell_k = \sum_{j=1}^k e^{s_j - m_k}, \quad \text{acc}_k = \sum_{j=1}^k e^{s_j - m_k} \cdot V_j$$

处理第 k+1 个 score 时：

$$\begin{aligned}
m_{k+1} &= \max(m_k, s_{k+1}) \\
\ell_{k+1} &= \ell_k \cdot e^{m_k - m_{k+1}} + e^{s_{k+1} - m_{k+1}} \\
&= \sum_{j=1}^k e^{s_j - m_{k+1}} + e^{s_{k+1} - m_{k+1}} = \sum_{j=1}^{k+1} e^{s_j - m_{k+1}}
\end{aligned}$$

同理可证 $\text{acc}_{k+1}$ 成立。因此 $\text{acc}_N / \ell_N$ 即为精确的 softmax-weighted 输出。

## 3. CUDA Kernel 实现

### 3.1 Thread-to-Data Mapping

```
每个 CUDA thread 处理 1 个 Q row
grid  = (B * H * N, 1, 1)    # 每个 (batch, head, query_pos) 一个 block
block = (256, 1, 1)           # 256 个线程
```

线性 thread index → (batch, head, query_pos) 的解码：

```c
size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
size_t i = idx % N;            // 当前 query position (0..N-1)
size_t h = (idx / N) % H;      // 当前 head (0..H-1)
size_t b = idx / (N * H);      // 当前 batch (0..B-1)
```

### 3.2 数据访问模式

```c
const scalar_t *Qi = pQ + head_off + i * D;  // 当前 Q row（只读一次）
scalar_t       *Oi = pOut + head_off + i * D; // 输出位置（读+写 = 累加器）

// 内层循环：逐个扫描所有 key/value positions
for (size_t j = 0; j < N; j++) {
    if (causal && j > i) continue;  // causal mask 在这处理
    const scalar_t *Kj = pK + head_off + j * D;
    const scalar_t *Vj = pVal + head_off + j * D;
    ...
}
```

**内存特性**：
- `Qi` 只在循环前加载一次（寄存器中重用 N 次）
- `Kj` 和 `Vj` 按 j 顺序扫描，每次读 D 个 float → 对 L1/L2 cache 友好
- 输出 `Oi` 既是累加器也是最终输出，省去了单独的累加器寄存器数组

### 3.3 内循环：Dot Product + Online Softmax

```c
// 1. 计算 dot product Qi · Kj
float score = 0.0f;
for (size_t d = 0; d < D; d++)
    score += Qi[d] * Kj[d];
score *= softmax_scale;   // 乘以 1/sqrt(D)

// 2. Online softmax 更新
float m_new = fmaxf(m_i, score);
float exp_val = expf(score - m_new);
float l_new  = l_i * expf(m_i - m_new) + exp_val;
float rescale = expf(m_i - m_new);

// 3. 累加 Vj 到输出
for (size_t d = 0; d < D; d++)
    Oi[d] = Oi[d] * rescale + exp_val * Vj[d];

m_i = m_new;
l_i = l_new;
```

### 3.4 最后归一化

```c
float inv_l = 1.0f / l_i;
for (size_t d = 0; d < D; d++)
    Oi[d] *= inv_l;
```

### 3.5 Host 端启动函数

```c
void FlashAttention(
    const CudaArray &q, const CudaArray &k, const CudaArray &v,
    CudaArray *out,
    uint32_t B, uint32_t H, uint32_t N, uint32_t D,
    bool causal, float softmax_scale)
{
    size_t total_rows = B * H * N;
    CudaDims dim = CudaOneDim(total_rows);
    FlashAttnFwdKernel<<<dim.grid, dim.block>>>(
        q.ptr, k.ptr, v.ptr, out->ptr,
        B, H, N, D, causal, softmax_scale);
}
```

### 3.6 与 Full FlashAttention 的差异

| 特性 | 完整 FlashAttention | 本实现 |
|------|-------------------|--------|
| K/V 分 tile 加载到 shared memory | ✓ | ✗（直接读 global memory） |
| Q 分 tile 加载到 shared memory | ✓ | ✗（每个 thread 独占 Q row） |
| 内循环中 shared memory 上的 tile matmul | ✓ | ✗（逐元素 dot product） |
| Backward recompute + dQ/dK/dV in CUDA | ✓ | 梯度用 numpy 解析计算 |

本实现更准确的名字是 **"Fused Online-Softmax Attention"**。它省去了 $N \times N$ attention 矩阵的材料化，但没有使用 shared memory tiling 进一步优化内存带宽。

## 4. 框架集成：4 层调用链

```
nn_transformer.py          MultiHeadAttention.forward()
                            │ use_flash_attn=True
                            ▼
ops_mathematic.py          ops.flash_attention(q, k, v, ...)
                            │ FlashAttention.compute()
                            ▼
ndarray.py                 q.flash_attention(k, v, ...)
                            │ self.device.flash_attention(...)
                            ▼
ndarray_backend_cuda.cu    FlashAttnFwdKernel<<<>>>  (GPU)
ndarray_backend_numpy.py   flash_attention()          (CPU ref)
```

### 4.1 切换机制（nn_transformer.py）

```python
class MultiHeadAttention(Module):
    def __init__(self, *, use_flash_attn=False, ...):
        self.use_flash_attn = use_flash_attn

    def forward(self, q, k, v):
        if self.use_flash_attn:
            softmax_scale = 1.0 / np.sqrt(q.shape[-1])
            result = ops.flash_attention(q, k, v,
                                         causal=self.causal,
                                         softmax_scale=softmax_scale)
            probs = None  # FlashAttention 不暴露中间 attention weights
        else:
            pre_score = self.matmul(q, k) / np.sqrt(q_dim)
            if self.causal:
                pre_score += self.create_causal_mask(...)
            probs = self.dropout(self.softmax(pre_score))
            result = self.matmul(probs, v.transpose())
        return result, probs
```

`use_flash_attn` 参数从 `MultiHeadAttention` → `AttentionLayer` → `TransformerLayer` → `Transformer` 逐级透传。

### 4.2 TensorOp 层（ops_mathematic.py）

```python
class FlashAttention(TensorOp):
    def __init__(self, causal=False, softmax_scale=1.0):
        self.causal = causal
        self.softmax_scale = softmax_scale

    def compute(self, q, k, v):
        # 直接调用 NDArray 的 flash_attention 方法
        return q.flash_attention(k, v,
                                 causal=self.causal,
                                 softmax_scale=self.softmax_scale)

    def gradient(self, out_grad, node):
        # 见第 5 节
        ...

def flash_attention(q, k, v, causal=False, softmax_scale=1.0):
    return FlashAttention(causal=causal, softmax_scale=softmax_scale)(q, k, v)
```

### 4.3 NDArray 层（ndarray.py）

```python
def flash_attention(self, k, v, causal=False, softmax_scale=1.0):
    assert self.ndim == 4 and self.shape == k.shape == v.shape
    B, H, N, D = self.shape
    out = NDArray.make(self.shape, device=self.device)
    self.device.flash_attention(
        self.compact()._handle,
        k.compact()._handle,
        v.compact()._handle,
        out._handle,
        B, H, N, D,
        causal,
        softmax_scale,
    )
    return out
```

### 4.4 后端注册（ndarray_backend_numpy.py + ndarray_backend_cuda.cu）

Numpy 后端（用于正确性验证）：
```python
def flash_attention(q, k, v, out, B, H, N, D, causal, softmax_scale):
    q_arr = q.array.reshape(B, H, N, D)
    k_arr = k.array.reshape(B, H, N, D)
    v_arr = v.array.reshape(B, H, N, D)

    S = q_arr @ k_arr.transpose(0, 1, 3, 2)
    S *= softmax_scale
    if causal:
        mask = np.triu(np.ones((N, N), dtype=np.float32) * (-np.inf), 1)
        S += mask.reshape(1, 1, N, N)

    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=-1, keepdims=True)
    O = P @ v_arr

    out.array[:] = O.reshape(-1)
```

CUDA 注册：
```cpp
PYBIND11_MODULE(ndarray_backend_cuda, m) {
    // ... 其他注册 ...
    m.def("flash_attention", FlashAttention);
}
```

### 4.5 一个实际的坑：`#define V 2` 宏冲突

源文件 `ndarray_backend_cuda.cu` 第 397 行有 `#define V 2`。CUDA kernel 参数名 `V` 会被预处理器替换为 `2`，导致编译错误：

```
error: expected a ")"
    const scalar_t *Q, const scalar_t *K, const scalar_t *2,
```

修复：将参数重命名为 `pVal`。

## 5. 梯度（Backward Pass）

### 5.1 策略

FlashAttention 不保存中间 attention matrix P，backward 有两种策略：

1. **Recompute**：在 backward 中重算 $S = QK^T$ 和 $P = \text{softmax}(S)$，然后计算解析梯度（需要 $O(N^2)$ 内存存储 P，与 forward 的节省目标矛盾）
2. **Backward with recomputation + tiling**：逐 tile 重算 forward，不存储完整的 P

本实现使用策略 1 的 numpy 版本——重算 P 并计算解析梯度。这在概念上简单，但确实会材料化 N×N 的 attention matrix。如果要在训练中使用，应实现策略 2。

### 5.2 数学推导

Attention forward: $S = QK^T \cdot \text{scale}$, $P = \text{softmax}(S)$, $O = PV$

梯度：

$$dP = dO \cdot V^T$$

Softmax 梯度（$P_j = e^{s_j} / \sum_k e^{s_k}$）：

$$\frac{\partial P_j}{\partial s_i} = P_j(\delta_{ij} - P_i)$$

链式法则：

$$dS_i = \sum_j dP_j \cdot \frac{\partial P_j}{\partial s_i} = P_i \cdot \left(dP_i - \sum_j dP_j \cdot P_j\right)$$

矩阵形式（element-wise）：

$$dS = P \odot (dP - \text{sum}(dP \odot P, \text{axis}=-1, \text{keepdims=True}))$$

其余梯度：

$$dQ = dS \cdot K \cdot \text{scale}$$
$$dK = dS^T \cdot Q \cdot \text{scale}$$
$$dV = P^T \cdot dO$$

### 5.3 代码实现

```python
def gradient(self, out_grad, node):
    q, k, v = [inp.realize_cached_data() for inp in node.inputs]
    dO = out_grad.realize_cached_data()

    B, H, N, D = q.shape
    scale = self.softmax_scale

    # 转 numpy 做批量线性代数
    q_np = q.numpy().reshape(B, H, N, D)
    k_np = k.numpy().reshape(B, H, N, D)
    v_np = v.numpy().reshape(B, H, N, D)
    dO_np = dO.numpy().reshape(B, H, N, D)

    # ── recompute forward: S, P ──
    S_np = q_np @ k_np.transpose(0, 1, 3, 2)  # (B, H, N, N)
    S_np *= scale
    if self.causal:
        mask = np.triu(np.ones((N, N), dtype=np.float32) * (-np.inf), 1)
        S_np += mask.reshape(1, 1, N, N)

    S_max = S_np.max(axis=-1, keepdims=True)
    S_exp = np.exp(S_np - S_max)
    P_np = S_exp / S_exp.sum(axis=-1, keepdims=True)

    # ── gradients ──
    dP_np = dO_np @ v_np.transpose(0, 1, 3, 2)          # dP = dO @ V^T
    dS_np = P_np * (dP_np - (dP_np * P_np).sum(         # softmax grad
        axis=-1, keepdims=True))
    dQ_np = (dS_np @ k_np) * scale                        # dQ = dS @ K * scale
    dK_np = dS_np.transpose(0, 1, 3, 2) @ q_np * scale   # dK = dS^T @ Q * scale
    dV_np = P_np.transpose(0, 1, 3, 2) @ dO_np           # dV = P^T @ dO

    # 转回 Tensor
    from ..autograd import Tensor
    dQ = Tensor(dQ_np.reshape(B * H * N * D), device=q.device, dtype=q.dtype)
    dK = Tensor(dK_np.reshape(B * H * N * D), device=k.device, dtype=k.dtype)
    dV = Tensor(dV_np.reshape(B * H * N * D), device=v.device, dtype=v.dtype)

    return (dQ, dK, dV)
```

## 6. 性能数据分析

### 6.1 MHA（MultiHeadAttention）Forward

| Config | orig | flash | torch | flash_vs_orig |
|--------|------|-------|-------|---------------|
| tiny (2×4×16×16) | 257 µs | 46 µs | 117 µs | **5.6x** |
| small (8×8×64×32) | 10.0 ms | 570 µs | 115 µs | **17.6x** |
| medium (16×8×128×64) | 111 ms | 6.4 ms | 814 µs | **17.4x** |
| large (4×8×128×64) | 31 ms | 2.2 ms | 226 µs | **14.2x** |

**结论：MHA 层面 flash 取得了 5.6x–17.6x 的加速。**

- tiny 加速最小（5.6x）因为 N=16 时 5D broadcast 的绝对开销还不大
- small/medium/large 稳定在 14–18x，消除了 5D broadcast matmul 和 softmax 的 host-device sync

### 6.2 TransformerLayer Forward

| Config | orig | flash | torch | flash_vs_orig |
|--------|------|-------|-------|---------------|
| tiny | 1.12 ms | 860 µs | 222 µs | **1.3x** |
| small | 21.4 ms | 12.9 ms | 229 µs | **1.7x** |
| medium | 296 ms | 187 ms | 2.1 ms | **1.6x** |
| large | 78 ms | 50 ms | 564 µs | **1.6x** |

**结论：TransformerLayer 层面仅 1.3x–1.7x 加速。** 原因是 FlashAttention 只加速了 MHA 部分（占总时间 ~15%），其余 85% 时间花在 Linear、LayerNorm、Dropout 等仍然走原路径的 ops 上。

### 6.3 序列长度 Scaling

| seq_len | flash_vs_orig |
|---------|---------------|
| 16 | **1.4x** |
| 32 | **1.3x** |
| 64 | **1.6x** |
| 128 | **2.2x** |
| 256 | **3.1x** |
| 512 | **4.3x** |

**序列越长 flash 优势越大**——MHA 原实现的 O(N²) 广播开销随 seq_len 增长，而 flash 的 O(N) 扫描保持了线性增长。这是 FlashAttention 的核心价值。

### 6.4 cProfile 函数调用对比

| 指标 | flash_attn=True | flash_attn=False |
|------|-----------------|------------------|
| 函数调用数 | 5,808 | 7,208 |
| 总耗时 | 0.018s | 0.022s |

flash 路径少了 ~1,400 次函数调用，主要来自 `matmul`、`softmax` 内部不再需要的多次 broadcast/reshape/mul/sum/div 调用链。

### 6.5 正确性验证

```
max absolute difference: 4.768372e-07  ✓ PASS
```

低于 1e-4 阈值，差异来自 float32 精度下 `expf` 计算顺序的微小不同。

## 7. 关键技术决策回顾

| 决策 | 选择 | 原因 |
|------|------|------|
| Thread mapping | 1 thread = 1 Q row | 最大化并行度（B×H×N 个独立任务） |
| Shared memory | 不使用 | 优先正确性，后续可加 tile 优化 |
| 累加器位置 | 复用 `Oi`（output array） | 避免固定大小寄存器数组，兼容任意大 D |
| `expf` 精度 | 直接用 CUDA intrinsic | 单精度 attention 足够 |
| 梯度 | numpy 解析计算 | 正确且可读，重算 P 的代价可接受 |
| `V` 宏冲突 | 用 `pVal` 替代 `V` | 源文件有 `#define V 2` 导致预处理器替换 |

## 8. 下一步优化方向

| 优先级 | 方向 | 预期收益 |
|--------|------|----------|
| **P0** | 把 matmul 替换为 cuBLAS `cublasSgemm` | Linear 层加速 5–10x |
| **P0** | Shared memory tiling in FlashAttnKernel | MHA 再快 2–3x |
| **P1** | LayerNorm CUDA kernel | 减少 3 个逐 op 的 norm 调用 |
| **P1** | Warp-level reduction for dot product | 减少每步 O(D) 的串行计算 |
| **P2** | Backward recompute in CUDA kernel | 训练时 backward 不再需要 numpy 中转 |
| **P2** | Fused LayerNorm + Linear kernel | 进一步减少 kernel launch overhead |
| **P3** | 内存池减少 alloc (`make`) 开销 | 减少 15–25% 的 metadata 开销 |
