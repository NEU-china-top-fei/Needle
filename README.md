# Needle GPU

**从课程深度学习框架到可复现的 CUDA 算子优化项目。**

基于 [CMU 10-714 Deep Learning Systems](https://dlsyscourse.org/) 的 Needle，实现并比较 FP32 SIMT GEMM、WMMA Tensor Core、`cp.async` 双缓冲，以及 Transformer 前向的注意力和 LayerNorm 融合。Python 自动微分框架通过 pybind11 调用自写 CUDA 内核。

[实现与边界](docs/architecture.md) · [实测报告](docs/performance.md) · [验证记录](docs/validation.md) · [项目贡献与简历表述](docs/project-plan.md)

## 已验证的结果

在 **RTX 4060 Laptop 8 GB，CUDA 13.2，SM89** 上重复运行三轮：

| 工作负载 | 基线 → 优化 | 结果 | 数值口径 |
|---|---|---|---|
| 2048³ GEMM 框架调用 | FP32 SIMT 12.6145 ms → TC 2.6691 ms | **4.73×** | FP16 输入 / FP32 累加，相对 FP32 L2 误差 2.94e−4 |
| 2048³ 预打包 TC 内核 | 同步 WMMA 1.2861 ms → 双缓冲 1.1933 ms | **1.08×，14.40 TFLOPS** | 同样的 FP16 输入 / FP32 输出 |
| TransformerLayer，B=8/S=64/D=256/H=8 | 原始前向 22.4074 ms → TC + 注意力融合 + LN 融合 2.8092 ms | **7.98×** | 同权重、同输入、dropout=0，相对 L2 误差 5.62e−4 |

表中为三轮统计量的中位数。GEMM 框架计时包含转换与分配；TC 不等同于严格 FP32 计算。2048³ 同精度 cuBLAS 预打包时间为 **0.6110 ms**，自写内核仍慢约 **1.95×**。完整形状扫描、波动范围、精度和原始 JSON 见[实测报告](docs/performance.md)。

GPU 测试 **110 passed / 1 skipped**；memcheck、racecheck、synccheck 均通过。独立 CPU 构建测试 **55 passed / 56 skipped**；CPU 环境跳过 CUDA 测试。GitHub Actions 配置覆盖 CPU 构建和测试。

## 技术实现

| 模式 | 输入 / 累加输出 | 实现 |
|---|---|---|
| `baseline`（默认） | FP32 / FP32 | 原始课程 GEMM |
| `simt` | FP32 / FP32 | 分块、寄存器复用、float4 访存、双缓冲；非对齐尺寸回退 |
| `wmma` | FP16 / FP32 | 64×64×32 分块，四个 warp，同步加载 |
| `tensorcore` | FP16 / FP32 | 相同分块，WMMA + cp.async 双缓冲 |

普通 `@`、Linear 和 Transformer 投影共享同一分派路径。NDArray 存储仍为 FP32，TC 在 GPU 上转换并补齐输入，最后裁剪输出；不修改默认精度。实现来自作者的 [GEMM 笔记](https://neu-china-top-fei.github.io/2026/06/04/DLsysSet/GEMM/)，详见[架构说明](docs/architecture.md)。

## 快速开始

需要 Python 3.10+、CMake 3.24+ 和 C++ 编译器；GPU 路径还需要 CUDA Toolkit / cuBLAS，以及 **SM80+** NVIDIA GPU。所有命令从仓库根目录执行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

# RTX 4060 使用 89；其他 SM80+ GPU 请调整架构。
cmake -S . -B build/release \
  -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j 2
export PYTHONPATH="$PWD/build/release/python"
python -m pytest -q tests --require-cuda
```

只有 CPU 时：

```bash
cmake -S . -B build/cpu -DNEEDLE_CUDA=OFF \
  -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu -j 2
export PYTHONPATH="$PWD/build/cpu/python"
python -m pytest -q tests
```

源码与编译模块分离。**修改 Python 源码后也要重新执行 `cmake --build`**，更新构建目录的可导入副本。不要直接使用课程归档中的旧二进制。

```python
import numpy as np
import needle as ndl

x = ndl.Tensor(np.random.randn(128, 256).astype(np.float32), device=ndl.cuda())
layer = ndl.nn.Linear(256, 128, device=ndl.cuda())
with ndl.gemm_mode("tensorcore"):
    y = layer(x)
print(y.shape)  # (128, 128)
```

## 复现实验

```bash
# 正确性 + 三种 Compute Sanitizer 检查；GPU 不可用时明确失败。
python scripts/validate_gpu.py

# 三轮 GEMM（含 cuBLAS 对照）和同权重 TransformerLayer 消融。
python scripts/run_experiments.py

# 从原始 JSON 生成结果表。
python scripts/summarize_results.py --input results/rtx4060 --output results/report.md
```

脚本默认使用 `build/release`，可通过 `--build` 指定其他构建目录。基准记录硬件、软件版本、源码哈希、随机种子、精度、误差与耗时样本；本地新结果写入被忽略的 `results/`。已展示的数据位于 `docs/results/`。

单项实验可运行 `benchmarks/gemm.py` 或 `benchmarks/transformer_ablation.py`，均支持 `--help`。CPU 冒烟用例：

```bash
python benchmarks/gemm.py --device cpu --modes baseline --shapes 17x33x19 --iterations 2
python benchmarks/transformer_ablation.py --device cpu --batch 1 --seq 4 --dim 16 --heads 2 --iterations 2
```

## 项目结构

```text
python/needle/         NDArray、自动微分、模型与 GEMM 策略
src/                   CPU/CUDA 后端与 pybind11 绑定
  cuda/                SIMT / Tensor Core 内核与 cuBLAS benchmark 对照
benchmarks/            GEMM 与 TransformerLayer 消融
scripts/               GPU 验收、实验复现、profile、报告与源码导出
examples/              保留的课程模型示例
tests/                当前项目测试（不依赖课程 grader）
docs/                  架构、实测数据、验证记录与项目说明
archive/               课程 notebook、原始测试及旧实验记录
.github/workflows/     CPU 持续集成
```

原有 `hw4` / `hw4_extra` 已合并为一份源码，去除了发布目录中的编译产物、缓存和训练数据。完整整理前工作目录保留在本机被忽略的 `.local_archive/pre-github-cleanup/`；课程来源与历史基点见 [archive/README.md](archive/README.md)。

## 边界与下一步

主线是 **Transformer 前向优化**。融合 Attention / LayerNorm backward 仍依赖 NumPy；当前注意力不是完整 FlashAttention 论文复现，不宣称高性能 GPU 训练。CPU 上使用 FP32 回退。

Nsight Systems 显示当前优化模型仍有频繁分配/释放，以及占 GPU kernel 时间大头的融合注意力。下一步优先验证缓冲复用和协作分块注意力，而非继续堆叠 GEMM 技巧。计时口径与 profiler 限制见[验证记录](docs/validation.md)。

可执行 `python scripts/export_source.py` 导出干净源码包；导出遵循 `.gitignore`，不会包含环境、备份、数据集或构建产物。
