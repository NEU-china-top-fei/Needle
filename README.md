# Needle GPU

基于 [CMU 10-714](https://dlsyscourse.org/) Needle 的深度学习框架，支持 CPU / CUDA 张量计算、自动微分和模型训练，并实现 SIMT、WMMA、`cp.async` 双缓冲 GEMM 与 Transformer 前向算子融合。

[架构与实现](docs/architecture.md) · [性能实验](docs/performance.md) · [来源说明](NOTICE.md)

## 构建

需要 Python 3.10+、CMake 3.24+ 和 C++ 编译器。CUDA 后端需要 CUDA Toolkit / cuBLAS 和 SM80+ GPU。以下命令在仓库根目录执行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

# RTX 4060 使用 89；其他 GPU 请设置对应架构。
cmake -S . -B build/release \
  -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j 2
export PYTHONPATH="$PWD/build/release/python"
```

仅使用 CPU 时，将配置命令中的 `-DCMAKE_CUDA_ARCHITECTURES=89` 换为 `-DNEEDLE_CUDA=OFF`，并使用新的构建目录（例如 `build/cpu`）。后续构建命令及 `PYTHONPATH` 也使用该目录。修改 Python 源码后需重新执行 `cmake --build`，更新构建目录中的包。

## 训练示例

两个示例均使用 Needle 的模型层、交叉熵、自动微分和优化器，支持 `--device cpu` / `--device cuda`；默认使用 FP32。

### MNIST 分类

[train_mnist.py](examples/train_mnist.py)：`784 → 128 → 10` MLP，ReLU 激活，SGD 更新。首次运行下载并校验 MNIST，数据保存在本地 `data/mnist/`。

```bash
python examples/train_mnist.py --device cuda --download --epochs 5
```

默认配置在 RTX 4060 Laptop 上训练 5 个 epoch，测试集准确率 **95.83%**（60,000 张训练图像，10,000 张测试图像，seed=0）。

### Transformer 字符语言模型

[train_transformer.py](examples/train_transformer.py)：字符嵌入、位置嵌入、因果多头注意力、FFN 和输出投影，使用 Adam 训练下一个字符预测，最后进行自回归生成。默认使用仓库内的小语料，无需下载。

```bash
python examples/train_transformer.py --device cuda --epochs 10

# 换用自己的 UTF-8 文本；prompt 中的字符需要出现在语料中。
python examples/train_transformer.py --device cuda \
  --text data/corpus.txt --prompt "the " --sequence-length 32 --dim 64 --heads 4
```

默认模型为 1 层、2 个注意力头、隐藏维度 32、上下文长度 16。语料按顺序划分为 90% 训练、10% 验证；CUDA 上 10 个 epoch 后验证 loss 从 **4.9150 降至 2.4999**。小语料用于演示训练流程，生成质量有限。训练使用可微的 Needle Attention / LayerNorm 基础算子；融合内核用于下述前向性能实验。

## CUDA 优化

| GEMM 模式 | 精度 | 实现 |
|---|---|---|
| `baseline`（默认） | FP32 | 原始课程内核 |
| `simt` | FP32 | 分块、寄存器复用、float4 访存、双缓冲 |
| `wmma` | FP16 输入 / FP32 累加输出 | WMMA，同步加载 |
| `tensorcore` | FP16 输入 / FP32 累加输出 | WMMA + cp.async 双缓冲 |

通过 `with needle.gemm_mode("tensorcore"):` 切换 `@`、Linear 和 Transformer 投影的 GEMM 路径。TC 路径在 GPU 上转换并补齐输入，计算后裁剪输出；CPU 使用 FP32 回退。

RTX 4060 Laptop 8 GB、CUDA 13.2，三轮独立实验的中位数：

| 工作负载 | 基线 → 优化 | 加速 |
|---|---|---:|
| 2048³ GEMM，包含转换与分配 | FP32 SIMT 12.6145 ms → TC 2.6691 ms | 4.73× |
| 2048³ GEMM，预打包内核 | WMMA 1.2861 ms → 双缓冲 1.1933 ms | 1.08× |
| TransformerLayer，B=8 / S=64 / D=256 / H=8 | 原始前向 22.4074 ms → TC + Attention + LayerNorm 融合 2.8092 ms | 7.98× |

第一项包含输入精度变化；Transformer 前向相对 L2 误差为 5.62e−4。预打包 TC 内核为 **14.40 TFLOPS**，同口径 cuBLAS 为 0.6110 ms，自写内核仍慢约 1.95×。完整计时与精度口径见[性能实验](docs/performance.md)。

```bash
python -m pytest -q tests --require-cuda  
python scripts/run_experiments.py
python scripts/summarize_results.py
```



## 目录

```text
python/needle/    张量、自动微分、模型层、优化器和数据加载
src/             C++ / CUDA 后端与 pybind11 绑定
examples/        MNIST 与 Transformer 训练示例
benchmarks/      GEMM 对照与 Transformer 前向消融
scripts/         实验、性能分析与结果汇总
tests/           算子、梯度和训练测试
docs/            架构与性能实验
```
