# 验证记录

日期：2026-10-04。以下为本机执行结果；GitHub Actions 配置已加入，但远端工作流尚未运行。

## 环境

- GPU：NVIDIA GeForce RTX 4060 Laptop，8 GB，SM89。
- NVIDIA driver：595.91.07；CUDA Toolkit / nvcc：13.2.86；CUDA runtime：13020；cuBLAS：130401。
- GCC 13.3.0；Python 3.12.3；NumPy 2.5.3；pybind11 3.1.0；pytest 9.1.1。
- Nsight Systems 2025.6.3；Nsight Compute 2026.1.1。
- Release 构建，显式 `CMAKE_CUDA_ARCHITECTURES=89`。GPU 顺序测量，未锁频/修改功耗设置。

## 正确性与内存检查

| 检查 | 结果 | 证据 |
|---|---|---|
| GPU 项目测试 | **110 passed，1 skipped** | [pytest.txt](results/validation/pytest.txt) |
| Compute Sanitizer memcheck | **55 CUDA tests passed，0 errors** | [memcheck.txt](results/validation/memcheck.txt) |
| Compute Sanitizer racecheck | **55 CUDA tests passed，0 hazards** | [racecheck.txt](results/validation/racecheck.txt) |
| Compute Sanitizer synccheck | **55 CUDA tests passed，0 errors** | [synccheck.txt](results/validation/synccheck.txt) |
| 独立 CPU 构建与测试 | **55 passed，56 skipped** | CUDA 用例和 GPU 专用 PreparedGemm 在 CPU 环境跳过 |
| CPU GEMM / Transformer 冒烟 | 通过 | 无 GPU、无需原作业 build 目录 |
| 导出的独立源码包 | 构建通过；55 passed / 56 skipped | 解压到全新临时目录，CPU 编译、测试与 examples 导入通过 |
| 三轮 GPU 性能与数值检查 | 全部通过 | [原始 JSON 与统计](performance.md) |

GPU 测试唯一跳过项是 CPU 参数化分支中的 `PreparedGemm`，不是跳过 CUDA。`--require-cuda` 保证正式 GPU 验收不能以“设备不可用、全部跳过”通过。

覆盖内容：非对齐尺寸、转置和切片输入、零矩阵/单位矩阵、FP32 梯度、Linear 集成、同步/异步 Tensor Core、cuBLAS FP16/FP32 参考、不同维度的 LayerNorm、causal/non-causal 注意力。

归档中的完整课程测试依赖原始目录布局、mugrade、PyTorch 和额外数据，不属于本次验收。GitHub CPU CI 不替代本地 GPU 验收。

## Nsight Systems：优化后模型的剩余开销

对 B=8/S=64/D=256/H=8 的优化 TransformerLayer 预热 5 次后，使用 `cudaProfilerStart/Stop` 只采集随后 10 次前向。

| 分类 | 观测 | 口径 |
|---|---|---|
| CUDA API | 400 次 cudaMalloc、400 次 cudaFree | 每次前向各 40 次 |
| CUDA API 时间 | cudaFree 59.0%，cudaMalloc 29.4% | CUDA API 记录时间占比，**不是总 wall time 占比** |
| GPU kernel 时间 | 融合注意力 63.9%，Compact 12.4%，TC GEMM 10.4% | 各内核总持续时间占比 |

`cudaFree` 时间可以包含等待 GPU 的隐式同步，不能把上述比例全部解释为分配器本身的纯 CPU 开销。根据这些记录，下一步值得验证缓冲复用和分块协作注意力；这属于待检验的优化方向。

原始汇总：[CUDA API](results/profile/summary_cuda_api_sum.csv)、[GPU kernels](results/profile/summary_cuda_gpu_kern_sum.csv)。本机完整时间线在被忽略的 `results/profile/transformer.nsys-rep`。

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=results/profile/transformer python scripts/profile_transformer.py
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format csv \
  --output results/profile/summary results/profile/transformer.nsys-rep
```

Nsight Compute 的单次内核采集返回 `ERR_NVGPUCTRPERM`，当前用户没有硬件计数器权限。没有修改驱动安全设置，也没有据此编造 occupancy、bank conflict 或 Tensor Core 利用率数据。[诊断记录](results/profile/ncu-permission.txt)

## 复现与来源

运行 `python scripts/validate_gpu.py` 复现 GPU 验收，运行 `python scripts/run_experiments.py` 复现实验。已发布 JSON 只把输出文件路径改为仓库相对路径，数值和统计样本未变。

实验时 Git 基点为 `9bd5e2821be28c189d4ded709e3207d1073678df`，工作区尚未提交；各 JSON 额外记录实际 `src` / `python` / `benchmarks` 源码内容的 SHA-256。结果报告以该源码哈希而非旧 commit 单独标识实验版本。
