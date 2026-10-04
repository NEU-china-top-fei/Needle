# 开发与验证

从 README 的 CPU 构建开始。修改 Python 源码后也需执行 `cmake --build build/cpu`，使构建目录中的可导入副本更新。请不要把 `.so`、数据集、缓存或本地备份加入 Git。

- Python 与框架接口改动：运行 `python -m pytest -q tests` 和两个 CPU benchmark 冒烟用例。
- CUDA 改动：在 SM80+ GPU 上运行 `python scripts/validate_gpu.py`。该命令要求实际可用的 GPU，并执行 memcheck、racecheck、synccheck。
- 性能改动：运行 `python scripts/run_experiments.py`，保留输入精度、计时范围、硬件信息、源码哈希和重复实验。不要将纯内核时间与包含分配/转换的时间比较。
- 数据报告：原始 JSON 放入 `docs/results/`，用 `scripts/summarize_results.py` 生成表格；默认本地输出 `results/` 不提交。

GitHub Actions 验证 CPU 构建和 CPU 测试。GPU 结果来自单独记录的本地验收，不由 CPU CI 替代。

## 发布到现有 GitHub 仓库

当前配置的远端是 `NEU-china-top-fei/DLsysHW`。本次整理尚未提交或推送；可在检查改动后提交当前工作树：

```bash
git status --short
git diff --stat
git add -A
git commit -m "Organize Needle GPU project and add validated benchmarks"
git push origin main
```

状态中的原 `hw4/` 与 `hw4_extra/` 删除项属于源码合并和目录迁移；对应实现位于 `python/`、`src/`，历史资料位于 `archive/`，完整本地备份位于 `.local_archive/`。`.gitignore` 会排除备份、环境、数据集、构建目录和本地实验输出；`docs/results/` 中的验收记录与实测数据应提交。

仓库 About 可使用：**Educational deep learning framework with CUDA GEMM, Tensor Core pipelining, and reproducible Transformer forward benchmarks.**

建议 Topics：`cuda`、`tensor-cores`、`gemm`、`deep-learning-systems`、`autograd`、`transformer`。CI 在推送后运行；本地通过情况见 `docs/validation.md`。
