# 课程与历史实验归档

本目录保留 CMU 10-714 作业的 notebook、原始测试/参考数据以及旧实验记录，用于说明项目演进。它们不参与当前构建与默认 pytest 收集。

- `course/`：HW4 / HW4 Extra notebook（已清除执行输出）、原始测试和参考数组。原始测试依赖当时的目录结构、mugrade、PyTorch 和额外数据，未迁入当前验收流程。
- `experiments/`：原有 benchmark 脚本、日志和笔记。旧脚本可能使用原来的导入路径；保留的是历史记录，不是当前复现入口。

当前运行入口在根目录 README，主源码在 `python/needle` 和 `src`。历史日志中的加速比与新结果不能混用：旧 FP32 GEMM 的 23.1× 不是 Tensor Core 的测量值；旧注意力与 PyTorch MHA 的比较也不具有完全一致的计算范围。

原始版本的 Git 基点：`9bd5e2821be28c189d4ded709e3207d1073678df`。本次整理前的两个完整工作目录另存于本机 `.local_archive/pre-github-cleanup/`（已忽略，不上传），包括原始构建产物与数据集。目录整理没有重写 Git 历史。
