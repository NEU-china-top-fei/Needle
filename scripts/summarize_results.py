"""Generate Markdown tables from repeated, checked benchmark JSON files."""
import argparse
import json
from pathlib import Path
from statistics import median


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/rtx4060")
    p.add_argument("--output", default="results/report.md")
    args = p.parse_args()
    source, target = Path(args.input), Path(args.output)
    gemm = [json.loads(f.read_text()) for f in sorted(source.glob("gemm_*.json"))]
    if not gemm:
        p.error("No gemm_*.json input reports")
    reports = [json.loads(f.read_text()) for f in source.glob("*.json")]
    hashes = {r["metadata"]["source_sha256"] for r in reports}
    if len(hashes) != 1 or not all(row["correctness_passed"] for r in reports for row in r["results"]):
        p.error("Reports must share one source hash and pass correctness checks")
    meta = gemm[0]["metadata"]
    lines = ["# RTX 4060 Laptop 实测报告", "",
        f"设备：{meta['device']['name']}；CUDA runtime `{meta['device']['runtime']}`，"
        f"cuBLAS `{meta['device']['cublas']}`；Python {meta['python']}。",
        "",
        f"每组 {len(gemm)} 次独立进程重复，表中为各轮统计量的中位数，范围为跨轮最小—最大值。"
        "GEMM 每轮预热 10 次、计时 50 次；Transformer 每轮预热 5 次、计时 30 次。"
        "未锁定 GPU 时钟/功耗，均为同一设备的顺序实验。", "",
        "## GEMM：包含转换与分配的框架调用", "",
        "baseline/SIMT 使用 FP32 输入；TC 使用 FP16 输入、FP32 累加/输出。跨精度加速必须结合误差理解。", "",
        "| M×K×N | baseline ms | SIMT ms | TC ms（跨轮范围） | baseline / TC | SIMT / TC | TC 相对 FP32 L2 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    shapes = [tuple(r["shape_mkn"]) for r in gemm[0]["results"] if r["mode"] == "baseline"]
    def rows(shape, mode):
        return [r for report in gemm for r in report["results"]
                if tuple(r["shape_mkn"]) == shape and r["mode"] == mode]
    for shape in shapes:
        b = median(r["framework"]["median_ms"] for r in rows(shape, "baseline"))
        s = median(r["framework"]["median_ms"] for r in rows(shape, "simt"))
        times = [r["framework"]["median_ms"] for r in rows(shape, "tensorcore")]
        tc = median(times)
        err = max(r["error_vs_fp32_inputs"]["relative_l2"] for r in rows(shape, "tensorcore"))
        lines.append(f"| {'×'.join(map(str, shape))} | {b:.4f} | {s:.4f} | {tc:.4f} ({min(times):.4f}–{max(times):.4f}) | {b/tc:.2f}× | {s/tc:.2f}× | {err:.3e} |")
    lines += ["", "127×255×129 的 SIMT 模式回退到原始 FP32 内核。", "",
        "## GEMM：预打包后的内核时间", "",
        "三条路径均使用相同补齐形状和 FP16 输入 / FP32 累加输出，CUDA-event 批次平均计时。"
        "cuBLAS 不包含 handle 创建；各项均排除分配、转换、补零和裁剪。", "",
        "| M×K×N | WMMA ms | pipeline ms | cuBLAS ms | pipeline 有效 TFLOPS | WMMA / pipeline | pipeline / cuBLAS 延迟 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for shape in shapes:
        w = median(r["prepared_kernel"]["mean_ms"] for r in rows(shape, "wmma"))
        tc = median(r["prepared_kernel"]["mean_ms"] for r in rows(shape, "tensorcore"))
        lib = median(r["mean_ms"] for report in gemm for r in report["cublas_prepared_reference"]
                     if tuple(r["shape_mkn"]) == shape)
        flops = 2 * shape[0] * shape[1] * shape[2] / (tc * 1e9)
        lines.append(f"| {'×'.join(map(str, shape))} | {w:.5f} | {tc:.5f} | {lib:.5f} | {flops:.2f} | {w/tc:.2f}× | {tc/lib:.2f}× |")
    lines += ["", "pipeline / cuBLAS 大于 1 表示自写内核更慢。小矩阵计时可能受 launch 间隙影响；"
              "pipeline 相对同步 WMMA 的收益并不在所有形状下稳定出现。", "", "## TransformerLayer：同权重前向消融", "",
              "dropout=0，eval，FFN hidden=4D；每轮复用同一模型和输入。模型计时包含 Python 调度和计算图构造。", ""]
    for name in ("small", "medium"):
        runs = [json.loads(f.read_text()) for f in sorted(source.glob(f"transformer_{name}_*.json"))]
        config = runs[0]["metadata"]["arguments"]
        lines += [f"### {name}: B={config['batch']}, S={config['seq']}, D={config['dim']}, H={config['heads']}", "",
            "| 阶段 | median ms（跨轮范围） | 相对 baseline 加速 | 最大相对 L2 误差 |",
            "|---|---:|---:|---:|"]
        baseline = median(r["results"][0]["framework"]["median_ms"] for r in runs)
        for item in runs[0]["results"]:
            stage = item["stage"]
            stage_rows = [r for run in runs for r in run["results"] if r["stage"] == stage]
            times = [r["framework"]["median_ms"] for r in stage_rows]
            err = max(r["error_vs_baseline"]["relative_l2"] for r in stage_rows)
            lines.append(f"| {stage} | {median(times):.4f} ({min(times):.4f}–{max(times):.4f}) | {baseline/median(times):.2f}× | {err:.3e} |")
        lines.append("")
    lines += ["## 如何解读", "",
        "- 2048³ GEMM 的 TC 框架路径相对 SIMT 的提升包含 FP16 输入量化；不能描述为同精度 FP32 加速。",
        "- TC 预打包内核到框架调用之间有明显开销，转换/分配及输出整理值得继续分析。",
        "- Transformer 的主要额外收益来自注意力与 LayerNorm 融合；仅替换 GEMM 并不能获得最终模型加速比。",
        "- 在中大矩阵用例中，自写 TC 内核仍落后于同口径 cuBLAS；小矩阵的微秒级波动不能据此宣称稳定胜过库实现。", ""]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n")
    print(target)


if __name__ == "__main__":
    main()
