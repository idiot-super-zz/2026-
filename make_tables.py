"""从 runs/*/summary.json + benchmark.json 自动生成报告用表格（Markdown / LaTeX）。

报告中所有数字必须由本脚本产出，禁止手抄，以保证"报告数字与代码输出一致"。

用法：
    python make_tables.py                 # 扫描 runs/ 下所有实验
    python make_tables.py --out report_tables.md --latex report_tables.tex
"""

import argparse
import glob
import json
import os

ORDER = ["E0_baseline", "E1_cbam", "E2_dw", "E3_gn", "E4_combo", "E5_bn_bs16", "E6_gn_bs16"]

MODEL_LABEL = {
    "E0_baseline": "ResNet-18 (baseline)",
    "E1_cbam": "ResNet-18 + CBAM",
    "E2_dw": "ResNet-18 + DWConv",
    "E3_gn": "ResNet-18 + GroupNorm",
    "E4_combo": "CBAM + DWConv + GN",
    "E5_bn_bs16": "BN, bs=16",
    "E6_gn_bs16": "GN, bs=16",
    "F0_baseline": "ResNet-18 (224 baseline)",
    "F1_cbam": "+ CBAM (224)",
    "F2_dw": "+ DWConv (224)",
    "F3_gn": "+ GroupNorm (224)",
    "F4_combo": "CBAM + DWConv + GN (224)",
}


def load_runs(runs_dir: str):
    rows = []
    for summary_path in sorted(glob.glob(os.path.join(runs_dir, "*", "summary.json"))):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        run_dir = os.path.dirname(summary_path)
        bench_path = os.path.join(run_dir, "benchmark.json")
        if os.path.exists(bench_path):
            with open(bench_path, "r", encoding="utf-8") as f:
                summary.update(json.load(f))
        rows.append(summary)

    def sort_key(r):
        name = r.get("exp_name", "")
        return (ORDER.index(name) if name in ORDER else len(ORDER), name)

    return sorted(rows, key=sort_key)


def fmt(v, digits=2, dash="—"):
    return dash if v is None else f"{v:.{digits}f}"


def to_markdown(rows):
    header = ("| 实验 | 模型 | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) | "
              "训练时长 (min) | s/epoch | 显存 (MB) |")
    sep = "|" + "---|" * 9
    lines = [header, sep]
    best = max((r.get("best_top1", 0) for r in rows), default=0)
    for r in rows:
        name = r.get("exp_name", "?")
        label = MODEL_LABEL.get(name, name)
        top1 = r.get("best_top1")
        top1_str = fmt(top1) + (" **" if top1 == best and len(rows) > 1 else "")
        lines.append(
            f"| {name} | {label} | {fmt(r.get('params_m'), 2)} | {fmt(r.get('flops_g'), 2)} | "
            f"{top1_str} | {fmt(r.get('final_top5'))} | {fmt(r.get('train_time_min'), 1)} | "
            f"{fmt(r.get('sec_per_epoch'), 1)} | {fmt(r.get('peak_gpu_mem_mb'), 0)} |"
        )
    return "\n".join(lines)


def to_latex(rows):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CIFAR-100 上 ResNet-18 结构改造的精度与效率对比}",
        r"\label{tab:main}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"实验 & 模型 & Params (M) & FLOPs (G) & Top-1 (\%) & 训练时长 (min) & s/epoch \\",
        r"\midrule",
    ]
    for r in rows:
        name = r.get("exp_name", "?")
        label = MODEL_LABEL.get(name, name).replace("_", r"\_").replace("+", r"$+$")
        lines.append(
            f"{name.replace('_', chr(92) + '_')} & {label} & {fmt(r.get('params_m'), 2)} & "
            f"{fmt(r.get('flops_g'), 2)} & {fmt(r.get('best_top1'))} & "
            f"{fmt(r.get('train_time_min'), 1)} & {fmt(r.get('sec_per_epoch'), 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser("自动生成结果表格")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--out", type=str, default="report_tables.md")
    p.add_argument("--latex", type=str, default=None)
    args = p.parse_args()

    rows = load_runs(args.runs_dir)
    if not rows:
        print(f"{args.runs_dir}/ 下没有找到 summary.json，请先运行训练。")
        return

    md = to_markdown(rows)
    print(md)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# 自动生成结果表\n\n")
        f.write(md + "\n")
    print(f"\n已写入 {args.out}")

    if args.latex:
        with open(args.latex, "w", encoding="utf-8") as f:
            f.write(to_latex(rows) + "\n")
        print(f"已写入 {args.latex}")


if __name__ == "__main__":
    main()
