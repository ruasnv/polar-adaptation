#!/usr/bin/env python3
"""
analysis/table_syntax_semantics.py

Isolates the performance dissociation between syntactic acceptability (CoLA)
and semantic tasks, saving both a JSON cache and a clean LaTeX block.
"""
import json
import os
from pathlib import Path

TASK_PRIMARY = {
    "cola": "matthews_correlation", "mnli": "accuracy", "mrpc": "f1",
    "qnli": "accuracy", "rte": "accuracy", "sst2": "accuracy", "stsb": "pearson"
}


def main():
    root = Path("results/glue")
    if not root.exists():
        print("Error: Results target directory path missing.")
        return

    print("\nTable 3: Linguistic Task Split (Performance Paradigm)")
    print("─" * 70)
    print(f"{'GLUE Task':<12} | {'Task Metric':<22} | {'hybrid_paft':<12} | {'safe_hybrid':<12}")
    print("─" * 70)

    latex_lines = []
    json_data = {}

    for task, metric in TASK_PRIMARY.items():
        h_path = root / task / "hybrid_paft" / "metrics.json"
        s_path = root / task / "safe_hybrid_paft" / "metrics.json"

        val_h = 0.0
        val_s = 0.0

        if h_path.exists():
            with open(h_path) as f:
                val_h = json.load(f).get(metric, 0.0)
        if s_path.exists():
            with open(s_path) as f:
                val_s = json.load(f).get(metric, 0.0)

        task_type = "Syntax" if task == "cola" else "Semantic"
        print(f"{task:<12} ({task_type[0]}) | {metric:<22} | {val_h:<12.4f} | {val_s:<12.4f}")

        # Build clean rows matching LaTeX tabular layout expectations
        latex_lines.append(f"{task:<12} & {task_type:<10} & {val_h:<10.4f} & {val_s:<10.4f} \\\\")

        json_data[task] = {"type": task_type, "hybrid_paft": val_h, "safe_hybrid_paft": val_s}

    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the JSON cache
    with open(out_dir / "table_syntax_semantics.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # Save the missing .tex formatting file for LaTeX compilation
    with open(out_dir / "table_syntax_semantics.tex", "w") as f:
        f.write("\n".join(latex_lines))

    print("─" * 70)
    print("Successfully generated results/analysis/table_syntax_semantics.tex")


if __name__ == "__main__":
    main()