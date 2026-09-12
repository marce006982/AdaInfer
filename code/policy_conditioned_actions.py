#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize_group(df, group_col, label):
    rows = []
    for value, part in df.groupby(group_col, dropna=False):
        is_q8 = part["model"].str.contains("Q8")
        rows.append(
            {
                "condition": label,
                "value": value,
                "n": len(part),
                "q8_action_rate": is_q8.mean(),
                "q4_action_rate": 1.0 - is_q8.mean(),
                "mean_selected_budget": part["selected_budget"].mean(),
                "budget_retention": part["budget_retention"].mean(),
                "joint_sla_rate": part["joint_sla_met"].mean(),
                "mean_latency_ms": part["total_latency_ms"].mean(),
            }
        )
    return rows


def write_markdown(summary, out_md):
    lines = [
        "# RL policy conditioned action distribution",
        "",
        "The table is computed from `rl_validation_requests_20260514.csv` for the `q_learning` controller.",
        "It reports how the learned policy changes model family and budget choices under different request and network states.",
        "",
        "| Condition | Value | n | Q8 action rate | Mean budget | Retention | Joint SLA | Latency(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['condition']} | {row['value']} | {int(row['n'])} | "
            f"{row['q8_action_rate']:.3f} | {row['mean_selected_budget']:.1f} | "
            f"{row['budget_retention']:.3f} | {row['joint_sla_rate']:.3f} | {row['mean_latency_ms']:.1f} |"
        )
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(summary, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("quality_sla", "Q8 rate by quality SLA"),
        ("requested_budget", "Q8 rate by requested budget"),
        ("burst", "Q8 rate by burst state"),
        ("scenario", "Q8 rate by stress scenario"),
    ]
    for ax, (condition, title) in zip(axes.flatten(), panels):
        part = summary[summary["condition"] == condition].copy()
        part["value"] = part["value"].astype(str)
        ax.bar(part["value"], part["q8_action_rate"], color="#4C78A8")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Q8 action rate")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-png", required=True)
    args = parser.parse_args()

    usecols = [
        "strategy",
        "scenario",
        "q_sla",
        "requested_budget",
        "burst",
        "rtt_scale",
        "loss_rate",
        "model",
        "selected_budget",
        "budget_retention",
        "joint_sla_met",
        "total_latency_ms",
    ]
    df = pd.read_csv(args.requests, usecols=usecols)
    q = df[df["strategy"] == "q_learning"].copy()
    q["quality_sla"] = q["q_sla"].map(lambda x: f"{x:.2f}")
    q["requested_budget"] = q["requested_budget"].astype(int)
    q["burst"] = q["burst"].astype(int)
    q["rtt_scale"] = q["rtt_scale"].map(lambda x: f"{x:.1f}")
    q["loss_rate"] = q["loss_rate"].map(lambda x: f"{x:.2f}")

    rows = []
    rows.extend(summarize_group(q, "quality_sla", "quality_sla"))
    rows.extend(summarize_group(q, "requested_budget", "requested_budget"))
    rows.extend(summarize_group(q, "burst", "burst"))
    rows.extend(summarize_group(q, "rtt_scale", "rtt_scale"))
    rows.extend(summarize_group(q, "loss_rate", "loss_rate"))
    rows.extend(summarize_group(q, "scenario", "scenario"))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    write_markdown(summary, args.out_md)
    plot(summary, args.out_png)
    print(args.out_csv)
    print(args.out_png)


if __name__ == "__main__":
    main()
