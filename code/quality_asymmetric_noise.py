#!/usr/bin/env python3
import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import validate_rl_controller as vrl
import simulate_rl_controller_trace as base


def biased_step_eval(request, action, current_model, latency_ms, spec, rng, false_negative, false_positive):
    outcome = vrl.step_eval(request, action, current_model, latency_ms, spec)
    true_quality_ok = bool(outcome["quality_sla_met"])
    observed_quality_ok = true_quality_ok
    if true_quality_ok and rng.random() < false_negative:
        observed_quality_ok = False
    elif (not true_quality_ok) and rng.random() < false_positive:
        observed_quality_ok = True
    observed_joint = observed_quality_ok and bool(outcome["latency_sla_met"])
    observed_reward = (
        int(observed_joint)
        - spec.beta_latency * min(2.0, outcome["total_latency_ms"] / request["deadline_ms"])
        - spec.beta_switch * int(outcome["switch"])
    )
    outcome["observed_quality_sla_met"] = int(observed_quality_ok)
    outcome["observed_joint_sla_met"] = int(observed_joint)
    outcome["rl_reward"] = observed_reward
    return outcome


def train_biased_q(latency_mean, latency_samples, episodes, steps, seed, spec, false_negative, false_positive):
    rng = random.Random(seed)
    q = defaultdict(float)
    for episode in range(episodes):
        trace = vrl.make_trace(steps, seed + episode)
        current_model = "Qwen2.5-0.5B Q4"
        epsilon = 0.04 + (0.35 - 0.04) * np.exp(-episode / max(1.0, episodes / 4.0))
        for idx, request in enumerate(trace):
            state = vrl.state_key(request, current_model)
            feasible = vrl.feasible_actions(request)
            action = rng.choice(feasible) if rng.random() < epsilon else max(feasible, key=lambda a: q[(state, a)])
            latency_ms = base.draw_latency(latency_samples, latency_mean, action, rng)
            outcome = biased_step_eval(
                request, action, current_model, latency_ms, spec, rng, false_negative, false_positive
            )
            next_model = action[0]
            if idx + 1 < len(trace):
                next_state = vrl.state_key(trace[idx + 1], next_model)
                next_actions = vrl.feasible_actions(trace[idx + 1])
                target = outcome["rl_reward"] + 0.20 * max(q[(next_state, a)] for a in next_actions)
            else:
                target = outcome["rl_reward"]
            q[(state, action)] += 0.18 * (target - q[(state, action)])
            current_model = next_model
    return q


def evaluate_q(trace, latency_mean, latency_samples, q_table, seed, spec, scenario):
    rng = random.Random(seed)
    current_model = "Qwen2.5-0.5B Q4"
    rows = []
    for request in trace:
        state = vrl.state_key(request, current_model)
        action = max(vrl.feasible_actions(request), key=lambda a: q_table[(state, a)])
        latency_ms = base.draw_latency(latency_samples, latency_mean, action, rng)
        outcome = vrl.step_eval(request, action, current_model, latency_ms, spec)
        rows.append({**request, **outcome, "scenario": scenario, "strategy": "q_learning"})
        current_model = action[0]
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    seed_summary = (
        df.groupby(["scenario", "seed"], as_index=False)
        .agg(
            requests=("request_id", "count"),
            joint_sla_rate=("joint_sla_met", "mean"),
            quality_sla_rate=("quality_sla_met", "mean"),
            latency_sla_rate=("latency_sla_met", "mean"),
            mean_latency_ms=("total_latency_ms", "mean"),
            mean_energy_j=("infer_energy_j", "mean"),
            q8_action_rate=("model", lambda x: float((x == "Qwen3-0.6B Q8").mean())),
            reward=("rl_reward", "mean"),
        )
    )
    parts = []
    for metric in [
        "joint_sla_rate",
        "quality_sla_rate",
        "latency_sla_rate",
        "mean_latency_ms",
        "mean_energy_j",
        "q8_action_rate",
        "reward",
    ]:
        part = seed_summary.groupby("scenario")[metric].agg(["mean", "std"]).reset_index()
        part = part.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        parts.append(part)
    agg = parts[0]
    for part in parts[1:]:
        agg = agg.merge(part, on="scenario")
    return seed_summary, agg


def plot(agg, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), constrained_layout=True)
    x = np.arange(len(agg))
    labels = agg["scenario"].tolist()
    panels = [
        ("joint_sla_rate_mean", "joint_sla_rate_std", "True joint SLA"),
        ("mean_latency_ms_mean", "mean_latency_ms_std", "Latency (ms)"),
        ("q8_action_rate_mean", "q8_action_rate_std", "Q8 action rate"),
    ]
    for ax, (mean_col, std_col, title) in zip(axes, panels):
        ax.bar(x, agg[mean_col], yerr=agg[std_col].fillna(0), capsize=3, color="#4C78A8")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 1.05)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_md(out_md, agg):
    lines = [
        "# Asymmetric quality-feedback bias robustness",
        "",
        "| Scenario | True joint SLA | Quality SLA | Latency(ms) | Energy(J) | Q8 action rate | Reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in agg.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['joint_sla_rate_mean']:.3f}±{row['joint_sla_rate_std']:.3f} | "
            f"{row['quality_sla_rate_mean']:.3f}±{row['quality_sla_rate_std']:.3f} | "
            f"{row['mean_latency_ms_mean']:.1f}±{row['mean_latency_ms_std']:.1f} | "
            f"{row['mean_energy_j_mean']:.2f}±{row['mean_energy_j_std']:.2f} | "
            f"{row['q8_action_rate_mean']:.3f}±{row['q8_action_rate_std']:.3f} | "
            f"{row['reward_mean']:.3f}±{row['reward_std']:.3f} |"
        )
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--raw-q4", required=True)
    parser.add_argument("--raw-q8", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    scenarios = [
        ("clean", 0.0, 0.0),
        ("symmetric_15", 0.15, 0.15),
        ("false_negative_15", 0.15, 0.0),
        ("false_positive_15", 0.0, 0.15),
        ("false_positive_30", 0.0, 0.30),
    ]
    spec = vrl.RewardSpec("asymmetric_quality_noise", beta_budget=0.0, beta_latency=0.45, beta_switch=0.05)
    latency_mean = base.load_latency(args.core)
    latency_samples = base.load_latency_samples(args.raw_q4, args.raw_q8)
    rows = []
    for scenario, fn_rate, fp_rate in scenarios:
        for seed in seeds:
            q_table = train_biased_q(
                latency_mean,
                latency_samples,
                args.episodes,
                args.train_steps,
                seed + int(fn_rate * 10000) + int(fp_rate * 20000),
                spec,
                fn_rate,
                fp_rate,
            )
            trace = vrl.make_trace(args.eval_steps, seed + 140000)
            eval_rows = evaluate_q(trace, latency_mean, latency_samples, q_table, seed + 141000, spec, scenario)
            for row in eval_rows:
                row["seed"] = seed
                row["false_negative_rate"] = fn_rate
                row["false_positive_rate"] = fp_rate
            rows.extend(eval_rows)
    frame = pd.DataFrame(rows)
    seed_summary, agg = summarize(rows)
    frame.to_csv(out_dir / "rl_quality_asymmetric_noise_requests_20260514.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(out_dir / "rl_quality_asymmetric_noise_seed_summary_20260514.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(out_dir / "rl_quality_asymmetric_noise_summary_20260514.csv", index=False, encoding="utf-8-sig")
    plot(agg, out_dir / "rl_quality_asymmetric_noise_20260514.png")
    write_md(out_dir / "rl_quality_asymmetric_noise_20260514.md", agg)
    print(out_dir / "rl_quality_asymmetric_noise_20260514.md")


if __name__ == "__main__":
    main()
