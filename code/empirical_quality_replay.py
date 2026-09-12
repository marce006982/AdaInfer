#!/usr/bin/env python3
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import bootstrap_statistical_tests as boot
import simulate_rl_controller_trace as base
import validate_rl_controller as vrl


MODEL_ALIASES = {
    "qwen2.5-0.5b-q4": "Qwen2.5-0.5B Q4",
    "qwen3-0.6b-q8": "Qwen3-0.6B Q8",
}


def canonical_model(value):
    low = str(value).lower()
    for key, name in MODEL_ALIASES.items():
        if key in low:
            return name
    raise ValueError(f"Unknown model name: {value}")


def load_quality_samples(paths):
    samples = {model: [] for model in base.QUALITY}
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "field_score" in frame.columns:
            score_col = "field_score"
        elif "correct" in frame.columns:
            score_col = "correct"
        else:
            raise ValueError(f"No quality score column in {path}")
        for _, row in frame.iterrows():
            samples[canonical_model(row["model"])].append(float(row[score_col]))
    missing = [name for name, values in samples.items() if not values]
    if missing:
        raise ValueError(f"Missing quality samples for: {missing}")
    return samples


def draw_quality(samples, model, rng):
    values = samples[model]
    return float(values[rng.randrange(len(values))])


def step_eval_empirical(request, action, current_model, latency_ms, quality_samples, rng, spec):
    model, budget = action
    switch = int(model != current_model)
    total = request["wireless_ms"] + request["queue_ms"] + latency_ms + switch * base.SWITCH_COST[model]
    quality = draw_quality(quality_samples, model, rng)
    quality_ok = quality >= request["q_sla"]
    latency_ok = total <= request["deadline_ms"]
    retention = budget / request["requested_budget"]
    sla_term = int(quality_ok and latency_ok)
    reward = (
        sla_term
        + spec.beta_budget * retention
        - spec.beta_latency * min(2.0, total / request["deadline_ms"])
        - spec.beta_switch * switch
    )
    power_w = 5.293 if "Q4" in model else 5.260
    return {
        "model": model,
        "selected_budget": budget,
        "switch": switch,
        "infer_ms": latency_ms,
        "total_latency_ms": total,
        "infer_energy_j": power_w * (latency_ms + switch * base.SWITCH_COST[model]) / 1000.0,
        "quality": quality,
        "budget_retention": retention,
        "quality_sla_met": int(quality_ok),
        "latency_sla_met": int(latency_ok),
        "joint_sla_met": int(quality_ok and latency_ok),
        "rl_reward": reward,
    }


def choose_expected_sla(request, current_model, latency_mean, mean_quality):
    feasible = vrl.feasible_actions(request)
    quality_feasible = [a for a in feasible if mean_quality[a[0]] >= request["q_sla"]]
    pool = quality_feasible or feasible
    deadline_feasible = [
        a for a in pool if vrl.expected_total(request, a, current_model, latency_mean) <= request["deadline_ms"]
    ]
    pool = deadline_feasible or pool
    return min(pool, key=lambda a: vrl.expected_total(request, a, current_model, latency_mean))


def train_q_empirical(latency_mean, latency_samples, quality_samples, episodes, steps, seed, spec, alpha, gamma, eps_start, eps_end):
    rng = random.Random(seed)
    q = defaultdict(float)
    for episode in range(episodes):
        trace = vrl.make_trace(steps, seed + episode)
        current_model = "Qwen2.5-0.5B Q4"
        epsilon = eps_end + (eps_start - eps_end) * np.exp(-episode / max(1.0, episodes / 4.0))
        for idx, request in enumerate(trace):
            state = vrl.state_key(request, current_model)
            feasible = vrl.feasible_actions(request)
            action = rng.choice(feasible) if rng.random() < epsilon else max(feasible, key=lambda a: q[(state, a)])
            latency_ms = base.draw_latency(latency_samples, latency_mean, action, rng)
            outcome = step_eval_empirical(request, action, current_model, latency_ms, quality_samples, rng, spec)
            next_model = action[0]
            if idx + 1 < len(trace):
                next_state = vrl.state_key(trace[idx + 1], next_model)
                next_actions = vrl.feasible_actions(trace[idx + 1])
                target = outcome["rl_reward"] + gamma * max(q[(next_state, a)] for a in next_actions)
            else:
                target = outcome["rl_reward"]
            q[(state, action)] += alpha * (target - q[(state, action)])
            current_model = next_model
    return q


def evaluate(strategy, trace, latency_mean, latency_samples, quality_samples, seed, q_table, spec):
    rng = random.Random(seed)
    rows = []
    current_model = "Qwen2.5-0.5B Q4"
    mean_quality = {model: float(np.mean(values)) for model, values in quality_samples.items()}
    for request in trace:
        if strategy == "q_learning_empirical":
            state = vrl.state_key(request, current_model)
            action = max(vrl.feasible_actions(request), key=lambda a: q_table[(state, a)])
        elif strategy == "greedy_sla":
            action = choose_expected_sla(request, current_model, latency_mean, mean_quality)
        elif strategy == "dynamic_budget_q8":
            budget = base.choose_budget(
                latency_mean,
                "Qwen3-0.6B Q8",
                request["requested_budget"],
                request["wireless_ms"],
                request["queue_ms"],
                request["deadline_ms"],
            )
            action = ("Qwen3-0.6B Q8", budget)
        elif strategy == "static_q4":
            action = ("Qwen2.5-0.5B Q4", request["requested_budget"])
        elif strategy == "static_q8":
            action = ("Qwen3-0.6B Q8", request["requested_budget"])
        else:
            raise ValueError(strategy)
        latency_ms = base.draw_latency(latency_samples, latency_mean, action, rng)
        outcome = step_eval_empirical(request, action, current_model, latency_ms, quality_samples, rng, spec)
        rows.append({**request, **outcome, "strategy": strategy})
        current_model = action[0]
    return rows


def aggregate(summary):
    metrics = [
        "mean_total_latency_ms",
        "mean_infer_energy_j",
        "quality_sla_rate",
        "latency_sla_rate",
        "joint_sla_rate",
        "mean_budget_retention",
        "switch_rate_per_100",
        "mean_rl_reward",
    ]
    grouped = summary.groupby(["scenario", "strategy"])
    pieces = []
    for metric in metrics:
        part = grouped[metric].agg(["mean", "std"]).reset_index()
        part = part.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        pieces.append(part)
    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on=["scenario", "strategy"])
    return out


def write_md(path, agg, stats, quality_samples):
    lines = [
        "# Empirical task-quality replay",
        "",
        "This supplement samples per-request quality from measured MCQ/IoT task outputs instead of using one deterministic model-level quality constant.",
        "",
        "## Quality samples",
        "",
        "| Model | n | Mean | Std | >=0.45 | >=0.65 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, values in quality_samples.items():
        arr = np.array(values, dtype=float)
        lines.append(
            f"| {model} | {len(arr)} | {arr.mean():.3f} | {arr.std(ddof=1):.3f} | "
            f"{(arr >= 0.45).mean():.3f} | {(arr >= 0.65).mean():.3f} |"
        )
    lines.extend([
        "",
        "## Nominal replay metrics",
        "",
        "| Strategy | Joint SLA | Quality SLA | Latency(ms) | Energy(J) | Reward |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    nominal = agg[agg["scenario"] == "nominal"].copy()
    for _, row in nominal.sort_values("strategy").iterrows():
        lines.append(
            f"| {row['strategy']} | {row['joint_sla_rate_mean']:.3f}+/-{row['joint_sla_rate_std']:.3f} | "
            f"{row['quality_sla_rate_mean']:.3f}+/-{row['quality_sla_rate_std']:.3f} | "
            f"{row['mean_total_latency_ms_mean']:.1f}+/-{row['mean_total_latency_ms_std']:.1f} | "
            f"{row['mean_infer_energy_j_mean']:.2f}+/-{row['mean_infer_energy_j_std']:.2f} | "
            f"{row['mean_rl_reward_mean']:.3f}+/-{row['mean_rl_reward_std']:.3f} |"
        )
    lines.extend([
        "",
        "## Paired statistics vs empirical Q-learning",
        "",
        "| Baseline | Metric | Delta | 95% CI | p |",
        "|---|---|---:|---:|---:|",
    ])
    for _, row in stats.iterrows():
        lines.append(
            f"| {row['baseline']} | {row['metric']} | {row['mean_delta']:.4f} | "
            f"[{row['bootstrap_ci95_low']:.4f}, {row['bootstrap_ci95_high']:.4f}] | {row['wilcoxon_p']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--raw-q4", required=True)
    parser.add_argument("--raw-q8", required=True)
    parser.add_argument("--quality-csv", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    spec = vrl.RewardSpec("empirical_quality", beta_budget=0.0, beta_latency=0.45, beta_switch=0.05)
    latency_mean = base.load_latency(args.core)
    latency_samples = base.load_latency_samples(args.raw_q4, args.raw_q8)
    quality_samples = load_quality_samples([Path(p) for p in args.quality_csv])
    strategies = ["q_learning_empirical", "greedy_sla", "dynamic_budget_q8", "static_q4", "static_q8"]
    rows = []
    for seed in seeds:
        q_table = train_q_empirical(
            latency_mean,
            latency_samples,
            quality_samples,
            args.episodes,
            args.train_steps,
            seed,
            spec,
            alpha=0.18,
            gamma=0.20,
            eps_start=0.35,
            eps_end=0.04,
        )
        trace = vrl.make_trace(args.eval_steps, seed + 120000)
        for idx, strategy in enumerate(strategies):
            eval_rows = evaluate(
                strategy,
                trace,
                latency_mean,
                latency_samples,
                quality_samples,
                seed + 121000 + idx,
                q_table,
                spec,
            )
            for row in eval_rows:
                row["seed"] = seed
                row["scenario"] = "nominal"
            rows.extend(eval_rows)
    frame = pd.DataFrame(rows)
    summary = vrl.summarize(rows)
    agg = aggregate(summary)
    pairs = boot.paired_rows(
        summary,
        "nominal",
        "q_learning_empirical",
        ["greedy_sla", "dynamic_budget_q8", "static_q4", "static_q8"],
        ["joint_sla_rate", "quality_sla_rate", "mean_total_latency_ms", "mean_infer_energy_j", "mean_rl_reward"],
    )
    stats = boot.run_tests(pairs, out_dir / "empirical_quality_replay_bootstrap_stats_20260515.csv")
    frame.to_csv(out_dir / "empirical_quality_replay_requests_20260515.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "empirical_quality_replay_seed_summary_20260515.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(out_dir / "empirical_quality_replay_summary_20260515.csv", index=False, encoding="utf-8-sig")
    write_md(out_dir / "empirical_quality_replay_report_20260515.md", agg, stats, quality_samples)
    print(out_dir / "empirical_quality_replay_report_20260515.md")


if __name__ == "__main__":
    main()

