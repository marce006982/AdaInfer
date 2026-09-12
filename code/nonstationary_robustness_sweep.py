#!/usr/bin/env python3
"""Run the reviewer-requested non-stationary robustness checks.

The grid uses independently seeded 5k-request validation traces.  The timing
check reuses the same Q8-only profile shift at 20%, 50%, and 80% of 10k-request
traces.  Only seed-level summaries are retained for the sweeps.
"""

import argparse
import copy
from collections import defaultdict
from itertools import product
from pathlib import Path

import pandas as pd

import nonstationary_rl_adaptation as nonstat
import simulate_rl_controller_trace as base
import validate_rl_controller as vrl


GRID_WINDOWS = (20, 40, 60)
GRID_THRESHOLDS = (0.60, 0.68, 0.75)
SHIFT_FRACTIONS = (0.20, 0.50, 0.80)
SHIFT_STRATEGIES = (
    "q_online",
    "adaptive_linucb",
    "linucb_online",
    "greedy_sla_ewma",
    "greedy_sla_stale",
)


def parse_seeds(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def phase_summary(rows, seed, shift_step):
    for row in rows:
        row["seed"] = seed
    summary, _ = nonstat.summarize(rows, shift_step, window=40)
    return summary[summary["phase"] == "post_shift"].copy()


def evaluate(
    strategy,
    trace,
    latency_mean,
    latency_samples,
    seed,
    q_table,
    trained_linucb,
    spec,
    shift_step,
    adaptive_window=40,
    adaptive_threshold=0.68,
):
    q_copy = defaultdict(float, q_table.copy())
    linucb = copy.deepcopy(trained_linucb)
    adaptive_linucb = nonstat.ResetLinUCB(
        learner=copy.deepcopy(trained_linucb),
        window=adaptive_window,
        threshold=adaptive_threshold,
    )
    return nonstat.evaluate_adaptation(
        strategy,
        trace,
        latency_mean,
        latency_samples,
        seed,
        q_copy,
        linucb,
        adaptive_linucb,
        spec,
        shift_step,
        2.0,
    )


def aggregate(frame, groups):
    metrics = [
        "joint_sla_rate",
        "mean_latency_ms",
        "mean_energy_j",
        "q8_action_rate",
        "reward",
    ]
    return frame.groupby(groups, as_index=False)[metrics].agg(["mean", "std"]).reset_index()


def flatten_columns(frame):
    frame.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in frame.columns
    ]
    return frame


def write_report(out_path, grid_summary, timing_summary, validation_steps, timing_steps):
    lines = [
        "# Non-stationary robustness supplement",
        "",
        "The sweeps use the same Jetson Nano trace-driven replay as the main non-stationary experiment.",
        "The injected disturbance doubles only the Q8 compute-profile latency. Measured RTT samples, queue generation, deadlines, and the periodic burst-state marker are unchanged between pre- and post-shift phases.",
        "",
        "## Adaptive LinUCB validation grid",
        "",
        f"- Independent validation trajectories: {validation_steps} requests per seed",
        "- Parameters: window in {20, 40, 60}; reset threshold in {0.60, 0.68, 0.75}",
        "",
        "| Window | Threshold | Joint SLA | Latency (ms) | Energy (J) | Q8 rate | Reward |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in grid_summary.sort_values(["joint_sla_rate_mean", "reward_mean"], ascending=[False, False]).iterrows():
        lines.append(
            f"| {int(row['adaptive_window'])} | {row['adaptive_threshold']:.2f} | "
            f"{row['joint_sla_rate_mean']:.3f}+/-{row['joint_sla_rate_std']:.3f} | "
            f"{row['mean_latency_ms_mean']:.1f}+/-{row['mean_latency_ms_std']:.1f} | "
            f"{row['mean_energy_j_mean']:.2f}+/-{row['mean_energy_j_std']:.2f} | "
            f"{row['q8_action_rate_mean']:.3f}+/-{row['q8_action_rate_std']:.3f} | "
            f"{row['reward_mean']:.3f}+/-{row['reward_std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Shift-timing robustness",
            "",
            f"- Evaluation trajectories: {timing_steps} requests per seed",
            "- Shift position is expressed as a fraction of the request trace.",
            "",
            "| Shift position | Strategy | Joint SLA | Latency (ms) | Energy (J) | Q8 rate | Reward |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in timing_summary.sort_values(["shift_fraction", "strategy"]).iterrows():
        lines.append(
            f"| {row['shift_fraction']:.0%} | {row['strategy']} | "
            f"{row['joint_sla_rate_mean']:.3f}+/-{row['joint_sla_rate_std']:.3f} | "
            f"{row['mean_latency_ms_mean']:.1f}+/-{row['mean_latency_ms_std']:.1f} | "
            f"{row['mean_energy_j_mean']:.2f}+/-{row['mean_energy_j_std']:.2f} | "
            f"{row['q8_action_rate_mean']:.3f}+/-{row['q8_action_rate_std']:.3f} | "
            f"{row['reward_mean']:.3f}+/-{row['reward_std']:.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--raw-q4", required=True)
    parser.add_argument("--raw-q8", required=True)
    parser.add_argument("--rtt-trace", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--grid-eval-steps", type=int, default=5000)
    parser.add_argument("--timing-eval-steps", type=int, default=10000)
    parser.add_argument("--rtt-multiplier", type=float, default=80.0)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    spec = vrl.RewardSpec("online_adaptation", beta_budget=0.0, beta_latency=0.45, beta_switch=0.05)
    rtt_values = nonstat.load_rtt_trace(args.rtt_trace)
    latency_mean = base.load_latency(args.core)
    latency_samples = base.load_latency_samples(args.raw_q4, args.raw_q8)

    grid_rows = []
    timing_rows = []
    for seed in seeds:
        q_table, _, _ = vrl.train_q_learning(
            latency_mean,
            latency_samples,
            episodes=args.episodes,
            steps=args.train_steps,
            seed=seed,
            spec=spec,
            alpha=0.18,
            gamma=0.20,
            eps_start=0.35,
            eps_end=0.04,
        )
        trained_linucb = vrl.train_linucb(
            latency_mean, latency_samples, args.episodes, args.train_steps, seed, spec
        )

        grid_shift = args.grid_eval_steps // 2
        grid_trace = nonstat.make_measured_trace(
            args.grid_eval_steps,
            seed + 90000,
            rtt_values,
            grid_shift,
            args.rtt_multiplier,
            1.0,
        )
        for window, threshold in product(GRID_WINDOWS, GRID_THRESHOLDS):
            rows = evaluate(
                "adaptive_linucb",
                grid_trace,
                latency_mean,
                latency_samples,
                seed + 120000,
                q_table,
                trained_linucb,
                spec,
                grid_shift,
                adaptive_window=window,
                adaptive_threshold=threshold,
            )
            summary = phase_summary(rows, seed, grid_shift)
            summary["adaptive_window"] = window
            summary["adaptive_threshold"] = threshold
            grid_rows.append(summary)

        for fraction in SHIFT_FRACTIONS:
            shift_step = int(args.timing_eval_steps * fraction)
            trace = nonstat.make_measured_trace(
                args.timing_eval_steps,
                seed + 130000,
                rtt_values,
                shift_step,
                args.rtt_multiplier,
                1.0,
            )
            for strategy in SHIFT_STRATEGIES:
                rows = evaluate(
                    strategy,
                    trace,
                    latency_mean,
                    latency_samples,
                    seed + 140000 + int(fraction * 100),
                    q_table,
                    trained_linucb,
                    spec,
                    shift_step,
                )
                summary = phase_summary(rows, seed, shift_step)
                summary["shift_fraction"] = fraction
                summary["shift_step"] = shift_step
                timing_rows.append(summary)

    grid_seed = pd.concat(grid_rows, ignore_index=True)
    timing_seed = pd.concat(timing_rows, ignore_index=True)
    grid_summary = flatten_columns(aggregate(grid_seed, ["adaptive_window", "adaptive_threshold", "strategy"]))
    timing_summary = flatten_columns(aggregate(timing_seed, ["shift_fraction", "shift_step", "strategy"]))
    grid_seed.to_csv(out_dir / "adaptive_linucb_grid_seed_summary.csv", index=False, encoding="utf-8-sig")
    grid_summary.to_csv(out_dir / "adaptive_linucb_grid_summary.csv", index=False, encoding="utf-8-sig")
    timing_seed.to_csv(out_dir / "shift_timing_seed_summary.csv", index=False, encoding="utf-8-sig")
    timing_summary.to_csv(out_dir / "shift_timing_summary.csv", index=False, encoding="utf-8-sig")
    report_path = out_dir / "nonstationary_robustness_supplement.md"
    write_report(report_path, grid_summary, timing_summary, args.grid_eval_steps, args.timing_eval_steps)
    print(report_path)


if __name__ == "__main__":
    main()
