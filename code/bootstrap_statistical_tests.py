#!/usr/bin/env python3
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def wilcoxon_signed_rank_p(diffs):
    diffs = np.array([float(d) for d in diffs if abs(float(d)) > 1e-12], dtype=float)
    n = len(diffs)
    if n == 0:
        return 1.0
    order = np.argsort(np.abs(diffs))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    observed = abs(float(ranks[diffs > 0].sum() - ranks[diffs < 0].sum()))
    total = 0
    extreme = 0
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        stat = abs(float((ranks * np.array(signs)).sum()))
        total += 1
        if stat >= observed - 1e-12:
            extreme += 1
    return extreme / total


def bootstrap_ci(diffs, rng, samples=10000):
    diffs = np.array(diffs, dtype=float)
    if len(diffs) == 0:
        return np.nan, np.nan
    means = np.empty(samples, dtype=float)
    for idx in range(samples):
        draw = rng.choice(diffs, size=len(diffs), replace=True)
        means[idx] = draw.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cohens_d(diffs):
    diffs = np.array(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    if sd == 0:
        return np.inf if diffs.mean() > 0 else (-np.inf if diffs.mean() < 0 else 0.0)
    return float(diffs.mean() / sd)


def paired_rows(frame, scenario, treatment, baselines, metrics, seed_col="seed", strategy_col="strategy"):
    rows = []
    sub = frame[frame["scenario"] == scenario].copy() if "scenario" in frame.columns else frame.copy()
    treat = sub[sub[strategy_col] == treatment]
    for baseline in baselines:
        base = sub[sub[strategy_col] == baseline]
        merged = treat.merge(base, on=seed_col, suffixes=("_treat", "_base"))
        for metric in metrics:
            rows.append((scenario, treatment, baseline, metric, merged[f"{metric}_treat"], merged[f"{metric}_base"]))
    return rows


def run_tests(pairs, out_csv):
    rng = np.random.default_rng(20260514)
    rows = []
    for scenario, treatment, baseline, metric, treatment_values, baseline_values in pairs:
        diffs = np.array(treatment_values, dtype=float) - np.array(baseline_values, dtype=float)
        ci_low, ci_high = bootstrap_ci(diffs, rng)
        rows.append(
            {
                "scenario": scenario,
                "treatment": treatment,
                "baseline": baseline,
                "metric": metric,
                "n_pairs": len(diffs),
                "treatment_mean": float(np.mean(treatment_values)),
                "baseline_mean": float(np.mean(baseline_values)),
                "mean_delta": float(diffs.mean()),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wilcoxon_p": wilcoxon_signed_rank_p(diffs),
                "paired_cohens_d": cohens_d(diffs),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out


def write_md(out_md, table, title):
    lines = [
        f"# {title}",
        "",
        "| Scenario | Treatment | Baseline | Metric | Treatment mean | Baseline mean | Delta | 95% bootstrap CI | Wilcoxon p | d |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['treatment']} | {row['baseline']} | {row['metric']} | "
            f"{row['treatment_mean']:.4f} | {row['baseline_mean']:.4f} | {row['mean_delta']:.4f} | "
            f"[{row['bootstrap_ci95_low']:.4f}, {row['bootstrap_ci95_high']:.4f}] | "
            f"{row['wilcoxon_p']:.4f} | {row['paired_cohens_d']:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-seed-summary", required=True)
    parser.add_argument("--nonstationary-seed-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(args.validation_seed_summary, encoding="utf-8-sig")
    nonstat = pd.read_csv(args.nonstationary_seed_summary, encoding="utf-8-sig")

    main_pairs = paired_rows(
        validation,
        "nominal",
        "q_learning",
        ["dynamic_budget_q8", "hysteresis_rule", "linucb", "greedy_sla"],
        ["joint_sla_rate", "mean_total_latency_ms", "mean_infer_energy_j", "mean_rl_reward"],
    )
    main_table = run_tests(main_pairs, out_dir / "rl_validation_bootstrap_stats_20260514.csv")
    write_md(out_dir / "rl_validation_bootstrap_stats_20260514.md", main_table, "Main RL policy statistical comparison")

    nonstat = nonstat.rename(
        columns={
            "joint_sla_rate": "joint_sla_rate",
            "mean_latency_ms": "mean_latency_ms",
            "mean_energy_j": "mean_energy_j",
            "reward": "reward",
        }
    )
    nonstat_post = nonstat[nonstat["phase"] == "post_shift"].copy()
    nonstat_post["scenario"] = "post_shift"
    non_pairs = paired_rows(
        nonstat_post,
        "post_shift",
        "q_online",
        ["greedy_sla_stale", "linucb_online", "adaptive_linucb", "q_frozen", "dynamic_budget_q8"],
        ["joint_sla_rate", "mean_latency_ms", "mean_energy_j", "reward"],
    )
    non_table = run_tests(non_pairs, out_dir / "rl_nonstationary_bootstrap_stats_20260514.csv")
    write_md(out_dir / "rl_nonstationary_bootstrap_stats_20260514.md", non_table, "Non-stationary adaptation statistical comparison")
    print(out_dir / "rl_validation_bootstrap_stats_20260514.md")
    print(out_dir / "rl_nonstationary_bootstrap_stats_20260514.md")


if __name__ == "__main__":
    main()
