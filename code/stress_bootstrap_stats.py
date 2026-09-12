#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

import bootstrap_statistical_tests as boot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-seed-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--treatment", default="q_learning")
    parser.add_argument(
        "--baselines",
        default="dynamic_budget_q8,hysteresis_rule,linucb,greedy_sla",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.validation_seed_summary, encoding="utf-8-sig")
    scenarios = [
        s
        for s in sorted(frame["scenario"].unique())
        if s not in {"nominal"}
    ]
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    metrics = [
        "joint_sla_rate",
        "mean_total_latency_ms",
        "mean_infer_energy_j",
        "mean_rl_reward",
    ]
    pairs = []
    for scenario in scenarios:
        pairs.extend(
            boot.paired_rows(
                frame,
                scenario,
                args.treatment,
                baselines,
                metrics,
            )
        )
    table = boot.run_tests(pairs, out_dir / "rl_network_stress_bootstrap_stats_20260515.csv")
    boot.write_md(
        out_dir / "rl_network_stress_bootstrap_stats_20260515.md",
        table,
        "Network-stress paired statistical comparison",
    )
    print(out_dir / "rl_network_stress_bootstrap_stats_20260515.md")


if __name__ == "__main__":
    main()
