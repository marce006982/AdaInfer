#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

import bootstrap_statistical_tests as boot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonstationary-seed-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--treatment", default="q_online")
    parser.add_argument(
        "--baselines",
        default="greedy_sla_stale,greedy_sla_ewma,linucb_online,adaptive_linucb,q_frozen,dynamic_budget_q8",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.nonstationary_seed_summary, encoding="utf-8-sig")
    frame = frame[frame["phase"] == "post_shift"].copy()
    frame["scenario"] = "post_shift"
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    pairs = boot.paired_rows(
        frame,
        "post_shift",
        args.treatment,
        baselines,
        ["joint_sla_rate", "mean_latency_ms", "mean_energy_j", "q8_action_rate", "reward"],
    )
    table = boot.run_tests(pairs, out_dir / "rl_nonstationary_bootstrap_stats_20260515.csv")
    boot.write_md(
        out_dir / "rl_nonstationary_bootstrap_stats_20260515.md",
        table,
        "Non-stationary adaptation paired statistical comparison",
    )
    print(out_dir / "rl_nonstationary_bootstrap_stats_20260515.md")


if __name__ == "__main__":
    main()
