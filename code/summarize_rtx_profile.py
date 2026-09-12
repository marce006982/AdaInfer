#!/usr/bin/env python3
"""Aggregate local RTX profiling while retaining cold/warm variability."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def p95(values):
    return float(np.percentile(values, 95, method="linear"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    keys = ["input_tokens", "max_new_tokens", "batch_size", "run_label"]
    rows = []
    for key, group in df.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        for source, target in [
            ("wall_ms_per_request", "latency_ms"),
            ("tokens_per_sec_per_request", "tokens_per_sec"),
            ("power_w_avg", "power_w"),
            ("peak_torch_memory_mb", "peak_torch_memory_mb"),
        ]:
            values = pd.to_numeric(group[source], errors="coerce").dropna().to_numpy()
            row[f"{target}_n"] = int(len(values))
            row[f"{target}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{target}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{target}_p50"] = float(np.percentile(values, 50)) if len(values) else np.nan
            row[f"{target}_p95"] = p95(values) if len(values) else np.nan
        row["gpu_mem_used_mb_mean"] = float(pd.to_numeric(group["gpu_mem_used_mb_avg"], errors="coerce").mean())
        row["gpu_mem_used_mb_max"] = float(pd.to_numeric(group["gpu_mem_used_mb_max"], errors="coerce").max())
        rows.append(row)
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
