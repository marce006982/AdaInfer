#!/usr/bin/env python3
import argparse
import csv
import random
import statistics
import time
from collections import deque
from pathlib import Path


PROFILES = [
    {
        "device": "yahboom_nano",
        "token_budget": 16,
        "model": "Qwen2.5-0.5B Q4",
        "quality": 0.455,
        "latency_ms": 3126.171,
        "throughput": 11.137,
        "memory_mb": 1652.238,
        "power_w": 4.450,
    },
    {
        "device": "yahboom_nano",
        "token_budget": 32,
        "model": "Qwen2.5-0.5B Q4",
        "quality": 0.455,
        "latency_ms": 4488.099,
        "throughput": 10.845,
        "memory_mb": 1700.024,
        "power_w": 4.938,
    },
    {
        "device": "yahboom_nano",
        "token_budget": 64,
        "model": "Qwen2.5-0.5B Q4",
        "quality": 0.455,
        "latency_ms": 8040.906,
        "throughput": 10.662,
        "memory_mb": 1714.331,
        "power_w": 5.293,
    },
    {
        "device": "yahboom_nano",
        "token_budget": 128,
        "model": "Qwen2.5-0.5B Q4",
        "quality": 0.455,
        "latency_ms": 14368.089,
        "throughput": 10.558,
        "memory_mb": 1720.335,
        "power_w": 5.523,
    },
    {
        "device": "yahboom_nano",
        "token_budget": 16,
        "model": "Qwen3-0.6B Q8",
        "quality": 0.711,
        "latency_ms": 3591.670,
        "throughput": 8.743,
        "memory_mb": 2672.476,
        "power_w": 4.633,
    },
    {
        "device": "yahboom_nano",
        "token_budget": 32,
        "model": "Qwen3-0.6B Q8",
        "quality": 0.711,
        "latency_ms": 6020.373,
        "throughput": 8.163,
        "memory_mb": 2718.867,
        "power_w": 4.957,
    },
    {
        "device": "yahboom_nano",
        "token_budget": 64,
        "model": "Qwen3-0.6B Q8",
        "quality": 0.711,
        "latency_ms": 10019.562,
        "throughput": 7.698,
        "memory_mb": 2725.916,
        "power_w": 5.260,
    },
    {
        "device": "yahboom_nano",
        "token_budget": 128,
        "model": "Qwen3-0.6B Q8",
        "quality": 0.711,
        "latency_ms": 19181.687,
        "throughput": 7.481,
        "memory_mb": 2726.063,
        "power_w": 5.342,
    },
]


def now_us():
    return time.perf_counter() * 1_000_000.0


WEIGHTS = {
    "quality": 0.35,
    "latency_ms": 0.25,
    "throughput": 0.20,
    "memory_mb": 0.15,
    "power_w": 0.05,
}


def normalize(value, low, high, higher_is_better):
    eps = 1e-9
    score = (value - low) / (high - low + eps)
    if not higher_is_better:
        score = 1.0 - score
    return max(0.0, min(1.0, score))


def select_action(candidates, history, current_model, violation_count, threshold, h, delta):
    bounds = {}
    for key in ["latency_ms", "throughput", "memory_mb", "power_w"]:
        values = [item[key] for item in candidates] + list(history[key])
        bounds[key] = (min(values), max(values))

    best = None
    best_reward = -1.0
    for item in candidates:
        if item["quality"] < threshold:
            continue
        reward = WEIGHTS["quality"] * item["quality"]
        reward += WEIGHTS["latency_ms"] * normalize(item["latency_ms"], *bounds["latency_ms"], False)
        reward += WEIGHTS["throughput"] * normalize(item["throughput"], *bounds["throughput"], True)
        reward += WEIGHTS["memory_mb"] * normalize(item["memory_mb"], *bounds["memory_mb"], False)
        reward += WEIGHTS["power_w"] * normalize(item["power_w"], *bounds["power_w"], False)
        if reward > best_reward:
            best = item
            best_reward = reward

    if best is None:
        best = min(candidates, key=lambda item: item["latency_ms"])
        best_reward = 0.0

    current = next((item for item in candidates if item["model"] == current_model), candidates[0])
    current_ok = current["quality"] >= threshold
    if current_ok and best["model"] != current_model and violation_count < h:
        return current, best_reward, violation_count
    if current_ok and best["model"] != current_model and (best_reward - 0.0) < delta:
        return current, best_reward, violation_count
    return best, best_reward, 0 if best["quality"] >= threshold else violation_count + 1


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def run(iterations, window, h, delta, seed, platform):
    rng = random.Random(seed)
    history = {key: deque(maxlen=window) for key in ["latency_ms", "throughput", "memory_mb", "power_w"]}
    for item in PROFILES:
        for key in history:
            history[key].append(item[key])
    current_model = "Qwen2.5-0.5B Q4"
    violation_count = 0
    latencies_us = []
    switches = 0
    selected = current_model
    thresholds = [0.45, 0.65, 0.70]
    budgets = [16, 32, 64, 128]

    for _ in range(iterations):
        threshold = rng.choice(thresholds)
        budget = rng.choice(budgets)
        candidates = [item for item in PROFILES if item["token_budget"] == budget]
        start = now_us()
        choice, _, violation_count = select_action(
            candidates, history, current_model, violation_count, threshold, h, delta
        )
        elapsed_us = now_us() - start
        latencies_us.append(elapsed_us)
        if choice["model"] != current_model:
            switches += 1
        current_model = choice["model"]
        selected = choice["model"]
        for key in history:
            history[key].append(choice[key])

    return {
        "platform": platform,
        "iterations": iterations,
        "window": window,
        "hysteresis_h": h,
        "delta": delta,
        "mean_us": statistics.mean(latencies_us),
        "median_us": statistics.median(latencies_us),
        "p95_us": percentile(latencies_us, 0.95),
        "p99_us": percentile(latencies_us, 0.99),
        "max_us": max(latencies_us),
        "switches": switches,
        "switch_rate": switches / iterations,
        "last_selected": selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--hysteresis-h", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--platform", default="controller_host")
    parser.add_argument("--out")
    args = parser.parse_args()

    row = run(args.iterations, args.window, args.hysteresis_h, args.delta, args.seed, args.platform)
    fieldnames = list(row.keys())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        print(out)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
