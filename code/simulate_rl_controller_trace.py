#!/usr/bin/env python3
import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


QUALITY = {
    "Qwen2.5-0.5B Q4": 0.455,
    "Qwen3-0.6B Q8": 0.711,
}

SWITCH_COST = {
    "Qwen2.5-0.5B Q4": 6323.057,
    "Qwen3-0.6B Q8": 1749.366,
}


def load_latency(core_path):
    core = pd.read_csv(core_path, encoding="utf-8-sig")
    labels = {
        "qwen2.5-0.5b-q4": "Qwen2.5-0.5B Q4",
        "qwen3-0.6b-q8": "Qwen3-0.6B Q8",
    }
    table = {}
    for _, row in core[core["device"] == "yahboom_nano"].iterrows():
        label = labels.get(row["model_short"])
        if label:
            table[(label, int(row["token_budget"]))] = float(row["latency_ms_mean"])
    return table


def load_latency_samples(raw_q4_path, raw_q8_path):
    samples = {}
    for path, label in [
        (raw_q4_path, "Qwen2.5-0.5B Q4"),
        (raw_q8_path, "Qwen3-0.6B Q8"),
    ]:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        for _, row in frame.iterrows():
            budget = int(row["num_predict"])
            wall = float(row["wall_ms"])
            load = float(row.get("load_ms", 0.0) or 0.0)
            samples.setdefault((label, budget), []).append(max(0.0, wall - load))
    return samples


def make_trace(n, seed):
    rng = random.Random(seed)
    rows = []
    for idx in range(n):
        burst = (idx // 50) % 4 == 2
        wireless = rng.lognormvariate(5.1, 0.35)
        if burst:
            wireless += rng.uniform(1800, 5200)
        queue = rng.expovariate(1 / 180.0)
        if rng.random() < 0.08:
            queue += rng.uniform(500, 1600)
        phase = idx % 120
        high_critical = 80 <= phase < 100
        q_sla = 0.65 if high_critical else 0.45
        requested_budget = rng.choices([32, 64, 128], weights=[0.35, 0.45, 0.20])[0]
        deadline = 12000.0 if q_sla >= 0.65 else 9000.0
        rows.append(
            {
                "request_id": idx,
                "wireless_ms": wireless,
                "queue_ms": queue,
                "q_sla": q_sla,
                "requested_budget": requested_budget,
                "deadline_ms": deadline,
                "burst": int(burst),
                "criticality": "high" if high_critical else "low",
            }
        )
    return rows


def feasible_budgets(requested_budget):
    return [budget for budget in [16, 32, 64, 128] if budget <= requested_budget]


def actions_for(request):
    return [(model, budget) for model in QUALITY for budget in feasible_budgets(request["requested_budget"])]


def choose_budget(latency_mean, model, requested_budget, wireless, queue, deadline):
    for budget in sorted(feasible_budgets(requested_budget), reverse=True):
        if wireless + queue + latency_mean[(model, budget)] <= deadline:
            return budget
    return min(feasible_budgets(requested_budget))


def state_key(request, current_model):
    net_pressure = (request["wireless_ms"] + request["queue_ms"]) / request["deadline_ms"]
    pressure_bucket = "high" if net_pressure > 0.25 else "low"
    current = "q8" if "Q8" in current_model else "q4"
    return (
        "high_q" if request["q_sla"] >= 0.65 else "low_q",
        int(request["requested_budget"]),
        pressure_bucket,
        int(request["burst"]),
        current,
    )


def step_eval(request, action, current_model, latency_ms):
    model, budget = action
    switch = int(model != current_model)
    total = request["wireless_ms"] + request["queue_ms"] + latency_ms + switch * SWITCH_COST[model]
    quality_ok = QUALITY[model] >= request["q_sla"]
    latency_ok = total <= request["deadline_ms"]
    retention = budget / request["requested_budget"]
    reward = (
        1.0 * int(quality_ok and latency_ok)
        + 0.40 * retention
        - 0.15 * min(2.0, total / request["deadline_ms"])
        - 0.05 * switch
    )
    power_w = 5.293 if "Q4" in model else 5.260
    energy_j = power_w * (latency_ms + switch * SWITCH_COST[model]) / 1000.0
    return {
        "model": model,
        "selected_budget": budget,
        "switch": switch,
        "infer_ms": latency_ms,
        "total_latency_ms": total,
        "infer_energy_j": energy_j,
        "quality": QUALITY[model],
        "budget_retention": retention,
        "quality_sla_met": int(quality_ok),
        "latency_sla_met": int(latency_ok),
        "joint_sla_met": int(quality_ok and latency_ok),
        "rl_reward": reward,
    }


def draw_latency(samples, mean_table, action, rng):
    pool = samples.get(action)
    if pool:
        return rng.choice(pool)
    return mean_table[action]


def train_q_learning(latency_mean, latency_samples, episodes, steps, seed, alpha, gamma, epsilon):
    rng = random.Random(seed)
    q = defaultdict(float)
    returns = []
    for episode in range(episodes):
        trace = make_trace(steps, seed + episode)
        current_model = "Qwen2.5-0.5B Q4"
        total_reward = 0.0
        for idx, request in enumerate(trace):
            state = state_key(request, current_model)
            feasible = actions_for(request)
            if rng.random() < epsilon:
                action = rng.choice(feasible)
            else:
                action = max(feasible, key=lambda a: q[(state, a)])
            latency_ms = draw_latency(latency_samples, latency_mean, action, rng)
            outcome = step_eval(request, action, current_model, latency_ms)
            total_reward += outcome["rl_reward"]
            next_model = action[0]
            if idx + 1 < len(trace):
                next_state = state_key(trace[idx + 1], next_model)
                next_actions = actions_for(trace[idx + 1])
                target = outcome["rl_reward"] + gamma * max(q[(next_state, a)] for a in next_actions)
            else:
                target = outcome["rl_reward"]
            q[(state, action)] += alpha * (target - q[(state, action)])
            current_model = next_model
        returns.append(total_reward)
    return q, returns


def evaluate_policy(name, trace, latency_mean, latency_samples, q_table, seed):
    rng = random.Random(seed)
    rows = []
    current_model = "Qwen2.5-0.5B Q4"
    low_sla_count = 0
    for request in trace:
        if name == "static_q4":
            action = ("Qwen2.5-0.5B Q4", request["requested_budget"])
        elif name == "static_q8":
            action = ("Qwen3-0.6B Q8", request["requested_budget"])
        elif name == "dynamic_budget_q8":
            budget = choose_budget(
                latency_mean,
                "Qwen3-0.6B Q8",
                request["requested_budget"],
                request["wireless_ms"],
                request["queue_ms"],
                request["deadline_ms"],
            )
            action = ("Qwen3-0.6B Q8", budget)
        elif name == "hysteresis_rule":
            if QUALITY[current_model] < request["q_sla"]:
                model = "Qwen3-0.6B Q8"
                low_sla_count = 0
            elif request["q_sla"] <= QUALITY["Qwen2.5-0.5B Q4"]:
                low_sla_count += 1
                model = "Qwen2.5-0.5B Q4" if low_sla_count >= 2 else current_model
            else:
                model = current_model
                low_sla_count = 0
            budget = choose_budget(
                latency_mean,
                model,
                request["requested_budget"],
                request["wireless_ms"],
                request["queue_ms"],
                request["deadline_ms"],
            )
            action = (model, budget)
        elif name == "q_learning":
            state = state_key(request, current_model)
            action = max(actions_for(request), key=lambda a: q_table[(state, a)])
        else:
            raise ValueError(name)
        latency_ms = draw_latency(latency_samples, latency_mean, action, rng)
        outcome = step_eval(request, action, current_model, latency_ms)
        rows.append({**request, **outcome, "strategy": name})
        current_model = action[0]
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    return (
        df.groupby("strategy")
        .agg(
            requests=("request_id", "count"),
            mean_total_latency_ms=("total_latency_ms", "mean"),
            p95_total_latency_ms=("total_latency_ms", lambda x: x.quantile(0.95)),
            mean_infer_energy_j=("infer_energy_j", "mean"),
            quality_sla_rate=("quality_sla_met", "mean"),
            latency_sla_rate=("latency_sla_met", "mean"),
            joint_sla_rate=("joint_sla_met", "mean"),
            mean_budget_retention=("budget_retention", "mean"),
            switch_count=("switch", "sum"),
            mean_rl_reward=("rl_reward", "mean"),
        )
        .reset_index()
    )


def write_md(summary, returns, out_path):
    lines = [
        "# AdaInfer tabular Q-learning controller",
        "",
        "## Objective",
        "",
        "This experiment instantiates AdaInfer as a hardware-aware adaptive controller. "
        "Measured hardware profiles provide the environment model. The controller observes the quality threshold, request budget, network pressure, burst state, and resident model, then selects a model and generation budget. "
        "Training uses tabular Q-learning and closes the state-action-reward loop evaluated in the manuscript.",
        "",
        "## Training configuration",
        "",
        f"- Episodes: {len(returns)}",
        "- Steps per episode: 500",
        "- State: quality SLA bucket, requested budget, network pressure, burst flag, resident model",
        "- Action: model configuration plus generation budget",
        "- Reward: joint SLA + budget retention - latency ratio penalty - switch penalty",
        f"- First episode return: {returns[0]:.2f}",
        f"- Last episode return: {returns[-1]:.2f}",
        "",
        "## Evaluation results",
        "",
        "| Strategy | Requests | Mean latency(ms) | p95 latency(ms) | Joint SLA | Budget retention | Switches | Mean RL reward |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['strategy']} | {int(row['requests'])} | {row['mean_total_latency_ms']:.1f} | "
            f"{row['p95_total_latency_ms']:.1f} | {row['joint_sla_rate']:.3f} | "
            f"{row['mean_budget_retention']:.3f} | {int(row['switch_count'])} | {row['mean_rl_reward']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This run checks the complete data-to-controller path on a short trace. Use the full commands in docs/REPRODUCE.md to reproduce the reported experiments.",
        ]
    )
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--raw-q4", required=True)
    parser.add_argument("--raw-q8", required=True)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-requests", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    latency_mean = load_latency(args.core)
    latency_samples = load_latency_samples(args.raw_q4, args.raw_q8)
    q_table, returns = train_q_learning(
        latency_mean,
        latency_samples,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        alpha=0.18,
        gamma=0.85,
        epsilon=0.12,
    )
    test_trace = make_trace(args.steps, args.seed + 10000)
    strategies = ["static_q4", "static_q8", "dynamic_budget_q8", "hysteresis_rule", "q_learning"]
    rows = []
    for idx, strategy in enumerate(strategies):
        rows.extend(evaluate_policy(strategy, test_trace, latency_mean, latency_samples, q_table, args.seed + 20000 + idx))
    summary = summarize(rows)
    Path(args.out_requests).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_requests, index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_summary, index=False, encoding="utf-8-sig")
    write_md(summary, returns, args.out_md)
    print(args.out_requests)
    print(args.out_summary)
    print(args.out_md)


if __name__ == "__main__":
    main()
