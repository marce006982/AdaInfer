#!/usr/bin/env python3
import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import simulate_rl_controller_trace as base


MODELS = list(base.QUALITY.keys())
BUDGETS = [16, 32, 64, 128]
ALL_ACTIONS = [(model, budget) for model in MODELS for budget in BUDGETS]


@dataclass(frozen=True)
class RewardSpec:
    name: str
    quality_gate: bool = True
    beta_budget: float = 0.40
    beta_latency: float = 0.15
    beta_switch: float = 0.05


FULL_REWARD = RewardSpec("full")


def make_trace(n, seed, rtt_scale=1.0, burst_ratio=0.25, loss_rate=0.0, deadline_scale=1.0):
    rng = random.Random(seed)
    rows = []
    burst = False
    enter_prob = max(0.01, burst_ratio / 9.0)
    exit_prob = max(0.10, min(0.70, 0.50 - burst_ratio / 2.0))
    for idx in range(n):
        if burst:
            burst = rng.random() >= exit_prob
        else:
            burst = rng.random() < enter_prob

        wireless = rng.lognormvariate(5.1, 0.35) * rtt_scale
        if burst:
            wireless += rng.uniform(1800, 5200) * rtt_scale
        if rng.random() < loss_rate:
            wireless += rng.gammavariate(2.2, 1100.0) * rtt_scale

        queue = rng.expovariate(1 / 180.0)
        if rng.random() < 0.08 + loss_rate:
            queue += rng.uniform(500, 1600)

        phase = idx % 120
        high_critical = 80 <= phase < 100
        q_sla = 0.65 if high_critical else 0.45
        requested_budget = rng.choices([32, 64, 128], weights=[0.35, 0.45, 0.20])[0]
        deadline = (12000.0 if q_sla >= 0.65 else 9000.0) * deadline_scale
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
                "rtt_scale": rtt_scale,
                "loss_rate": loss_rate,
            }
        )
    return rows


def feasible_actions(request):
    return [
        (model, budget)
        for model in MODELS
        for budget in BUDGETS
        if budget <= request["requested_budget"]
    ]


def step_eval(request, action, current_model, latency_ms, spec=FULL_REWARD):
    model, budget = action
    switch = int(model != current_model)
    total = request["wireless_ms"] + request["queue_ms"] + latency_ms + switch * base.SWITCH_COST[model]
    quality_ok = base.QUALITY[model] >= request["q_sla"]
    latency_ok = total <= request["deadline_ms"]
    retention = budget / request["requested_budget"]
    sla_term = int((quality_ok if spec.quality_gate else True) and latency_ok)
    reward = (
        sla_term
        + spec.beta_budget * retention
        - spec.beta_latency * min(2.0, total / request["deadline_ms"])
        - spec.beta_switch * switch
    )
    power_w = 5.293 if "Q4" in model else 5.260
    energy_j = power_w * (latency_ms + switch * base.SWITCH_COST[model]) / 1000.0
    return {
        "model": model,
        "selected_budget": budget,
        "switch": switch,
        "infer_ms": latency_ms,
        "total_latency_ms": total,
        "infer_energy_j": energy_j,
        "quality": base.QUALITY[model],
        "budget_retention": retention,
        "quality_sla_met": int(quality_ok),
        "latency_sla_met": int(latency_ok),
        "joint_sla_met": int(quality_ok and latency_ok),
        "rl_reward": reward,
    }


def state_key(request, current_model):
    return base.state_key(request, current_model)


def expected_total(request, action, current_model, latency_mean):
    model, _ = action
    return (
        request["wireless_ms"]
        + request["queue_ms"]
        + latency_mean[action]
        + int(model != current_model) * base.SWITCH_COST[model]
    )


def choose_expected_best(request, current_model, latency_mean, spec):
    feasible = feasible_actions(request)
    return max(
        feasible,
        key=lambda action: step_eval(
            request, action, current_model, latency_mean[action], spec
        )["rl_reward"],
    )


def choose_latency_greedy(request, current_model, latency_mean):
    return min(feasible_actions(request), key=lambda a: expected_total(request, a, current_model, latency_mean))


def choose_quality_greedy(request, current_model, latency_mean):
    return max(
        feasible_actions(request),
        key=lambda a: (
            base.QUALITY[a[0]],
            a[1] / request["requested_budget"],
            -expected_total(request, a, current_model, latency_mean),
        ),
    )


def choose_sla_greedy(request, current_model, latency_mean):
    feasible = feasible_actions(request)
    quality_feasible = [a for a in feasible if base.QUALITY[a[0]] >= request["q_sla"]]
    pool = quality_feasible or feasible
    deadline_feasible = [a for a in pool if expected_total(request, a, current_model, latency_mean) <= request["deadline_ms"]]
    pool = deadline_feasible or pool
    return min(pool, key=lambda a: expected_total(request, a, current_model, latency_mean))


def train_q_learning(latency_mean, latency_samples, episodes, steps, seed, spec, alpha, gamma, eps_start, eps_end):
    rng = random.Random(seed)
    q = defaultdict(float)
    visits = Counter()
    records = []
    for episode in range(episodes):
        trace = make_trace(steps, seed + episode)
        current_model = "Qwen2.5-0.5B Q4"
        total_reward = 0.0
        actions_seen = Counter()
        epsilon = eps_end + (eps_start - eps_end) * math.exp(-episode / max(1.0, episodes / 4.0))
        for idx, request in enumerate(trace):
            state = state_key(request, current_model)
            feasible = feasible_actions(request)
            if rng.random() < epsilon:
                action = rng.choice(feasible)
            else:
                action = max(feasible, key=lambda a: q[(state, a)])
            latency_ms = base.draw_latency(latency_samples, latency_mean, action, rng)
            outcome = step_eval(request, action, current_model, latency_ms, spec)
            total_reward += outcome["rl_reward"]
            visits[(state, action)] += 1
            actions_seen[action] += 1
            next_model = action[0]
            if idx + 1 < len(trace):
                next_state = state_key(trace[idx + 1], next_model)
                next_actions = feasible_actions(trace[idx + 1])
                target = outcome["rl_reward"] + gamma * max(q[(next_state, a)] for a in next_actions)
            else:
                target = outcome["rl_reward"]
            q[(state, action)] += alpha * (target - q[(state, action)])
            current_model = next_model

        probs = np.array(list(actions_seen.values()), dtype=float)
        probs = probs / probs.sum() if probs.sum() else probs
        entropy = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())
        q_values = np.array(list(q.values()), dtype=float) if q else np.array([0.0])
        records.append(
            {
                "seed": seed,
                "reward_spec": spec.name,
                "episode": episode + 1,
                "epsilon": epsilon,
                "episode_return": total_reward,
                "action_entropy": entropy,
                "q_value_mean": float(q_values.mean()),
                "q_value_std": float(q_values.std()),
                "visited_state_actions": len(q),
            }
        )
    return q, pd.DataFrame(records), visits


def action_feature(request, current_model, action):
    pressure = (request["wireless_ms"] + request["queue_ms"]) / request["deadline_ms"]
    return np.array(
        [
            1.0,
            request["q_sla"],
            request["requested_budget"] / 128.0,
            min(2.0, pressure),
            float(request["burst"]),
            float("Q8" in current_model),
            float("Q8" in action[0]),
            action[1] / request["requested_budget"],
            float(action[0] != current_model),
        ],
        dtype=float,
    )


class DisjointLinUCB:
    def __init__(self, dim, alpha=0.45):
        self.alpha = alpha
        self.a_inv = {action: np.eye(dim) for action in ALL_ACTIONS}
        self.b = {action: np.zeros(dim) for action in ALL_ACTIONS}

    def score(self, request, current_model, action, explore=True):
        x = action_feature(request, current_model, action)
        inv = self.a_inv[action]
        theta = inv @ self.b[action]
        bonus = self.alpha * np.sqrt(x @ inv @ x) if explore else 0.0
        return float(theta @ x + bonus)

    def choose(self, request, current_model, explore=True):
        feasible = feasible_actions(request)
        return max(feasible, key=lambda a: self.score(request, current_model, a, explore))

    def update(self, request, current_model, action, reward):
        x = action_feature(request, current_model, action)
        inv = self.a_inv[action]
        inv_x = inv @ x
        denom = 1.0 + x @ inv_x
        self.a_inv[action] = inv - np.outer(inv_x, inv_x) / denom
        self.b[action] += reward * x


def train_linucb(latency_mean, latency_samples, episodes, steps, seed, spec):
    rng = random.Random(seed + 73000)
    learner = DisjointLinUCB(dim=9, alpha=0.45)
    for episode in range(episodes):
        trace = make_trace(steps, seed + 41000 + episode)
        current_model = "Qwen2.5-0.5B Q4"
        for request in trace:
            action = learner.choose(request, current_model, explore=True)
            latency_ms = base.draw_latency(latency_samples, latency_mean, action, rng)
            outcome = step_eval(request, action, current_model, latency_ms, spec)
            learner.update(request, current_model, action, outcome["rl_reward"])
            current_model = action[0]
    return learner


def evaluate_strategy(strategy, trace, latency_mean, latency_samples, seed, q_table=None, linucb=None, spec=FULL_REWARD):
    rng = random.Random(seed)
    rows = []
    current_model = "Qwen2.5-0.5B Q4"
    low_sla_count = 0
    for request in trace:
        if strategy == "static_q4":
            action = ("Qwen2.5-0.5B Q4", request["requested_budget"])
        elif strategy == "static_q8":
            action = ("Qwen3-0.6B Q8", request["requested_budget"])
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
        elif strategy == "hysteresis_rule":
            if base.QUALITY[current_model] < request["q_sla"]:
                model = "Qwen3-0.6B Q8"
                low_sla_count = 0
            elif request["q_sla"] <= base.QUALITY["Qwen2.5-0.5B Q4"]:
                low_sla_count += 1
                model = "Qwen2.5-0.5B Q4" if low_sla_count >= 2 else current_model
            else:
                model = current_model
                low_sla_count = 0
            budget = base.choose_budget(
                latency_mean,
                model,
                request["requested_budget"],
                request["wireless_ms"],
                request["queue_ms"],
                request["deadline_ms"],
            )
            action = (model, budget)
        elif strategy == "q_learning":
            state = state_key(request, current_model)
            action = max(feasible_actions(request), key=lambda a: q_table[(state, a)])
        elif strategy == "oracle_reward":
            action = choose_expected_best(request, current_model, latency_mean, spec)
        elif strategy == "linucb":
            action = linucb.choose(request, current_model, explore=False)
        elif strategy == "greedy_latency":
            action = choose_latency_greedy(request, current_model, latency_mean)
        elif strategy == "greedy_quality":
            action = choose_quality_greedy(request, current_model, latency_mean)
        elif strategy == "greedy_sla":
            action = choose_sla_greedy(request, current_model, latency_mean)
        else:
            raise ValueError(strategy)
        latency_ms = base.draw_latency(latency_samples, latency_mean, action, rng)
        outcome = step_eval(request, action, current_model, latency_ms, spec)
        rows.append({**request, **outcome, "strategy": strategy})
        current_model = action[0]
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    return (
        df.groupby(["seed", "scenario", "strategy"], as_index=False)
        .agg(
            requests=("request_id", "count"),
            mean_total_latency_ms=("total_latency_ms", "mean"),
            p95_total_latency_ms=("total_latency_ms", lambda x: x.quantile(0.95)),
            mean_infer_energy_j=("infer_energy_j", "mean"),
            quality_sla_rate=("quality_sla_met", "mean"),
            latency_sla_rate=("latency_sla_met", "mean"),
            joint_sla_rate=("joint_sla_met", "mean"),
            mean_budget_retention=("budget_retention", "mean"),
            switch_rate_per_100=("switch", lambda x: 100.0 * x.sum() / len(x)),
            mean_rl_reward=("rl_reward", "mean"),
        )
    )


def aggregate(seed_summary):
    metrics = [
        "mean_total_latency_ms",
        "p95_total_latency_ms",
        "mean_infer_energy_j",
        "quality_sla_rate",
        "latency_sla_rate",
        "joint_sla_rate",
        "mean_budget_retention",
        "switch_rate_per_100",
        "mean_rl_reward",
    ]
    grouped = seed_summary.groupby(["scenario", "strategy"])
    parts = []
    for metric in metrics:
        part = grouped[metric].agg(["mean", "std"]).reset_index()
        part = part.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        parts.append(part)
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["scenario", "strategy"])
    return out


def normal_p_value(xs, ys):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    nx, ny = len(xs), len(ys)
    vx, vy = xs.var(ddof=1), ys.var(ddof=1)
    denom = math.sqrt(vx / nx + vy / ny)
    if denom == 0:
        return 1.0 if abs(xs.mean() - ys.mean()) < 1e-12 else 0.0
    z = abs(xs.mean() - ys.mean()) / denom
    return float(math.erfc(z / math.sqrt(2.0)))


def stats_table(seed_summary):
    rows = []
    nominal = seed_summary[seed_summary["scenario"] == "nominal"]
    rl = nominal[nominal["strategy"] == "q_learning"]
    for baseline_name in sorted(set(nominal["strategy"]) - {"q_learning"}):
        other = nominal[nominal["strategy"] == baseline_name]
        merged = rl.merge(other, on="seed", suffixes=("_rl", "_base"))
        for metric in ["joint_sla_rate", "mean_total_latency_ms", "mean_infer_energy_j", "mean_rl_reward"]:
            rows.append(
                {
                    "baseline": baseline_name,
                    "metric": metric,
                    "rl_mean": merged[f"{metric}_rl"].mean(),
                    "baseline_mean": merged[f"{metric}_base"].mean(),
                    "delta_rl_minus_baseline": merged[f"{metric}_rl"].mean() - merged[f"{metric}_base"].mean(),
                    "approx_two_sided_p": normal_p_value(merged[f"{metric}_rl"], merged[f"{metric}_base"]),
                }
            )
    return pd.DataFrame(rows)


def plot_learning_curve(diag, out_png):
    grouped = diag.groupby("episode")["episode_return"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.plot(grouped["episode"], grouped["mean"], color="#2563EB", label="AdaInfer-RL")
    ax.fill_between(
        grouped["episode"],
        grouped["mean"] - grouped["std"].fillna(0),
        grouped["mean"] + grouped["std"].fillna(0),
        color="#93C5FD",
        alpha=0.35,
        label="seed std",
    )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode return")
    ax.set_title("Q-learning training dynamics")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_visit_heatmap(visits, out_png):
    state_counts = Counter()
    action_counts = Counter()
    for (state, action), count in visits.items():
        state_counts[state] += count
        action_counts[action] += count
    states = [item[0] for item in state_counts.most_common(24)]
    actions = [item[0] for item in action_counts.most_common()]
    mat = np.zeros((len(states), len(actions)))
    for i, state in enumerate(states):
        for j, action in enumerate(actions):
            mat[i, j] = visits.get((state, action), 0)
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    im = ax.imshow(np.log1p(mat), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(actions)))
    ax.set_xticklabels([f"{a[0].split()[1]}-{a[1]}" for a in actions], rotation=45, ha="right")
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(["/".join(map(str, s)) for s in states], fontsize=7)
    ax.set_xlabel("Action (model-budget)")
    ax.set_ylabel("State bucket")
    ax.set_title("State-action visitation frequency (log1p)")
    fig.colorbar(im, ax=ax, label="log(1 + visits)")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_baseline_bars(agg, out_png):
    nominal = agg[agg["scenario"] == "nominal"].copy()
    order = [
        "oracle_reward",
        "q_learning",
        "linucb",
        "greedy_sla",
        "hysteresis_rule",
        "dynamic_budget_q8",
        "greedy_latency",
        "greedy_quality",
        "static_q4",
        "static_q8",
    ]
    nominal["strategy"] = pd.Categorical(nominal["strategy"], order, ordered=True)
    nominal = nominal.sort_values("strategy")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    metrics = [
        ("joint_sla_rate_mean", "joint_sla_rate_std", "Joint SLA", "#16A34A"),
        ("mean_total_latency_ms_mean", "mean_total_latency_ms_std", "Latency (ms)", "#2563EB"),
        ("mean_infer_energy_j_mean", "mean_infer_energy_j_std", "Energy (J)", "#DC2626"),
    ]
    for ax, (mean_col, std_col, title, color) in zip(axes, metrics):
        ax.bar(nominal["strategy"].astype(str), nominal[mean_col], yerr=nominal[std_col].fillna(0), color=color, alpha=0.78)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=70)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_sensitivity(sens, out_png):
    pivot = sens.pivot_table(index="beta_latency", columns="beta_budget", values="joint_sla_rate")
    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c:.2f}" for c in pivot.index])
    ax.set_xlabel("beta_budget")
    ax.set_ylabel("beta_latency")
    ax.set_title("Reward sensitivity: Joint SLA")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_stress(stress, out_png):
    subset = stress[stress["strategy"].isin(["q_learning", "linucb", "hysteresis_rule", "dynamic_budget_q8", "greedy_sla"])]
    order = ["nominal", "rtt_x1_5", "burst_30", "burst_40_loss", "tight_deadline"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    for strategy, group in subset.groupby("strategy"):
        group = group.set_index("scenario").reindex(order).reset_index()
        ax.plot(group["scenario"], group["joint_sla_rate_mean"], marker="o", label=strategy)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Joint SLA")
    ax.set_xlabel("Network stress scenario")
    ax.set_title("CWSN-style network/burst stress test")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_markdown(out_path, agg, stats, ablation, sens, stress, episodes, eval_steps, seeds, main_spec):
    nominal = agg[agg["scenario"] == "nominal"].copy()
    preferred = ["oracle_reward", "q_learning", "linucb", "greedy_sla", "hysteresis_rule", "dynamic_budget_q8"]
    nominal = nominal[nominal["strategy"].isin(preferred)]
    lines = [
        "# AdaInfer-RL validation suite",
        "",
        "This file is generated from the measured Jetson profiling tables and the RL validation suite.",
        "",
        "## Setup",
        "",
        f"- Seeds: {len(seeds)} ({', '.join(map(str, seeds))})",
        f"- Training episodes per seed: {episodes}",
        f"- Evaluation requests per seed: {eval_steps}",
        f"- Main reward: {main_spec.name}, quality_gate={main_spec.quality_gate}, "
        f"beta_budget={main_spec.beta_budget:.2f}, beta_latency={main_spec.beta_latency:.2f}, "
        f"beta_switch={main_spec.beta_switch:.2f}.",
        "- State: quality bucket, requested budget, network pressure bucket, burst flag, resident model.",
        "- Actions: model precision and generation budget.",
        "- Baselines: oracle reward upper bound, LinUCB, SLA greedy, hysteresis, dynamic-budget Q8, static policies.",
        "",
        "## Nominal multi-seed result",
        "",
        "| Strategy | Joint SLA | Mean latency(ms) | Energy(J) | Budget retention | Switches/100 | Reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in nominal.sort_values("joint_sla_rate_mean", ascending=False).iterrows():
        lines.append(
            f"| {row['strategy']} | {row['joint_sla_rate_mean']:.3f}+/-{row['joint_sla_rate_std']:.3f} | "
            f"{row['mean_total_latency_ms_mean']:.1f}+/-{row['mean_total_latency_ms_std']:.1f} | "
            f"{row['mean_infer_energy_j_mean']:.2f}+/-{row['mean_infer_energy_j_std']:.2f} | "
            f"{row['mean_budget_retention_mean']:.3f}+/-{row['mean_budget_retention_std']:.3f} | "
            f"{row['switch_rate_per_100_mean']:.2f}+/-{row['switch_rate_per_100_std']:.2f} | "
            f"{row['mean_rl_reward_mean']:.3f}+/-{row['mean_rl_reward_std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Approximate significance tests",
            "",
            "| Baseline | Metric | RL mean | Baseline mean | Delta | Approx. p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    keep = stats[stats["baseline"].isin(["linucb", "greedy_sla", "hysteresis_rule", "dynamic_budget_q8"])]
    for _, row in keep.iterrows():
        lines.append(
            f"| {row['baseline']} | {row['metric']} | {row['rl_mean']:.4f} | "
            f"{row['baseline_mean']:.4f} | {row['delta_rl_minus_baseline']:.4f} | {row['approx_two_sided_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reward ablation",
            "",
            "| Ablation | Joint SLA | Latency(ms) | Energy(J) | Reward |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in ablation.iterrows():
        lines.append(
            f"| {row['reward_spec']} | {row['joint_sla_rate']:.3f} | {row['mean_total_latency_ms']:.1f} | "
            f"{row['mean_infer_energy_j']:.2f} | {row['mean_rl_reward']:.3f} |"
        )
    best = sens.sort_values("joint_sla_rate", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            "## Beta sensitivity",
            "",
            f"Best grid point by joint SLA: beta_budget={best['beta_budget']:.2f}, "
            f"beta_latency={best['beta_latency']:.2f}, joint SLA={best['joint_sla_rate']:.3f}, "
            f"latency={best['mean_total_latency_ms']:.1f} ms.",
            "",
            "## Network stress",
            "",
            "| Scenario | Strategy | Joint SLA | Latency(ms) | Switches/100 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    stress_keep = stress[stress["strategy"].isin(["q_learning", "linucb", "greedy_sla", "hysteresis_rule", "dynamic_budget_q8"])]
    for _, row in stress_keep.sort_values(["scenario", "strategy"]).iterrows():
        lines.append(
            f"| {row['scenario']} | {row['strategy']} | {row['joint_sla_rate_mean']:.3f} | "
            f"{row['mean_total_latency_ms_mean']:.1f} | {row['switch_rate_per_100_mean']:.2f} |"
        )
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--raw-q4", required=True)
    parser.add_argument("--raw-q8", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--ablation-episodes", type=int)
    parser.add_argument("--sensitivity-episodes", type=int)
    parser.add_argument("--sensitivity-seeds", type=int, default=4)
    parser.add_argument("--main-reward-name", default="ada_rl_sla_latency")
    parser.add_argument("--main-beta-budget", type=float, default=0.40)
    parser.add_argument("--main-beta-latency", type=float, default=0.15)
    parser.add_argument("--main-beta-switch", type=float, default=0.05)
    parser.add_argument("--main-no-quality-gate", action="store_true")
    parser.add_argument("--q-alpha", type=float, default=0.18)
    parser.add_argument("--q-gamma", type=float, default=0.85)
    parser.add_argument("--q-eps-start", type=float, default=0.35)
    parser.add_argument("--q-eps-end", type=float, default=0.04)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    ablation_episodes = args.ablation_episodes or max(80, args.episodes // 2)
    sensitivity_episodes = args.sensitivity_episodes or max(50, args.episodes // 3)
    main_spec = RewardSpec(
        args.main_reward_name,
        quality_gate=not args.main_no_quality_gate,
        beta_budget=args.main_beta_budget,
        beta_latency=args.main_beta_latency,
        beta_switch=args.main_beta_switch,
    )
    latency_mean = base.load_latency(args.core)
    latency_samples = base.load_latency_samples(args.raw_q4, args.raw_q8)

    strategies = [
        "static_q4",
        "static_q8",
        "dynamic_budget_q8",
        "hysteresis_rule",
        "greedy_latency",
        "greedy_quality",
        "greedy_sla",
        "linucb",
        "oracle_reward",
        "q_learning",
    ]
    scenarios = {
        "nominal": {},
        "rtt_x1_5": {"rtt_scale": 1.5},
        "burst_30": {"burst_ratio": 0.30},
        "burst_40_loss": {"burst_ratio": 0.40, "loss_rate": 0.06},
        "tight_deadline": {"deadline_scale": 0.82, "burst_ratio": 0.30},
    }

    all_eval_rows = []
    all_diag = []
    all_visits = Counter()
    trained = {}
    for seed in seeds:
        q_table, diag, visits = train_q_learning(
            latency_mean,
            latency_samples,
            episodes=args.episodes,
            steps=args.train_steps,
            seed=seed,
            spec=main_spec,
            alpha=args.q_alpha,
            gamma=args.q_gamma,
            eps_start=args.q_eps_start,
            eps_end=args.q_eps_end,
        )
        linucb = train_linucb(latency_mean, latency_samples, args.episodes, args.train_steps, seed, main_spec)
        trained[seed] = (q_table, linucb)
        all_diag.append(diag)
        all_visits.update(visits)
        for scenario, kwargs in scenarios.items():
            trace = make_trace(args.eval_steps, seed + 10000, **kwargs)
            for offset, strategy in enumerate(strategies):
                rows = evaluate_strategy(
                    strategy,
                    trace,
                    latency_mean,
                    latency_samples,
                    seed + 20000 + offset,
                    q_table=q_table,
                    linucb=linucb,
                    spec=main_spec,
                )
                for row in rows:
                    row["seed"] = seed
                    row["scenario"] = scenario
                all_eval_rows.extend(rows)

    eval_frame = pd.DataFrame(all_eval_rows)
    seed_summary = summarize(all_eval_rows)
    agg = aggregate(seed_summary)
    stats = stats_table(seed_summary)
    diag_frame = pd.concat(all_diag, ignore_index=True)

    ablation_rows = []
    ablation_specs = [
        main_spec,
        RewardSpec(
            f"{main_spec.name}_no_quality_gate",
            quality_gate=False,
            beta_budget=main_spec.beta_budget,
            beta_latency=main_spec.beta_latency,
            beta_switch=main_spec.beta_switch,
        ),
        RewardSpec(
            f"{main_spec.name}_no_switch_penalty",
            beta_budget=main_spec.beta_budget,
            beta_latency=main_spec.beta_latency,
            beta_switch=0.0,
        ),
        RewardSpec("balanced_budget_reward", beta_budget=0.40, beta_latency=0.15, beta_switch=0.05),
    ]
    for spec in ablation_specs:
        rows = []
        for seed in seeds:
            q_table, _, _ = train_q_learning(
                latency_mean,
                latency_samples,
                episodes=ablation_episodes,
                steps=args.train_steps,
                seed=seed + 50000,
                spec=spec,
                alpha=args.q_alpha,
                gamma=args.q_gamma,
                eps_start=args.q_eps_start,
                eps_end=args.q_eps_end,
            )
            trace = make_trace(args.eval_steps, seed + 60000)
            eval_rows = evaluate_strategy(
                "q_learning",
                trace,
                latency_mean,
                latency_samples,
                seed + 61000,
                q_table=q_table,
                spec=spec,
            )
            for row in eval_rows:
                row["seed"] = seed
                row["scenario"] = spec.name
            rows.extend(eval_rows)
        tmp = summarize(rows)
        mean_row = tmp.groupby("scenario", as_index=False).mean(numeric_only=True).iloc[0].to_dict()
        mean_row["reward_spec"] = spec.name
        ablation_rows.append(mean_row)
    ablation = pd.DataFrame(ablation_rows)

    sens_rows = []
    for beta_budget in [0.00, 0.20, 0.40, 0.60, 0.80]:
        for beta_latency in [0.05, 0.15, 0.30, 0.45]:
            spec = RewardSpec(f"bb{beta_budget:.2f}_bl{beta_latency:.2f}", beta_budget=beta_budget, beta_latency=beta_latency)
            rows = []
            for seed in seeds[: args.sensitivity_seeds]:
                q_table, _, _ = train_q_learning(
                    latency_mean,
                    latency_samples,
                    episodes=sensitivity_episodes,
                    steps=args.train_steps,
                    seed=seed + int(beta_budget * 1000) + int(beta_latency * 10000),
                    spec=spec,
                    alpha=args.q_alpha,
                    gamma=args.q_gamma,
                    eps_start=args.q_eps_start,
                    eps_end=args.q_eps_end,
                )
                trace = make_trace(args.eval_steps, seed + 70000)
                eval_rows = evaluate_strategy(
                    "q_learning",
                    trace,
                    latency_mean,
                    latency_samples,
                    seed + 71000,
                    q_table=q_table,
                    spec=spec,
                )
                for row in eval_rows:
                    row["seed"] = seed
                    row["scenario"] = spec.name
                rows.extend(eval_rows)
            tmp = summarize(rows)
            sens_rows.append(
                {
                    "beta_budget": beta_budget,
                    "beta_latency": beta_latency,
                    "joint_sla_rate": tmp["joint_sla_rate"].mean(),
                    "mean_total_latency_ms": tmp["mean_total_latency_ms"].mean(),
                    "mean_infer_energy_j": tmp["mean_infer_energy_j"].mean(),
                    "mean_budget_retention": tmp["mean_budget_retention"].mean(),
                    "mean_rl_reward": tmp["mean_rl_reward"].mean(),
                }
            )
    sens = pd.DataFrame(sens_rows)

    eval_frame.to_csv(out_dir / "rl_validation_requests_20260514.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(out_dir / "rl_validation_seed_summary_20260514.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(out_dir / "rl_validation_aggregate_20260514.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(out_dir / "rl_validation_stats_20260514.csv", index=False, encoding="utf-8-sig")
    diag_frame.to_csv(out_dir / "rl_training_diagnostics_20260514.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"state": str(state), "action": f"{action[0]}::{action[1]}", "visits": count}
            for (state, action), count in all_visits.items()
        ]
    ).to_csv(out_dir / "rl_state_action_visits_20260514.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(out_dir / "rl_reward_ablation_20260514.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(out_dir / "rl_reward_sensitivity_grid_20260514.csv", index=False, encoding="utf-8-sig")

    plot_learning_curve(diag_frame, out_dir / "rl_training_curve_20260514.png")
    plot_visit_heatmap(all_visits, out_dir / "rl_state_action_heatmap_20260514.png")
    plot_baseline_bars(agg, out_dir / "rl_baseline_comparison_20260514.png")
    plot_sensitivity(sens, out_dir / "rl_reward_sensitivity_heatmap_20260514.png")
    plot_stress(agg, out_dir / "rl_network_stress_20260514.png")
    write_markdown(
        out_dir / "rl_validation_report_20260514.md",
        agg,
        stats,
        ablation,
        sens,
        agg,
        args.episodes,
        args.eval_steps,
        seeds,
        main_spec,
    )
    print(out_dir / "rl_validation_report_20260514.md")


if __name__ == "__main__":
    main()
