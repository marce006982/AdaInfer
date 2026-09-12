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


NORMAL_QUALITY = {
    "Qwen2.5-0.5B Q4": 0.455,
    "Qwen3-0.6B Q8": 0.711,
}

def load_rtt_trace(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    values = [float(v) for v in frame["rtt_ms"].dropna().tolist()]
    if not values:
        values = [2.0]
    return values


def make_measured_trace(n, seed, rtt_values, shift_step, rtt_multiplier, post_shift_rtt_multiplier=1.0):
    rng = random.Random(seed)
    rows = []
    for idx in range(n):
        phase = idx % 120
        high_critical = 80 <= phase < 100
        q_sla = 0.65 if high_critical else 0.45
        requested_budget = rng.choices([32, 64, 128], weights=[0.35, 0.45, 0.20])[0]
        rtt = rtt_values[idx % len(rtt_values)] * rtt_multiplier
        if idx >= shift_step:
            rtt *= post_shift_rtt_multiplier
        queue = rng.expovariate(1 / 180.0)
        # Keep the network-state pattern fixed across phases so the injected
        # disturbance is limited to the Q8 compute profile.
        burst = (idx // 50) % 4 == 2
        rows.append(
            {
                "request_id": idx,
                "wireless_ms": rtt,
                "queue_ms": queue,
                "q_sla": q_sla,
                "requested_budget": requested_budget,
                "deadline_ms": 12000.0 if high_critical else 9000.0,
                "burst": int(burst),
                "criticality": "high" if high_critical else "low",
                "phase": "post_shift" if idx >= shift_step else "pre_shift",
            }
        )
    return rows


def step_eval_quality(request, action, current_model, latency_ms, quality, spec):
    model, budget = action
    switch = int(model != current_model)
    total = request["wireless_ms"] + request["queue_ms"] + latency_ms + switch * base.SWITCH_COST[model]
    quality_ok = quality[model] >= request["q_sla"]
    latency_ok = total <= request["deadline_ms"]
    reward = (
        int(quality_ok and latency_ok)
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
        "quality": quality[model],
        "budget_retention": budget / request["requested_budget"],
        "quality_sla_met": int(quality_ok),
        "latency_sla_met": int(latency_ok),
        "joint_sla_met": int(quality_ok and latency_ok),
        "rl_reward": reward,
    }


def stale_greedy_sla(request, current_model, latency_mean):
    feasible = vrl.feasible_actions(request)
    quality_feasible = [a for a in feasible if NORMAL_QUALITY[a[0]] >= request["q_sla"]]
    pool = quality_feasible or feasible
    deadline_feasible = [
        a for a in pool if vrl.expected_total(request, a, current_model, latency_mean) <= request["deadline_ms"]
    ]
    pool = deadline_feasible or pool
    return min(pool, key=lambda a: vrl.expected_total(request, a, current_model, latency_mean))


def greedy_sla_from_latency_estimates(request, current_model, latency_estimates):
    feasible = vrl.feasible_actions(request)
    quality_feasible = [a for a in feasible if NORMAL_QUALITY[a[0]] >= request["q_sla"]]
    pool = quality_feasible or feasible

    def expected_total(action):
        model, _ = action
        return (
            request["wireless_ms"]
            + request["queue_ms"]
            + latency_estimates[action]
            + int(model != current_model) * base.SWITCH_COST[model]
        )

    deadline_feasible = [a for a in pool if expected_total(a) <= request["deadline_ms"]]
    pool = deadline_feasible or pool
    return min(pool, key=expected_total)


def actual_expected_latency(latency_mean, action, q8_multiplier=1.0):
    value = latency_mean[action]
    if action[0] == "Qwen3-0.6B Q8":
        value *= q8_multiplier
    return value


def true_oracle(request, current_model, latency_mean, quality, spec, q8_multiplier=1.0):
    return max(
        vrl.feasible_actions(request),
        key=lambda a: step_eval_quality(
            request,
            a,
            current_model,
            actual_expected_latency(latency_mean, a, q8_multiplier),
            quality,
            spec,
        )["rl_reward"],
    )


class ResetLinUCB:
    def __init__(self, learner=None, window=40, threshold=0.68, max_resets=1):
        self.learner = learner or vrl.DisjointLinUCB(dim=9, alpha=0.45)
        self.window = window
        self.threshold = threshold
        self.max_resets = max_resets
        self.recent_rewards = []
        self.resets = 0

    def choose(self, request, current_model, explore=True):
        return self.learner.choose(request, current_model, explore=explore)

    def update(self, request, current_model, action, reward):
        self.learner.update(request, current_model, action, reward)
        self.recent_rewards.append(float(reward))
        if len(self.recent_rewards) > self.window:
            self.recent_rewards = self.recent_rewards[-self.window :]
        if (
            self.resets < self.max_resets
            and len(self.recent_rewards) == self.window
            and float(np.mean(self.recent_rewards)) < self.threshold
        ):
            self.learner = vrl.DisjointLinUCB(dim=9, alpha=0.45)
            self.recent_rewards = []
            self.resets += 1


def evaluate_adaptation(
    strategy,
    trace,
    latency_mean,
    latency_samples,
    seed,
    q_table,
    linucb,
    adaptive_linucb,
    spec,
    shift_step,
    q8_shift_multiplier,
):
    rng = random.Random(seed)
    rows = []
    current_model = "Qwen2.5-0.5B Q4"
    epsilon = 0.04
    alpha = 0.20
    gamma = 0.20
    ewma_alpha = 0.25
    ewma_latency = dict(latency_mean)
    for idx, request in enumerate(trace):
        quality = NORMAL_QUALITY
        if strategy in ["q_online", "q_frozen"]:
            state = vrl.state_key(request, current_model)
            feasible = vrl.feasible_actions(request)
            if strategy == "q_online" and rng.random() < epsilon:
                action = rng.choice(feasible)
            else:
                action = max(feasible, key=lambda a: q_table[(state, a)])
        elif strategy == "linucb_online":
            action = linucb.choose(request, current_model, explore=True)
        elif strategy == "adaptive_linucb":
            action = adaptive_linucb.choose(request, current_model, explore=True)
        elif strategy == "greedy_sla_stale":
            action = stale_greedy_sla(request, current_model, latency_mean)
        elif strategy == "greedy_sla_ewma":
            action = greedy_sla_from_latency_estimates(request, current_model, ewma_latency)
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
        elif strategy == "oracle_adaptive":
            q8_multiplier = q8_shift_multiplier if idx >= shift_step else 1.0
            action = true_oracle(request, current_model, latency_mean, quality, spec, q8_multiplier)
        else:
            raise ValueError(strategy)

        latency_ms = base.draw_latency(latency_samples, latency_mean, action, rng)
        if idx >= shift_step and action[0] == "Qwen3-0.6B Q8":
            latency_ms *= q8_shift_multiplier
        outcome = step_eval_quality(request, action, current_model, latency_ms, quality, spec)
        if strategy == "q_online":
            state = vrl.state_key(request, current_model)
            next_model = action[0]
            if idx + 1 < len(trace):
                next_state = vrl.state_key(trace[idx + 1], next_model)
                next_actions = vrl.feasible_actions(trace[idx + 1])
                target = outcome["rl_reward"] + gamma * max(q_table[(next_state, a)] for a in next_actions)
            else:
                target = outcome["rl_reward"]
            q_table[(state, action)] += alpha * (target - q_table[(state, action)])
        elif strategy == "linucb_online":
            linucb.update(request, current_model, action, outcome["rl_reward"])
        elif strategy == "adaptive_linucb":
            adaptive_linucb.update(request, current_model, action, outcome["rl_reward"])
        elif strategy == "greedy_sla_ewma":
            ewma_latency[action] = (1.0 - ewma_alpha) * ewma_latency[action] + ewma_alpha * latency_ms
        resets = adaptive_linucb.resets if strategy == "adaptive_linucb" else 0
        rows.append({**request, **outcome, "strategy": strategy, "adaptive_resets": resets})
        current_model = action[0]
    return rows


def summarize(rows, shift_step, window, recovery_threshold=0.90):
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["seed", "strategy", "phase"], as_index=False)
        .agg(
            requests=("request_id", "count"),
            joint_sla_rate=("joint_sla_met", "mean"),
            mean_latency_ms=("total_latency_ms", "mean"),
            mean_energy_j=("infer_energy_j", "mean"),
            q8_action_rate=("model", lambda x: float((x == "Qwen3-0.6B Q8").mean())),
            reward=("rl_reward", "mean"),
        )
    )
    recover_rows = []
    for (seed, strategy), group in df.groupby(["seed", "strategy"]):
        group = group.sort_values("request_id").copy()
        post = group[group["request_id"] >= shift_step].copy()
        post["rolling_sla"] = post["joint_sla_met"].rolling(window, min_periods=window).mean()
        recovered = post[post["rolling_sla"] >= recovery_threshold]
        recovery = np.nan if recovered.empty else int(recovered.iloc[0]["request_id"] - shift_step + 1)
        recover_rows.append({"seed": seed, "strategy": strategy, "time_to_recover": recovery})
    return summary, pd.DataFrame(recover_rows)


def plot(rows, shift_step, out_png):
    df = pd.DataFrame(rows).sort_values(["strategy", "request_id"])
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.8), sharex=True, constrained_layout=True)
    for strategy, group in df.groupby("strategy"):
        by_req = group.groupby("request_id")
        sla_curve = by_req["joint_sla_met"].mean().rolling(40, min_periods=10).mean()
        q8_curve = by_req["model"].apply(lambda x: float((x == "Qwen3-0.6B Q8").mean())).rolling(40, min_periods=10).mean()
        axes[0].plot(sla_curve.index, sla_curve.values, label=strategy)
        axes[1].plot(q8_curve.index, q8_curve.values, label=strategy)
    for ax in axes:
        ax.axvline(shift_step, color="#111827", linestyle="--", linewidth=1.2)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[0].set_ylabel("Rolling joint SLA")
    axes[1].set_ylabel("Rolling Q8 action rate")
    axes[1].set_xlabel("Request")
    axes[0].set_title("Non-stationary Q8 latency shift adaptation")
    axes[0].legend(ncol=2, fontsize=8)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_md(out_md, summary_agg, recovery_agg, shift_step, rtt_count, q8_shift_multiplier):
    lines = [
        "# Non-stationary RL adaptation experiment",
        "",
        f"- Measured LAN RTT samples: {rtt_count}",
        f"- Latency shift at request: {shift_step}",
        f"- Shift: Qwen3-0.6B Q8 actual latency is multiplied by {q8_shift_multiplier:.1f} after the shift.",
        "- Greedy-SLA and dynamic-budget keep using stale offline latency; online Q-learning, vanilla LinUCB, reset LinUCB, and EWMA Greedy-SLA receive runtime feedback.",
        "",
        "## Phase metrics",
        "",
        "| Strategy | Phase | Joint SLA | Latency(ms) | Energy(J) | Q8 action rate | Reward |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_agg.sort_values(["phase", "strategy"]).iterrows():
        lines.append(
            f"| {row['strategy']} | {row['phase']} | {row['joint_sla_rate_mean']:.3f}+/-{row['joint_sla_rate_std']:.3f} | "
            f"{row['mean_latency_ms_mean']:.1f}+/-{row['mean_latency_ms_std']:.1f} | "
            f"{row['mean_energy_j_mean']:.2f}+/-{row['mean_energy_j_std']:.2f} | "
            f"{row['q8_action_rate_mean']:.3f}+/-{row['q8_action_rate_std']:.3f} | "
            f"{row['reward_mean']:.3f}+/-{row['reward_std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Recovery",
            "",
            "| Strategy | Time to rolling-SLA>=0.90 recovery (requests) |",
            "|---|---:|",
        ]
    )
    for _, row in recovery_agg.sort_values("time_to_recover_mean").iterrows():
        value = "not recovered" if pd.isna(row["time_to_recover_mean"]) else f"{row['time_to_recover_mean']:.1f}+/-{row['time_to_recover_std']:.1f}"
        lines.append(f"| {row['strategy']} | {value} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_summary(summary):
    pieces = []
    for metric in ["joint_sla_rate", "mean_latency_ms", "mean_energy_j", "q8_action_rate", "reward"]:
        part = summary.groupby(["strategy", "phase"])[metric].agg(["mean", "std"]).reset_index()
        part = part.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        pieces.append(part)
    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on=["strategy", "phase"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--raw-q4", required=True)
    parser.add_argument("--raw-q8", required=True)
    parser.add_argument("--rtt-trace", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--shift-step", type=int, default=500)
    parser.add_argument("--rtt-multiplier", type=float, default=80.0)
    parser.add_argument(
        "--post-shift-rtt-multiplier",
        type=float,
        default=1.0,
        help="Additional post-shift RTT multiplier; 1.0 keeps the shift compute-only.",
    )
    parser.add_argument("--q8-shift-multiplier", type=float, default=2.0)
    parser.add_argument("--adaptive-window", type=int, default=40)
    parser.add_argument("--adaptive-threshold", type=float, default=0.68)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    spec = vrl.RewardSpec("online_adaptation", beta_budget=0.0, beta_latency=0.45, beta_switch=0.05)
    rtt_values = load_rtt_trace(args.rtt_trace)
    latency_mean = base.load_latency(args.core)
    latency_samples = base.load_latency_samples(args.raw_q4, args.raw_q8)
    strategies = [
        "oracle_adaptive",
        "q_online",
        "q_frozen",
        "linucb_online",
        "adaptive_linucb",
        "greedy_sla_ewma",
        "greedy_sla_stale",
        "dynamic_budget_q8",
    ]
    rows = []
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
        linucb = vrl.train_linucb(latency_mean, latency_samples, args.episodes, args.train_steps, seed, spec)
        adaptive_linucb = ResetLinUCB(
            learner=vrl.train_linucb(latency_mean, latency_samples, args.episodes, args.train_steps, seed, spec),
            window=args.adaptive_window,
            threshold=args.adaptive_threshold,
        )
        trace = make_measured_trace(
            args.eval_steps,
            seed + 90000,
            rtt_values,
            args.shift_step,
            args.rtt_multiplier,
            args.post_shift_rtt_multiplier,
        )
        for idx, strategy in enumerate(strategies):
            q_copy = defaultdict(float, q_table.copy())
            eval_rows = evaluate_adaptation(
                strategy,
                trace,
                latency_mean,
                latency_samples,
                seed + 91000 + idx,
                q_copy,
                linucb,
                adaptive_linucb,
                spec,
                args.shift_step,
                args.q8_shift_multiplier,
            )
            for row in eval_rows:
                row["seed"] = seed
            rows.extend(eval_rows)

    frame = pd.DataFrame(rows)
    summary, recovery = summarize(rows, args.shift_step, window=args.adaptive_window)
    summary_agg = aggregate_summary(summary)
    recovery_agg = recovery.groupby("strategy")["time_to_recover"].agg(["mean", "std"]).reset_index()
    recovery_agg = recovery_agg.rename(columns={"mean": "time_to_recover_mean", "std": "time_to_recover_std"})
    frame.to_csv(out_dir / "rl_nonstationary_requests_20260514.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "rl_nonstationary_seed_summary_20260514.csv", index=False, encoding="utf-8-sig")
    summary_agg.to_csv(out_dir / "rl_nonstationary_summary_20260514.csv", index=False, encoding="utf-8-sig")
    recovery_agg.to_csv(out_dir / "rl_nonstationary_recovery_20260514.csv", index=False, encoding="utf-8-sig")
    plot(rows, args.shift_step, out_dir / "rl_nonstationary_adaptation_20260514.png")
    write_md(
        out_dir / "rl_nonstationary_adaptation_20260514.md",
        summary_agg,
        recovery_agg,
        args.shift_step,
        len(rtt_values),
        args.q8_shift_multiplier,
    )
    print(out_dir / "rl_nonstationary_adaptation_20260514.md")


if __name__ == "__main__":
    main()

