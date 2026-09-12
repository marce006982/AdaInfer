#!/usr/bin/env python3
import argparse
import csv
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


MODELS = ("Q4", "Q8")
BUDGETS = (16, 32, 64, 128)
ACTIONS = [(model, budget) for model in MODELS for budget in BUDGETS]
Q_SLA = (0.45, 0.65)
NETWORK_BUCKETS = (0, 1, 2)
BURSTS = (0, 1)


def perf_us():
    return time.perf_counter() * 1_000_000.0


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def deep_size(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += deep_size(key, seen) + deep_size(value, seen)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            size += deep_size(value, seen)
    return size


def profile_latency(action, network_bucket, burst):
    model, budget = action
    base = {
        ("Q4", 16): 3150.0,
        ("Q4", 32): 4490.0,
        ("Q4", 64): 8040.0,
        ("Q4", 128): 14370.0,
        ("Q8", 16): 3590.0,
        ("Q8", 32): 6020.0,
        ("Q8", 64): 10020.0,
        ("Q8", 128): 19180.0,
    }[action]
    return base * (1.0 + 0.10 * network_bucket + 0.15 * burst)


def profile_quality(action):
    return 0.455 if action[0] == "Q4" else 0.711


def feasible_actions(requested_budget):
    return [action for action in ACTIONS if action[1] <= requested_budget]


def random_state(rng):
    q_sla = rng.choice(Q_SLA)
    requested_budget = rng.choice(BUDGETS)
    network_bucket = rng.choice(NETWORK_BUCKETS)
    burst = rng.choice(BURSTS)
    resident = rng.choice(MODELS)
    return (q_sla, requested_budget, network_bucket, burst, resident)


def context_vector(state, action):
    q_sla, requested_budget, network_bucket, burst, resident = state
    model, budget = action
    return [
        1.0,
        q_sla,
        requested_budget / 128.0,
        network_bucket / 2.0,
        float(burst),
        float(resident == model),
        float(model == "Q8"),
        budget / 128.0,
        profile_latency(action, network_bucket, burst) / 20000.0,
    ]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mat_vec(mat, vec):
    return [dot(row, vec) for row in mat]


def outer(a, b):
    return [[x * y for y in b] for x in a]


def identity(dim):
    return [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]


class GreedyController:
    name = "greedy_sla"

    def __init__(self):
        self.memory = ACTIONS

    def select(self, state):
        q_sla, requested_budget, network_bucket, burst, _ = state
        feasible = feasible_actions(requested_budget)
        ok = [a for a in feasible if profile_quality(a) >= q_sla]
        candidates = ok if ok else feasible
        return min(candidates, key=lambda a: profile_latency(a, network_bucket, burst))

    def update(self, state, action, reward):
        return None


class QController:
    name = "tabular_q_learning"

    def __init__(self, rng):
        self.q = defaultdict(float)
        for q_sla in Q_SLA:
            for requested_budget in BUDGETS:
                for network_bucket in NETWORK_BUCKETS:
                    for burst in BURSTS:
                        for resident in MODELS:
                            state = (q_sla, requested_budget, network_bucket, burst, resident)
                            for action in feasible_actions(requested_budget):
                                self.q[(state, action)] = rng.uniform(-0.05, 0.05)

    @property
    def memory(self):
        return self.q

    def select(self, state):
        return max(feasible_actions(state[1]), key=lambda action: self.q[(state, action)])

    def update(self, state, action, reward):
        self.q[(state, action)] += 0.1 * (reward - self.q[(state, action)])


class LinUCBController:
    name = "linucb"

    def __init__(self, alpha=0.45, dim=9):
        self.alpha = alpha
        self.dim = dim
        self.a_inv = {action: identity(dim) for action in ACTIONS}
        self.b = {action: [0.0] * dim for action in ACTIONS}

    @property
    def memory(self):
        return {"a_inv": self.a_inv, "b": self.b}

    def select(self, state):
        best_action = None
        best_score = -1e9
        for action in feasible_actions(state[1]):
            x = context_vector(state, action)
            a_inv_x = mat_vec(self.a_inv[action], x)
            theta = mat_vec(self.a_inv[action], self.b[action])
            bonus = self.alpha * max(0.0, dot(x, a_inv_x)) ** 0.5
            score = dot(theta, x) + bonus
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def update(self, state, action, reward):
        x = context_vector(state, action)
        a_inv = self.a_inv[action]
        a_inv_x = mat_vec(a_inv, x)
        denom = 1.0 + dot(x, a_inv_x)
        correction = outer(a_inv_x, a_inv_x)
        for i in range(self.dim):
            for j in range(self.dim):
                a_inv[i][j] -= correction[i][j] / denom
        for i in range(self.dim):
            self.b[action][i] += reward * x[i]


def reward_for(state, action):
    q_sla, _, network_bucket, burst, resident = state
    quality_ok = 1.0 if profile_quality(action) >= q_sla else -1.0
    latency_penalty = profile_latency(action, network_bucket, burst) / 20000.0
    switch_penalty = 0.03 if resident != action[0] else 0.0
    return quality_ok - latency_penalty - switch_penalty


def benchmark(controller, states):
    select_us = []
    update_us = []
    total_us = []
    for state in states:
        start = perf_us()
        action = controller.select(state)
        mid = perf_us()
        controller.update(state, action, reward_for(state, action))
        end = perf_us()
        select_us.append(mid - start)
        update_us.append(end - mid)
        total_us.append(end - start)
    return {
        "strategy": controller.name,
        "decision_mean_us": statistics.mean(select_us),
        "decision_p95_us": percentile(select_us, 0.95),
        "update_mean_us": statistics.mean(update_us),
        "update_p95_us": percentile(update_us, 0.95),
        "total_mean_us": statistics.mean(total_us),
        "total_p95_us": percentile(total_us, 0.95),
        "total_p99_us": percentile(total_us, 0.99),
        "memory_kb": deep_size(controller.memory) / 1024.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--platform", default="local")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    states = [random_state(rng) for _ in range(args.iterations)]
    controllers = [GreedyController(), QController(random.Random(args.seed + 1)), LinUCBController()]
    rows = []
    for controller in controllers:
        row = benchmark(controller, states)
        row.update({"platform": args.platform, "iterations": args.iterations})
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "platform",
        "strategy",
        "iterations",
        "decision_mean_us",
        "decision_p95_us",
        "update_mean_us",
        "update_p95_us",
        "total_mean_us",
        "total_p95_us",
        "total_p99_us",
        "memory_kb",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
