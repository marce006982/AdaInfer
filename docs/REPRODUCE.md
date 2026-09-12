# Reproducing AdaInfer

Run the commands below from the repository root. The compact repository contains all measured inputs and summary results. Request-level records are distributed with the `v1.0.0` GitHub release and should be extracted into `data/` before running the full analysis. Direct reruns to a new subdirectory of `output/`.

## Environment

The CPU/statistics environment was validated with Python 3.13.1:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the optional RTX scripts, install the CUDA build compatible with the machine’s NVIDIA driver and then use `requirements-rtx.txt`. The packaged run used PyTorch 2.11.0+cu130, Transformers 5.1.0, and Datasets 4.8.4.

## Regenerate manuscript figures and tables

From the repository root, run:

```bash
python analysis/build_figures.py
```

This regenerates Figures 3–5 and the generated table fragments from saved CSV records. Figures 1 and 2 are author-created assets and are not reconstructed from that script.

## Evidence boundary

The stationary Jetson Nano replay samples measured Qwen2.5/Q4 and Qwen3/Q8 inference-latency observations. Request budgets, deadlines, pressure, burst, loss, and RTT-scale variables are deterministic synthetic replay states. The stationary experiment does not consume the saved LAN RTT trace.

The non-stationary replay cycles through the saved LAN RTT values in both phases and injects a Qwen3/Q8 compute-latency multiplier after the shift. The RTX scripts are independent FP16 Qwen3-0.6B measurements; they are not an end-to-end RTX controller comparison.

## Stationary protocol

The manuscript run uses seeds 2026–2035, 600 training episodes, 1,000 evaluation requests per seed, reward `ada_rl_sla_latency_g02`, `beta_budget=0.00`, `beta_latency=0.45`, `beta_switch=0.05`, and Q-learning `gamma=0.20`.

```bash
python code/validate_rl_controller.py --core data/core_metrics_workload_steady_16_32_64_128_20260513.csv --raw-q4 data/yahboom_qwen2_5_0_5b_q4_workload_16_32_64_128_20260513.csv --raw-q8 data/yahboom_qwen3_0_6b_q8_workload_16_32_64_128_20260513.csv --out-dir output/stationary --episodes 600 --train-steps 500 --eval-steps 1000 --main-reward-name ada_rl_sla_latency_g02 --main-beta-budget 0.00 --main-beta-latency 0.45 --main-beta-switch 0.05 --q-alpha 0.18 --q-gamma 0.20 --q-eps-start 0.35 --q-eps-end 0.04 --seeds 2026,2027,2028,2029,2030,2031,2032,2033,2034,2035

python code/stress_bootstrap_stats.py --validation-seed-summary output/stationary/rl_validation_seed_summary_20260514.csv --out-dir output/network_stress_stats

python code/policy_conditioned_actions.py --requests output/stationary/rl_validation_requests_20260514.csv --out-csv output/policy_actions.csv --out-md output/policy_actions.md --out-png output/policy_actions.png
```

## Quality-feedback experiments

```bash
python code/quality_noise_robustness.py --core data/core_metrics_workload_steady_16_32_64_128_20260513.csv --raw-q4 data/yahboom_qwen2_5_0_5b_q4_workload_16_32_64_128_20260513.csv --raw-q8 data/yahboom_qwen3_0_6b_q8_workload_16_32_64_128_20260513.csv --out-dir output/quality_noise

python code/quality_asymmetric_noise.py --core data/core_metrics_workload_steady_16_32_64_128_20260513.csv --raw-q4 data/yahboom_qwen2_5_0_5b_q4_workload_16_32_64_128_20260513.csv --raw-q8 data/yahboom_qwen3_0_6b_q8_workload_16_32_64_128_20260513.csv --out-dir output/quality_asymmetric_noise

python code/empirical_quality_replay.py --core data/core_metrics_workload_steady_16_32_64_128_20260513.csv --raw-q4 data/yahboom_qwen2_5_0_5b_q4_workload_16_32_64_128_20260513.csv --raw-q8 data/yahboom_qwen3_0_6b_q8_workload_16_32_64_128_20260513.csv --quality-csv data/iot_json_qwen2_5_0_5b_q4_20260514.csv --quality-csv data/iot_json_qwen3_0_6b_q8_20260514.csv --quality-csv data/mcq_yahboom_qwen2_5_0_5b_q4_extended_20260513.csv --quality-csv data/mcq_yahboom_qwen3_0_6b_q8_extended_20260513.csv --out-dir output/empirical_quality
```

The two noise scripts use 300 training episodes, 500 steps per episode, 1,000 evaluation requests, and ten seeds unless overridden.

## Compute-profile drift and robustness

```bash
python code/nonstationary_rl_adaptation.py --core data/core_metrics_workload_steady_16_32_64_128_20260513.csv --raw-q4 data/yahboom_qwen2_5_0_5b_q4_workload_16_32_64_128_20260513.csv --raw-q8 data/yahboom_qwen3_0_6b_q8_workload_16_32_64_128_20260513.csv --rtt-trace data/yahboom_lan_rtt_trace_20260514.csv --out-dir output/nonstationary --episodes 500 --train-steps 500 --eval-steps 10000 --shift-step 5000 --rtt-multiplier 80 --post-shift-rtt-multiplier 1.0 --q8-shift-multiplier 2.0

python code/nonstationary_bootstrap_stats.py --nonstationary-seed-summary output/nonstationary/rl_nonstationary_seed_summary_20260514.csv --out-dir output/nonstationary

python code/nonstationary_robustness_sweep.py --core data/core_metrics_workload_steady_16_32_64_128_20260513.csv --raw-q4 data/yahboom_qwen2_5_0_5b_q4_workload_16_32_64_128_20260513.csv --raw-q8 data/yahboom_qwen3_0_6b_q8_workload_16_32_64_128_20260513.csv --rtt-trace data/yahboom_lan_rtt_trace_20260514.csv --out-dir output/nonstationary_robustness --episodes 500 --train-steps 500 --grid-eval-steps 5000 --timing-eval-steps 10000 --rtt-multiplier 80
```

The robustness sweep uses the same ten seeds, a 3×3 adaptive-LinUCB grid, and shifts at 20%, 50%, and 80% of the trace.

## Controller measurements and smoke test

```bash
python code/measure_controller_overhead.py --iterations 100000 --platform local --out output/controller_overhead.csv
python code/controller_policy_overhead.py --iterations 50000 --platform local --out output/controller_policy_overhead.csv

python code/simulate_rl_controller_trace.py --core data/core_metrics_workload_steady_16_32_64_128_20260513.csv --raw-q4 data/yahboom_qwen2_5_0_5b_q4_workload_16_32_64_128_20260513.csv --raw-q8 data/yahboom_qwen3_0_6b_q8_workload_16_32_64_128_20260513.csv --episodes 2 --steps 20 --seed 2026 --out-requests output/smoke/requests.csv --out-summary output/smoke/summary.csv --out-md output/smoke/report.md
```

The smoke test checks package wiring only; it is not a statistical reproduction.

## Optional RTX 5060 Ti validation

Set `QWEN3_06B_PATH` to a local Qwen3-0.6B checkpoint. The MMLU script also requires an existing local Hugging Face MMLU cache. `--local-files-only` prevents model downloads.

```bash
python code/bench_rtx5060ti_local.py --model "$QWEN3_06B_PATH" --prompts data/prompts_edge_workload_20260513.jsonl --out output/rtx/profiling.csv --input-tokens 128 512 1024 --max-new-tokens 16 64 128 --batch-sizes 1 2 --repeats 2 --max-prompts 4 --local-files-only
python code/measure_rtx_cold_start.py --model "$QWEN3_06B_PATH" --prompt "Summarize the edge deployment status." --out output/rtx/cold_start.csv --local-files-only
python code/bench_iot_json_tasks.py --backend hf --device-label RTX_5060_Ti_8GB --model "$QWEN3_06B_PATH" --tasks data/iot_structured_tasks_20260514.jsonl --out output/rtx/iot_quality.csv --local-files-only
python code/bench_rtx_mcq_100.py --model "$QWEN3_06B_PATH" --out output/rtx/mmlu_100.csv --local-files-only
```
