# AdaInfer

Code and data for **AdaInfer: Feedback-Based Runtime Configuration Control for Edge LLM Inference**.

AdaInfer is a lightweight runtime controller for edge LLM services. It jointly selects a model configuration and a feasible generation cap from the request target, runtime pressure, burst state, and resident configuration, then updates a compact tabular policy from observed quality and latency.

## Main results

- Matches the joint target satisfaction of dynamic-budget Qwen3/Q8 while reducing average latency by **32.4%** under request-level quality feedback.
- Improves joint target satisfaction by **15.5 percentage points** over mean-quality Greedy-SLA.
- After a two-fold inference-latency shift, reduces average latency by **12.9%** relative to EWMA Greedy-SLA while maintaining higher target satisfaction.
- Completes selection and policy update in **27.8 microseconds** on Jetson Nano.

## Repository layout

| Path | Contents |
| --- | --- |
| `code/` | Controller evaluation, robustness, profiling, and statistical-analysis scripts |
| `data/` | Measured inputs, compact experiment records, and summary statistics |
| `analysis/` | Figure and table regeneration script |
| `results/` | Published table fragments and compact result summaries |
| `docs/REPRODUCE.md` | Complete reproduction commands and experiment settings |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python code/simulate_rl_controller_trace.py \
  --core data/core_metrics_workload_steady_16_32_64_128_20260513.csv \
  --raw-q4 data/yahboom_qwen2_5_0_5b_q4_workload_16_32_64_128_20260513.csv \
  --raw-q8 data/yahboom_qwen3_0_6b_q8_workload_16_32_64_128_20260513.csv \
  --episodes 2 --steps 20 --seed 2026 \
  --out-requests output/smoke/requests.csv \
  --out-summary output/smoke/summary.csv \
  --out-md output/smoke/report.md
```

## Full request-level records

The `v1.0.0` release contains `AdaInfer_Full_Request-Level_Data_v1.0.0.zip`, which includes every request-level record used in the reported experiments. Extract the archive into the repository root so that the additional files are placed under `data/`.

The repository itself keeps the measured inputs and summary records required for inspection and lightweight reruns. Large generated request traces are kept in the release asset to avoid Git repository size limits.

## Reproduction

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the stationary evaluation, empirical-quality replay, noise robustness, compute-profile drift, controller overhead, and optional RTX 5060 Ti measurements.

To regenerate the data-derived figures and table fragments after installing the full data release:

```bash
python analysis/build_figures.py
```

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Until the accompanying article is published, cite this repository by title, authors, version, and release date.

## License

The source code is released under the MIT License. The data files are released under the Creative Commons Attribution 4.0 International License. See `LICENSE-CODE` and `LICENSE-DATA`.

