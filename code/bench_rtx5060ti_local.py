#!/usr/bin/env python3
"""Reproducible local RTX 5060 Ti profiling for HF causal LMs.

The benchmark keeps model loading separate from steady-state generation and
records enough metadata to avoid confusing GPU memory background usage with
Torch allocations. Input lengths are controlled at the token-id level so the
same workload can be replayed across budgets and batch sizes.
"""

import argparse
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def query_nvidia_smi():
    fields = ["name", "memory.used", "memory.total", "power.draw", "temperature.gpu", "utilization.gpu"]
    cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
        if len(parts) != len(fields):
            return None
        def number(value):
            return None if value == "[Not Supported]" else float(value)
        return {
            "gpu_name": parts[0],
            "gpu_mem_used_mb": number(parts[1]),
            "gpu_mem_total_mb": number(parts[2]),
            "power_w": number(parts[3]),
            "temperature_c": number(parts[4]),
            "gpu_util_pct": number(parts[5]),
        }
    except Exception:
        return None


class NvidiaSampler:
    def __init__(self, interval_s):
        self.interval_s = interval_s
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            sample = query_nvidia_smi()
            if sample:
                self.samples.append(sample)
            time.sleep(self.interval_s)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def summary(self):
        result = {"nvidia_samples": len(self.samples)}
        for key in ("gpu_mem_used_mb", "gpu_mem_total_mb", "power_w", "temperature_c", "gpu_util_pct"):
            values = [s[key] for s in self.samples if s.get(key) is not None]
            if values:
                result[f"{key}_avg"] = sum(values) / len(values)
                result[f"{key}_max"] = max(values)
        return result


def prepare_ids(tokenizer, prompt, target_tokens):
    """Return exactly target_tokens ids without padding-induced compute."""
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    if len(ids) >= target_tokens:
        return ids[:target_tokens]
    repeated = (ids * ((target_tokens + len(ids) - 1) // len(ids)))[:target_tokens]
    return repeated


def run_generation(model, tokenizer, device, prompt, input_tokens, max_new_tokens, batch_size, sampler_interval):
    ids = prepare_ids(tokenizer, prompt, input_tokens)
    input_ids = torch.tensor([ids] * batch_size, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with NvidiaSampler(sampler_interval) as sampler:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000.0
    generated = int(output.shape[-1] - input_tokens)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    row = {
        "wall_ms_batch": wall_ms,
        "wall_ms_per_request": wall_ms / batch_size,
        "input_tokens": input_tokens,
        "output_tokens_batch": generated * batch_size,
        "output_tokens_per_request": generated,
        "tokens_per_sec_per_request": generated / (wall_ms / 1000.0 / batch_size) if wall_ms else 0.0,
        "peak_torch_memory_mb": peak_mb,
    }
    row.update(sampler.summary())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device-label", default="RTX_5060_Ti_8GB")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--max-new-tokens", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--sampler-interval-s", type=float, default=0.2)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(args.prompts)[:args.max_prompts]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only, trust_remote_code=True)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=args.local_files_only, trust_remote_code=True
    ).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - load_start) * 1000.0
    model.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "device", "framework", "model", "dtype", "prompt_id", "run_label", "batch_size",
        "input_tokens", "max_new_tokens", "model_load_ms", "wall_ms_batch", "wall_ms_per_request",
        "output_tokens_batch", "output_tokens_per_request", "tokens_per_sec_per_request",
        "peak_torch_memory_mb", "gpu_mem_used_mb_avg", "gpu_mem_used_mb_max", "gpu_mem_total_mb_avg",
        "gpu_mem_total_mb_max",
        "power_w_avg", "power_w_max", "temperature_c_avg", "temperature_c_max", "gpu_util_pct_avg",
        "gpu_util_pct_max", "nvidia_samples",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for batch_size in args.batch_sizes:
            for input_tokens in args.input_tokens:
                for max_new_tokens in args.max_new_tokens:
                    for prompt in prompts:
                        for repeat in range(args.repeats):
                            row = run_generation(
                                model, tokenizer, device, prompt["prompt"], input_tokens,
                                max_new_tokens, batch_size, args.sampler_interval_s
                            )
                            row.update({
                                "device": args.device_label,
                                "framework": "Transformers",
                                "model": args.model,
                                "dtype": args.dtype,
                                "prompt_id": prompt["id"],
                                "run_label": "cold_after_load" if repeat == 0 else f"warm{repeat}",
                                "batch_size": batch_size,
                                "input_tokens": input_tokens,
                                "max_new_tokens": max_new_tokens,
                                "model_load_ms": model_load_ms,
                            })
                            writer.writerow(row)
                            f.flush()
                            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
