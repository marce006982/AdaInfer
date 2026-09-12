#!/usr/bin/env python3
"""Measure process-level model load plus the first generation on the RTX 5060 Ti."""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def smi():
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total,power.draw,temperature.gpu", "--format=csv,noheader,nounits"]
        p = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip().split(",")
        return {"gpu_mem_used_mb": float(p[0]), "gpu_mem_total_mb": float(p[1]), "power_w": float(p[2]), "temperature_c": float(p[3])}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    prompt = json.loads(Path(args.prompt).read_text(encoding="utf-8").splitlines()[0])["prompt"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["repeat", "model_load_ms", "first_generate_ms", "total_cold_ms", "input_tokens", "output_tokens", "peak_torch_memory_mb", "gpu_mem_used_mb", "gpu_mem_total_mb", "power_w", "temperature_c"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for repeat in range(args.repeats):
            start = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, local_files_only=args.local_files_only, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            load_ms = (time.perf_counter() - start) * 1000.0
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            gen_start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generate_ms = (time.perf_counter() - gen_start) * 1000.0
            row = {"repeat": repeat + 1, "model_load_ms": load_ms, "first_generate_ms": generate_ms, "total_cold_ms": load_ms + generate_ms, "input_tokens": int(inputs["input_ids"].shape[-1]), "output_tokens": int(output.shape[-1] - inputs["input_ids"].shape[-1]), "peak_torch_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0}
            row.update(smi()); writer.writerow(row); f.flush(); print(json.dumps(row))
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
