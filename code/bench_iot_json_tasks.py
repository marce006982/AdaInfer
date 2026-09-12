#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import time
import urllib.request
from pathlib import Path


def load_tasks(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(item):
    return (
        "You are an IoT gateway parser. Return only one compact JSON object. "
        "Do not include markdown, explanation, or extra text.\n"
        f"{item['prompt']}\nJSON:"
    )


def post_json(url, payload, timeout=600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def run_ollama(host, model, prompt, max_new_tokens):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": 0,
            "num_ctx": 1024,
        },
        "keep_alive": "10m",
    }
    start = time.perf_counter()
    with post_json(host.rstrip("/") + "/api/generate", payload) as resp:
        item = json.loads(resp.read().decode("utf-8"))
    end = time.perf_counter()
    return item.get("response", ""), (end - start) * 1000.0


def run_hf(model, tokenizer, device, prompt, max_new_tokens):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text, (end - start) * 1000.0


def extract_json(text):
    # Models may emit a valid object followed by an explanation or a second
    # retry. raw_decode accepts the first complete object and ignores trailing text.
    text = text.strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            if isinstance(value, dict):
                return value, ""
        except json.JSONDecodeError:
            continue
    return None, "no_valid_json_object"


def normalize(value):
    if isinstance(value, str):
        return value.strip().lower().replace("_", " ")
    return value


def value_match(pred, gold):
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return math.isclose(float(pred), float(gold), rel_tol=0.02, abs_tol=0.05)
    return normalize(pred) == normalize(gold)


def score_json(pred, expected):
    if not isinstance(pred, dict):
        return 0.0, 0, len(expected), []
    correct = 0
    details = []
    for key, gold in expected.items():
        ok = key in pred and value_match(pred[key], gold)
        correct += int(ok)
        details.append(f"{key}:{int(ok)}")
    return correct / len(expected) if expected else 0.0, correct, len(expected), details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "ollama"], required=True)
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    hf_model = None
    tokenizer = None
    device = None
    if args.backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        ).to(device)
        hf_model.eval()

    tasks = load_tasks(args.tasks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "device", "backend", "model", "task_id", "task_type", "schema_valid",
        "field_score", "correct_fields", "total_fields", "latency_ms",
        "parse_error", "raw_output", "parsed_json",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in tasks:
            prompt = build_prompt(item)
            if args.backend == "hf":
                raw, latency_ms = run_hf(hf_model, tokenizer, device, prompt, args.max_new_tokens)
            else:
                raw, latency_ms = run_ollama(args.host, args.model, prompt, args.max_new_tokens)
            parsed, error = extract_json(raw)
            score, correct, total, _ = score_json(parsed, item["expected"])
            row = {
                "device": args.device_label,
                "backend": args.backend,
                "model": args.model,
                "task_id": item["id"],
                "task_type": item["task_type"],
                "schema_valid": int(parsed is not None),
                "field_score": score,
                "correct_fields": correct,
                "total_fields": total,
                "latency_ms": latency_ms,
                "parse_error": error,
                "raw_output": raw.replace("\n", "\\n"),
                "parsed_json": json.dumps(parsed, ensure_ascii=False) if parsed is not None else "",
            }
            writer.writerow(row)
            f.flush()
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
