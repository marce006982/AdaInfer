#!/usr/bin/env python3
"""Run a reproducible local MMLU MCQ quality check from cached Arrow files."""
import argparse, csv, json, re, time
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELLED_ANSWER_RE = re.compile(r"\b(?:answer|choice)\s*[:=]?\s*[\"']?\s*([A-D])\b", re.I)
STANDALONE_ANSWER_RE = re.compile(r"\b([A-D])\b", re.I)

def parse_answer(text):
    raw = text.strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            answer = next((v for key, v in value.items() if str(key).lower() == "answer"), "")
            if isinstance(answer, str) and answer.strip().upper()[:1] in "ABCD":
                return answer.strip().upper()[:1]
    labelled = LABELLED_ANSWER_RE.search(raw)
    if labelled:
        return labelled.group(1).upper()
    standalone = STANDALONE_ANSWER_RE.search(raw)
    return standalone.group(1).upper() if standalone else ""

def load_cached_questions(cache_root, max_questions):
    """Load only MMLU Arrow test files already present on this machine.

    The local datasets version does not accept ``local_files_only`` for MMLU.
    Reading the Arrow cache directly makes the offline input explicit and avoids
    an accidental download or hidden dataset revision change.
    """
    cache_root = Path(cache_root)
    paths = sorted(cache_root.glob("*/*/*/mmlu-test.arrow"))
    if not paths:
        raise FileNotFoundError(f"No cached MMLU test Arrow files under {cache_root}")
    datasets = [Dataset.from_file(str(path)) for path in paths]
    available = sum(len(dataset) for dataset in datasets)
    if max_questions > available:
        raise ValueError(f"Requested {max_questions} questions, but cache contains {available}")
    # Round-robin sampling avoids allowing one cached subject to dominate.
    rows = []
    offsets = [0] * len(datasets)
    while len(rows) < max_questions:
        made_progress = False
        for dataset_index, dataset in enumerate(datasets):
            if offsets[dataset_index] < len(dataset) and len(rows) < max_questions:
                rows.append(dataset[offsets[dataset_index]])
                offsets[dataset_index] += 1
                made_progress = True
        if not made_progress:
            break
    return rows, [path.parent.parent.parent.name for path in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-questions", type=int, default=100); ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--mmlu-cache", default=str(Path.home() / ".cache" / "huggingface" / "datasets" / "cais___mmlu"))
    ap.add_argument("--no-chat-template", action="store_true")
    args = ap.parse_args()
    rows, cached_subjects = load_cached_questions(args.mmlu_cache, args.max_questions)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, local_files_only=args.local_files_only, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval(); device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["question_id", "subject", "gold", "pred", "correct", "latency_ms", "prompt_tokens", "protocol", "raw_output"]
    correct = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for idx, item in enumerate(rows):
            choices = item["choices"]
            prompt = (
                "Answer this multiple-choice question. Return a JSON object with exactly one field, "
                'for example {"answer":"C"}. The value must be only A, B, C, or D.\\n\\n'
                "Question: " + item["question"] + "\\n" +
                "\\n".join(f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices))
            )
            if args.no_chat_template:
                rendered_prompt = prompt
                protocol = "bare_prompt_json_answer_greedy"
            else:
                rendered_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                protocol = "qwen3_chat_template_nonthinking_json_answer_greedy"
            inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            latency = (time.perf_counter()-start)*1000
            raw = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            pred = parse_answer(raw); ok = int(pred == chr(65 + int(item["answer"])))
            correct += ok; w.writerow({"question_id": idx, "subject": item["subject"], "gold": chr(65 + int(item["answer"])), "pred": pred, "correct": ok, "latency_ms": latency, "prompt_tokens": inputs["input_ids"].shape[-1], "protocol": protocol, "raw_output": raw.replace("\n", "\\n")}); f.flush()
    print(json.dumps({"n": len(rows), "accuracy": correct / len(rows) if rows else 0.0,
                      "cached_subjects": cached_subjects}, ensure_ascii=False))

if __name__ == "__main__": main()
