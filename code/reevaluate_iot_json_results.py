#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from bench_iot_json_tasks import extract_json, score_json, load_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tasks = {item["id"]: item for item in load_tasks(args.tasks)}
    with open(args.input, "r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    for name in ["schema_valid", "field_score", "correct_fields", "total_fields", "parse_error", "parsed_json"]:
        if name not in fieldnames:
            fieldnames.append(name)

    for row in rows:
        raw = row.get("raw_output", "").replace("\\n", "\n")
        task = tasks[row["task_id"]]
        parsed, error = extract_json(raw)
        score, correct, total, _ = score_json(parsed, task["expected"])
        row["schema_valid"] = int(parsed is not None)
        row["field_score"] = score
        row["correct_fields"] = correct
        row["total_fields"] = total
        row["parse_error"] = error
        row["parsed_json"] = json.dumps(parsed, ensure_ascii=False) if parsed is not None else ""

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
