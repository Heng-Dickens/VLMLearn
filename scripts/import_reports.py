#!/usr/bin/env python3
"""Transcribe saved report tables; never call models or recompute historical AP."""
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments/archive"


def cells(line):
    return [part.strip() for part in line.strip().strip("|").split("|")]


def plain(value):
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", value).replace("`", "")


def link(value):
    match = re.search(r"\]\(([^)]*)\)", value)
    return unquote(match.group(1)) if match else ""


def number(value):
    return "" if value in ("—", "None", "") else float(value.rstrip("%"))


def write_csv(name, rows):
    if not rows:
        raise ValueError(f"No rows extracted for {name}")
    path = ROOT / "experiments" / name
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    runs = []
    for dataset in ("diaowu_val300", "chache_val300"):
        path = ARCHIVE / f"{dataset}.txt"
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not re.match(r"^\| \d+ \|", line):
                continue
            a = cells(line)
            if len(a) != 16:
                raise ValueError(f"Unexpected table width: {path}:{lineno}")
            row = dict(dataset=dataset, run_id=f"{dataset}-{int(a[0]):02d}",
                       method=a[1], model=a[2], prompt=plain(a[3]), shot=int(a[4]),
                       input=a[5], negative_examples=a[6], ap50_pct=number(a[7]),
                       ap50_95_pct=number(a[8]))
            for index, threshold in ((9, "05"), (10, "07")):
                for metric, value in zip(("precision", "recall", "f2"), a[index].split("/")):
                    row[f"{metric}_score{threshold}_pct"] = number(value)
            row.update(tp_fp_fn_score05=a[11], successful_total=a[12], status=a[13],
                       iou_maxdets=a[14], original_detail_path=link(a[15]),
                       source_file=f"archive/{path.name}", source_line=lineno)
            runs.append(row)
    write_csv("runs.csv", runs)

    geometry = []
    path = ARCHIVE / "box_precision_20261029.txt"
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not re.match(r"^\| (?:chache_val300|diaowu_99|diaowu_val300)-\d+ \|", line):
            continue
        a = cells(line)
        if len(a) != 11:
            raise ValueError(f"Unexpected geometry table width: {lineno}")
        geometry.append(dict(run_id=a[0], method_model=a[1], condition=plain(a[2]),
            ap85_pct=number(a[3]), ap90_pct=number(a[4]), ap95_pct=number(a[5]),
            ap85_95_pct=number(a[6]), joint20_ap_pct=number(a[7]),
            joint20_score05=a[8], joint20_score07=a[9], status=a[10],
            original_detail_path=link(a[2]), source_file=f"archive/{path.name}", source_line=lineno))
    write_csv("box_precision.csv", geometry)

    early = []
    path = ARCHIVE / "prompt_results_early.txt"
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.startswith("| ["):
            continue
        a = cells(line)
        if len(a) != 9:
            raise ValueError(f"Unexpected early table width: {lineno}")
        early.append(dict(run=plain(a[0]), prompt_type=a[1], text_summary=a[2],
            ap50_pct=number(a[3]), ap50_95_pct=number(a[4]), precision_recall=a[5],
            threshold=a[6], status=a[7], original_detail_path=link(a[0]),
            source_file=f"archive/{path.name}", source_line=lineno))
    write_csv("early_prompt_runs.csv", early)

    original_names = {"diaowu_val300.txt": "results_summary.md",
        "chache_val300.txt": "results_summary(1).md",
        "box_precision_20261029.txt": "README.md",
        "prompt_results_early.txt": "prompt_results.md"}
    manifest = []
    for filename, original_name in original_names.items():
        data = (ARCHIVE / filename).read_bytes()
        manifest.append(dict(path=f"archive/{filename}", original_filename=original_name,
            size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (ROOT / "experiments/source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(runs=len(runs), box_precision=len(geometry), early=len(early))))


if __name__ == "__main__":
    main()
