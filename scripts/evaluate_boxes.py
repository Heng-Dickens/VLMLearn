#!/usr/bin/env python3
"""Evaluate normalized JSON records; examples and limitations in scripts/README.md."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlmlearn.metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-dets", type=int, default=100)
    args = parser.parse_args()
    try:
        result = evaluate(json.loads(args.gt.read_text()), json.loads(args.pred.read_text()),
                          args.threshold, args.max_dets)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
