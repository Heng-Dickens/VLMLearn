#!/usr/bin/env python3
"""Inspect a local trusted YOLOE checkpoint, without changing its trainability."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error("checkpoint must exist locally")
    from ultralytics import YOLOE
    model = YOLOE(str(args.weights)).model
    print("SAVPE present:", any("savpe" in n for n, _ in model.named_modules()))
    total = trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        trainable += param.numel() if param.requires_grad else 0
        print(f"{name}\t{list(param.shape)}\t{param.numel()}\trequires_grad={param.requires_grad}")
    print(f"total={total}, currently_trainable={trainable}")
    print("Checkpoint flags only: Trainer can rebuild the model or change freeze rules.")
    print("For actual training, collect the same listing after Trainer setup and before optimizer creation.")


if __name__ == "__main__":
    main()
