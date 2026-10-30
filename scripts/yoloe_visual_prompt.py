#!/usr/bin/env python3
"""Cross-image, multi-reference YOLOE v8/11 example, adapted to pinned THU-MIG/yoloe API.

New example, not recovered historical inference code. Requires upstream environment.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

UPSTREAM_COMMIT = "40cd606cabdbe2b566d6f14a6b162c89206e9a1b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.config.is_file():
        parser.error("weights and config must be existing local files")
    if not 0 <= args.conf <= 1:
        parser.error("conf must be in [0,1]")
    config = json.loads(args.config.read_text())
    names = config["names"]
    if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        parser.error("names must be distinct nonempty strings")
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

    model = YOLOE(str(args.weights))
    references = defaultdict(list)
    base = args.config.resolve().parent
    for ref in config["references"]:
        cid = ref["class_id"]
        if isinstance(cid, bool) or not isinstance(cid, int) or not 0 <= cid < len(names):
            parser.error("reference class_id out of range")
        path = base / ref["image"]
        with Image.open(path) as im:
            width, height = im.size
        box = ref["box"]
        if len(box) != 4 or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in box):
            parser.error("reference box must contain four finite pixel coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            parser.error(f"reference box outside image: {path}")
        # One indicated instance per extraction; group global class IDs ourselves.
        prompts = dict(bboxes=np.array([box], dtype=np.float32), cls=np.array([0]))
        model.predictor = None
        model.predict(str(path), prompts=prompts, predictor=YOLOEVPSegPredictor,
                      return_vpe=True, device=args.device, verbose=False, save=False)
        vpe = model.predictor.vpe.detach()
        if vpe.ndim != 3 or vpe.shape[:2] != (1, 1):
            raise RuntimeError(f"Unexpected VPE shape: {tuple(vpe.shape)}; check upstream API")
        references[cid].append(vpe)
    if set(references) != set(range(len(names))):
        parser.error("every class needs at least one reference instance")
    pe = torch.cat([torch.stack(references[cid]).mean(0) for cid in range(len(names))], dim=1)
    pe = torch.nn.functional.normalize(pe, dim=-1)
    model.set_classes(names, pe)
    model.predictor = None

    records, targets = [], config["targets"]
    image_ids = [t["image_id"] for t in targets]
    if any(not isinstance(i, str) or not i for i in image_ids) or len(set(image_ids)) != len(image_ids):
        parser.error("targets need distinct nonempty string image_id")
    for target in targets:
        results = model.predict(str(base / target["image"]), conf=args.conf,
                                device=args.device, verbose=False, save=False)
        for result in results:
            for det in result.boxes:
                cid = int(det.cls.item())
                records.append(dict(image_id=target["image_id"], label=names[cid],
                                    box=det.xyxy[0].cpu().tolist(), score=float(det.conf.item())))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    digest = hashlib.sha256()
    with args.weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = dict(status="complete", reference_config=config, weights_sha256=digest.hexdigest(),
                    api_target_commit=UPSTREAM_COMMIT, score_source="detector", conf=args.conf,
                    successful_images=len(targets), coordinate_format="original-image pixels xyxy",
                    note="API target is not an automatic check of installed upstream commit")
    (args.output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(records)} boxes from {len(targets)} images to {args.output}")


if __name__ == "__main__":
    main()
