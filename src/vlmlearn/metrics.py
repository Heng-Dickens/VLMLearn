"""Transparent small-box evaluator, not a replacement for COCO/LVIS official evaluators.

No crowd/ignore/area buckets. Greedy score-ordered same-class matching.
Input-order tie break. AP over classes with GT only, using 101 recall samples.
"""
from collections import Counter, defaultdict
from math import isfinite
from .boxes import iou, joint20, validate_box


def validate_records(records, predictions=False):
    if not isinstance(records, list):
        raise ValueError("records must be a JSON list")
    for r in records:
        if not isinstance(r, dict) or any(k not in r for k in ("image_id", "label", "box")):
            raise ValueError("record needs image_id, label, box")
        if any(not isinstance(r[k], str) or not r[k] for k in ("image_id", "label")):
            raise ValueError("image_id and label must be nonempty strings")
        validate_box(r["box"])
        if predictions and "score" in r:
            s = r["score"]
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not isfinite(s) or not 0 <= s <= 1:
                raise ValueError("score must be a finite number in [0,1]")


def select_predictions(predictions, threshold=0.0, max_dets=100):
    if not 0 <= threshold <= 1 or not isfinite(threshold):
        raise ValueError("score threshold must be in [0,1]")
    if not isinstance(max_dets, int) or isinstance(max_dets, bool) or max_dets <= 0:
        raise ValueError("max_dets must be a positive integer")
    selected, counts = [], Counter()
    # Unscored outputs stay in their original order at score 1, and AP is disabled.
    ranked = sorted(enumerate(predictions), key=lambda x: (-x[1].get("score", 1.0), x[0]))
    for _, r in ranked:
        key = (r["image_id"], r["label"])
        if r.get("score", 1.0) >= threshold and counts[key] < max_dets:
            selected.append(r)
            counts[key] += 1
    return selected


def match(gt, predictions, criterion):
    groups = defaultdict(list)
    for index, r in enumerate(gt):
        groups[(r["image_id"], r["label"])].append((index, r))
    used, hits = set(), []
    for p in predictions:
        candidates = [(iou(p["box"], g["box"]), index)
                      for index, g in groups[(p["image_id"], p["label"])]
                      if index not in used and criterion(p["box"], g["box"])]
        if candidates:
            # Maximum IoU among eligible GT; earlier GT wins exact ties.
            _, best = max(candidates, key=lambda pair: (pair[0], -pair[1]))
            used.add(best)
            hits.append(1)
        else:
            hits.append(0)
    return hits


def counts(gt, pred, criterion):
    tp = sum(match(gt, pred, criterion))
    return dict(tp=tp, fp=len(pred) - tp, fn=len(gt) - tp,
                precision=tp / len(pred) if pred else None,
                recall=tp / len(gt) if gt else None)


def ap101(hits, n_gt):
    if n_gt == 0:
        return None
    precision, recall, tp = [], [], 0
    for rank, hit in enumerate(hits, 1):
        tp += hit
        precision.append(tp / rank)
        recall.append(tp / n_gt)
    return sum(max((p for p, r in zip(precision, recall) if r + 1e-12 >= t / 100), default=0)
               for t in range(101)) / 101


def evaluate(gt, predictions, threshold=0.5, max_dets=100):
    validate_records(gt)
    validate_records(predictions, predictions=True)
    candidates = select_predictions(predictions, 0.0, max_dets)
    operating = select_predictions(predictions, threshold, max_dets)
    criteria = {f"iou{int(round(t * 100))}": (lambda p, g, t=t: iou(p, g) + 1e-12 >= t)
                for t in [0.5 + 0.05 * i for i in range(10)]}
    criteria["joint20"] = joint20
    classes = sorted({g["label"] for g in gt})
    unscored = any("score" not in p for p in predictions)
    result = dict(score_threshold=threshold, max_dets_per_image_per_class=max_dets,
                  units="rates_0_to_1", tie_break="stable_input_order", unscored=unscored,
                  ap_unavailable_reason="unscored_predictions" if unscored else None,
                  all_scores_equal=len({p.get("score") for p in predictions}) == 1 if predictions else False,
                  operating_point={}, ap={}, per_class_ap={})
    for name, criterion in criteria.items():
        result["operating_point"][name] = counts(gt, operating, criterion)
        per_class = {}
        for label in classes:
            class_gt = [g for g in gt if g["label"] == label]
            class_pred = [p for p in candidates if p["label"] == label]
            per_class[label] = None if unscored else ap101(match(class_gt, class_pred, criterion), len(class_gt))
        result["per_class_ap"][name] = per_class
        result["ap"][name] = (sum(per_class.values()) / len(classes)) if classes and not unscored else None
    values = [result["ap"][f"iou{t}"] for t in range(50, 100, 5)]
    result["ap"]["iou50_95"] = sum(values) / len(values) if all(v is not None for v in values) else None
    return result
