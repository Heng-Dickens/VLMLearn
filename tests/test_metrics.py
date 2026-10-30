import copy
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlmlearn.boxes import check_refinement, geometry_errors, iou, joint20
from vlmlearn.metrics import evaluate


def row(box=(0, 0, 100, 100), label="a", image_id="1", score=None):
    r = dict(image_id=image_id, label=label, box=list(box))
    if score is not None:
        r["score"] = score
    return r


class MetricsTests(unittest.TestCase):
    def test_perfect_and_duplicate(self):
        r = evaluate([row()], [row(score=0.9), row(score=0.8)])
        self.assertEqual(r["ap"]["iou50"], 1)
        self.assertEqual(r["operating_point"]["iou50"]["fp"], 1)
        self.assertEqual(r["operating_point"]["iou50"]["precision"], 0.5)

    def test_empty_predictions(self):
        r = evaluate([row()], [])
        self.assertEqual(r["ap"]["iou50"], 0)
        self.assertIsNone(r["operating_point"]["iou50"]["precision"])
        self.assertEqual(r["operating_point"]["iou50"]["recall"], 0)

    def test_no_ground_truth(self):
        r = evaluate([], [row(score=0.9)])
        self.assertIsNone(r["ap"]["iou50"])
        self.assertEqual(r["operating_point"]["iou50"]["fp"], 1)

    def test_class_and_image_must_match(self):
        for p in [row(label="b", score=0.9), row(image_id="2", score=0.9)]:
            r = evaluate([row()], [p])
            self.assertEqual(r["operating_point"]["iou50"]["tp"], 0)
            self.assertEqual(r["operating_point"]["iou50"]["fp"], 1)
            self.assertEqual(r["ap"]["iou50"], 0)

    def test_ap_does_not_use_operating_threshold(self):
        p = [row(score=0.1)]
        r = evaluate([row()], p, threshold=0.9)
        self.assertEqual(r["ap"]["iou50"], 1)
        self.assertEqual(r["operating_point"]["iou50"]["tp"], 0)

    def test_score_order_and_ties(self):
        bad, good = row(box=(200, 200, 300, 300), score=0.5), row(score=0.5)
        r = evaluate([row()], [bad, good])
        self.assertEqual(r["ap"]["iou50"], 0.5)
        self.assertTrue(r["all_scores_equal"])
        self.assertEqual(evaluate([row()], [good, bad])["ap"]["iou50"], 1)

    def test_multiple_classes_macro_ap(self):
        gt = [row(), row(label="b")]
        r = evaluate(gt, [row(score=0.9)])
        self.assertEqual(r["ap"]["iou50"], 0.5)

    def test_missing_scores_disable_ap(self):
        r = evaluate([row()], [row()])
        self.assertTrue(r["unscored"])
        self.assertIsNone(r["ap"]["iou50"])
        self.assertEqual(r["operating_point"]["iou50"]["tp"], 1)

    def test_max_dets_per_image_class(self):
        r = evaluate([row()], [row(box=(200, 200, 300, 300), score=0.9), row(score=0.8)], max_dets=1)
        self.assertEqual(r["ap"]["iou50"], 0)
        self.assertEqual(r["operating_point"]["iou50"]["fp"], 1)

    def test_joint20_independent_from_iou85(self):
        p, g = (20, 0, 120, 100), (0, 0, 100, 100)
        self.assertTrue(joint20(p, g))
        self.assertLess(iou(p, g), 0.85)
        self.assertFalse(joint20((20.01, 0, 120.01, 100), g))

    def test_matching_chooses_eligible_unmatched_gt(self):
        gt = [row(), row(box=(80, 0, 180, 100))]
        pred = [row(score=0.9), row(box=(80, 0, 180, 100), score=0.8)]
        self.assertEqual(evaluate(gt, pred)["operating_point"]["iou50"]["tp"], 2)

    def test_invalid_data(self):
        for bad in [row(box=(0, 0, 0, 1), score=0.5), row(score=math.nan), row(score=2)]:
            with self.assertRaises(ValueError):
                evaluate([row()], [bad])
        for threshold in [-1, math.nan, 2]:
            with self.assertRaises(ValueError):
                evaluate([row()], [], threshold=threshold)

    def test_does_not_mutate_inputs(self):
        gt, pred = [row()], [row(score=0.1)]
        before = copy.deepcopy((gt, pred))
        evaluate(gt, pred)
        self.assertEqual((gt, pred), before)


class RefinementTests(unittest.TestCase):
    def test_boundary_and_excess(self):
        self.assertTrue(check_refinement([100, 100, 200, 200], [120, 100, 220, 200], [640, 480])[0])
        self.assertFalse(check_refinement([100, 100, 200, 200], [120.01, 100, 220.01, 200], [640, 480])[0])
        self.assertTrue(check_refinement([100, 100, 200, 200], [90, 90, 210, 210], [640, 480])[0])
        self.assertFalse(check_refinement([100, 100, 200, 200], [89, 90, 211, 210], [640, 480])[0])

    def test_invalid_and_outside(self):
        self.assertEqual(check_refinement([0, 0, 100, 100], [-1, 0, 99, 100], [640, 480]), (False, "outside_image"))
        self.assertEqual(check_refinement([0, 0, 100, 100], [0, 0, 0, 1], [640, 480]), (False, "invalid_box"))
        self.assertEqual(check_refinement([0, 0, 100, 100], [0, 0, math.inf, 1], [640, 480]), (False, "invalid_box"))

    def test_quality_denominator_is_gt(self):
        self.assertEqual(geometry_errors([0, 0, 120, 100], [0, 0, 100, 100])["width"], 0.2)


if __name__ == "__main__":
    unittest.main()
