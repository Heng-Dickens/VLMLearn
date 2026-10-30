"""Original-image pixel xyxy geometry. These functions never consult ground truth to refine."""
from math import isfinite


def validate_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("box must be [x1, y1, x2, y2]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in box):
        raise ValueError("box coordinates must be finite numbers")
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")
    return x1, y1, x2, y2


def shape(box):
    x1, y1, x2, y2 = validate_box(box)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def iou(a, b):
    a, b = validate_box(a), validate_box(b)
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def geometry_errors(pred, gt):
    px, py, pw, ph = shape(pred)
    gx, gy, gw, gh = shape(gt)
    return dict(width=abs(pw - gw) / gw, height=abs(ph - gh) / gh,
                center_x=abs(px - gx) / gw, center_y=abs(py - gy) / gh)


def joint20(pred, gt, tolerance=0.2):
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    return all(v <= tolerance + 1e-12 for v in geometry_errors(pred, gt).values())


def check_refinement(original, proposed, image_size, limit=0.2):
    """Return (accepted, reason); reject invalid/oversized moves rather than silently clipping.

    Width/height and center movement are relative to the ORIGINAL prediction.
    This is different from joint20(), whose denominator is ground truth.
    """
    validate_box(original)
    if not isfinite(limit) or not 0 <= limit < 1:
        raise ValueError("limit must be finite and in [0,1)")
    if len(image_size) != 2 or any(not isfinite(v) or v <= 0 for v in image_size):
        raise ValueError("image_size must be positive [width,height]")
    try:
        p = validate_box(proposed)
    except ValueError:
        return False, "invalid_box"
    width, height = image_size
    if p[0] < 0 or p[1] < 0 or p[2] > width or p[3] > height:
        return False, "outside_image"
    ox, oy, ow, oh = shape(original)
    px, py, pw, ph = shape(p)
    eps = 1e-12
    if abs(px - ox) > limit * ow + eps or abs(py - oy) > limit * oh + eps:
        return False, "center_limit"
    if abs(pw - ow) > limit * ow + eps or abs(ph - oh) > limit * oh + eps:
        return False, "size_limit"
    return True, "accepted"
