"""ROI metrics. Recall-first: the gate exists so the detector never misses.

An object (connected component of the GT motion mask in one frame) counts as
recalled if predicted ROI boxes cover at least `cover` of its bounding box.
A predicted ROI box (connected component of the spike map, optionally dilated)
counts as correct if it overlaps any GT object box.
"""
from collections import defaultdict

import numpy as np
from scipy import ndimage


def _boxes(mask):
    lab, n = ndimage.label(mask)
    return [(s[0].start, s[0].stop, s[1].start, s[1].stop) for s in ndimage.find_objects(lab)] if n else []


def _box_mask(boxes, shape):
    m = np.zeros(shape, bool)
    for y0, y1, x0, x1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


def frame_roi_stats(pred, gt, cover=0.5, dilate=2, min_area=2):
    """pred, gt: (H, W) binary. Returns (gt_hit, gt_total, pred_ok, pred_total, roi_area_frac)."""
    if dilate:
        pred = ndimage.binary_dilation(pred, iterations=dilate)
    pboxes = [b for b in _boxes(pred) if (b[1] - b[0]) * (b[3] - b[2]) >= min_area]
    gboxes = _boxes(gt)
    pm = _box_mask(pboxes, gt.shape)
    gm = _box_mask(gboxes, gt.shape)
    hit = sum(pm[y0:y1, x0:x1].mean() >= cover for y0, y1, x0, x1 in gboxes)
    ok = sum(gm[y0:y1, x0:x1].any() for y0, y1, x0, x1 in pboxes)
    return hit, len(gboxes), ok, len(pboxes), pm.mean()


class RoiAccumulator:
    """Per-scene ROI recall/precision plus a failure list (worst sequences)."""

    def __init__(self, skip_first=2, **kw):
        self.skip_first = skip_first  # membranes need a couple of frames to warm up
        self.kw = kw
        self.s = defaultdict(lambda: np.zeros(4))  # gt_hit, gt_total, pred_ok, pred_total
        self.per_seq = []

    def add(self, spk, gt, scenes, seq_ids):
        """spk, gt: (T, B, H, W) numpy binary."""
        T, B = spk.shape[:2]
        for b in range(B):
            acc = np.zeros(5)
            for t in range(self.skip_first, T):
                acc += frame_roi_stats(spk[t, b] > 0, gt[t, b] > 0, **self.kw)
            acc[4] /= max(1, T - self.skip_first)
            self.s[scenes[b]] += acc[:4]
            self.per_seq.append(dict(seq=int(seq_ids[b]), scene=str(scenes[b]),
                                     recall=acc[0] / max(1, acc[1]),
                                     precision=acc[2] / max(1, acc[3]) if acc[3] else 1.0,
                                     roi_area=acc[4]))

    def summary(self):
        out = {}
        for sc in sorted({p["scene"] for p in self.per_seq}):
            s = self.s[sc]
            seqs = [p for p in self.per_seq if p["scene"] == sc]
            out[sc] = dict(recall=s[0] / max(1, s[1]),
                           precision=s[2] / max(1, s[3]) if s[3] else 1.0,
                           roi_area=float(np.mean([p["roi_area"] for p in seqs])),
                           recall_std=float(np.std([p["recall"] for p in seqs])))
        tot = sum(self.s.values())
        out["ALL"] = dict(recall=tot[0] / max(1, tot[1]),
                          precision=tot[2] / max(1, tot[3]) if tot[3] else 1.0,
                          roi_area=float(np.mean([p["roi_area"] for p in self.per_seq])),
                          recall_std=float(np.std([p["recall"] for p in self.per_seq])))
        return out

    def failures(self, k=10):
        return sorted(self.per_seq, key=lambda p: (p["recall"], p["precision"]))[:k]
