"""Scene data for the ROI experiments.

Two sources:
  * Synthetic scenes (default): moving objects + static look-alike distractors,
    with per-scene nuisances (sensor noise, illumination flicker, texture, slow
    large objects). Used to exercise the harness and to show *why* each input
    encoding fails, not to stand in for real footage.
  * Your own data: an .npz with `frames` (N, T, H, W) float in [0, 1] and
    `masks` (N, T, H, W) {0, 1} marking moving-object pixels, plus optional
    `scenes` (N,) string labels. See `load_npz`.
"""
from dataclasses import dataclass

import numpy as np

SCENES = ("clean", "noise", "flicker", "texture", "small_fast", "slow_large")


@dataclass
class SceneSet:
    frames: np.ndarray  # (N, T, H, W) float32 in [0, 1]
    masks: np.ndarray  # (N, T, H, W) uint8, moving-object pixels
    scenes: np.ndarray  # (N,) str


def _draw_rect(img, mask, cx, cy, w, h, value):
    H, W = img.shape
    x0, x1 = int(max(0, cx - w / 2)), int(min(W, cx + w / 2))
    y0, y1 = int(max(0, cy - h / 2)), int(min(H, cy + h / 2))
    if x1 <= x0 or y1 <= y0:
        return
    img[y0:y1, x0:x1] = value
    if mask is not None:
        mask[y0:y1, x0:x1] = 1


def make_sequence(scene, T=32, H=64, W=64, rng=None):
    rng = rng or np.random.default_rng()
    bg_level = rng.uniform(0.2, 0.5)
    if scene == "texture":
        # Low-frequency static texture: no motion, but strong spatial edges.
        coarse = rng.uniform(-0.15, 0.15, size=(H // 8, W // 8))
        bg = bg_level + np.kron(coarse, np.ones((8, 8)))
    else:
        bg = np.full((H, W), bg_level)

    # Speed/size ranges per scene. slow_large is the case frame-diff handles worst:
    # a big uniform object moving < 1 px/frame only lights up its leading edge.
    n_obj = rng.integers(1, 3)
    if scene == "small_fast":
        size_rng, speed_rng = (3, 6), (2.5, 4.0)
    elif scene == "slow_large":
        size_rng, speed_rng = (14, 22), (0.3, 0.8)
    else:
        size_rng, speed_rng = (6, 12), (1.0, 2.5)

    objs = []
    for _ in range(n_obj):
        ang = rng.uniform(0, 2 * np.pi)
        sp = rng.uniform(*speed_rng)
        objs.append(dict(
            x=rng.uniform(10, W - 10), y=rng.uniform(10, H - 10),
            vx=sp * np.cos(ang), vy=sp * np.sin(ang),
            w=rng.uniform(*size_rng), h=rng.uniform(*size_rng),
            val=np.clip(bg_level + rng.choice([-1, 1]) * rng.uniform(0.25, 0.45), 0, 1),
        ))
    # Static distractors look like objects but never move: a raw-frame model that
    # matches on shape instead of motion will flag them (precision drop).
    distractors = [dict(x=rng.uniform(8, W - 8), y=rng.uniform(8, H - 8),
                        w=rng.uniform(*size_rng), h=rng.uniform(*size_rng),
                        val=np.clip(bg_level + rng.choice([-1, 1]) * rng.uniform(0.25, 0.45), 0, 1))
                   for _ in range(rng.integers(1, 3))]

    frames = np.zeros((T, H, W), np.float32)
    masks = np.zeros((T, H, W), np.uint8)
    for t in range(T):
        img = bg.copy()
        for d in distractors:
            _draw_rect(img, None, d["x"], d["y"], d["w"], d["h"], d["val"])
        for o in objs:
            o["x"] += o["vx"]
            o["y"] += o["vy"]
            if not 4 < o["x"] < W - 4:
                o["vx"] *= -1
            if not 4 < o["y"] < H - 4:
                o["vy"] *= -1
            _draw_rect(img, masks[t], o["x"], o["y"], o["w"], o["h"], o["val"])
        if scene == "flicker":
            img = img * (1 + 0.12 * np.sin(2 * np.pi * t / 5 + rng.uniform(0, 6)))
        noise_sd = 0.06 if scene == "noise" else 0.01
        img = img + rng.normal(0, noise_sd, size=img.shape)
        frames[t] = np.clip(img, 0, 1)
    return frames, masks


def make_synthetic(n_per_scene=40, T=32, H=64, W=64, scenes=SCENES, seed=0):
    rng = np.random.default_rng(seed)
    fr, ms, sc = [], [], []
    for s in scenes:
        for _ in range(n_per_scene):
            f, m = make_sequence(s, T, H, W, rng)
            fr.append(f)
            ms.append(m)
            sc.append(s)
    return SceneSet(np.stack(fr), np.stack(ms), np.array(sc))


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    frames = d["frames"].astype(np.float32)
    if frames.max() > 1.0:
        frames /= 255.0
    masks = (d["masks"] > 0).astype(np.uint8)
    scenes = d["scenes"].astype(str) if "scenes" in d else np.array(["all"] * len(frames))
    return SceneSet(frames, masks, scenes)
