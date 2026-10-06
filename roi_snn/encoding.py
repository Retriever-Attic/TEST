"""Input encodings fed to the SNN, one frame = one timestep.

  raw         grayscale frame (upper-bound diagnostic, not a deployable design)
  diff        f[t] - f[t-1] (current pipeline)
  absdiff     |f[t] - f[t-1]|
  thresh      |diff| kept only where it exceeds `thr` (suppresses noise/flicker)
  polarity    2 channels: ON (brighter > thr) / OFF (darker > thr), DVS-like
  raw+diff    2 channels: raw and diff (separates "model capacity" from "input")
"""
import torch

MODES = ("raw", "diff", "absdiff", "thresh", "polarity", "raw+diff")


def in_channels(mode):
    return 2 if mode in ("polarity", "raw+diff") else 1


def encode(frames, mode, thr=0.05):
    """frames: (B, T, H, W) -> (T, B, C, H, W). First diff step is zero."""
    prev = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
    d = frames - prev
    if mode == "raw":
        x = frames.unsqueeze(2)
    elif mode == "diff":
        x = d.unsqueeze(2)
    elif mode == "absdiff":
        x = d.abs().unsqueeze(2)
    elif mode == "thresh":
        x = (d.abs() * (d.abs() > thr)).unsqueeze(2)
    elif mode == "polarity":
        x = torch.stack([(d > thr).float(), (d < -thr).float()], dim=2)
    elif mode == "raw+diff":
        x = torch.stack([frames, d], dim=2)
    else:
        raise ValueError(f"unknown mode {mode}")
    return x.transpose(0, 1).contiguous()


def input_activity(x):
    """Fraction of nonzero input elements (event rate proxy)."""
    return (x.abs() > 1e-6).float().mean().item()
