"""Train + evaluate one configuration.

  python -m roi_snn.train --mode diff --beta 0.875
  python -m roi_snn.train --mode raw --data my_scenes.npz --epochs 20

Writes results/<tag>.json with per-scene ROI recall/precision, per-layer firing
rates, input activity and the worst sequences (the failure list).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_npz, make_synthetic
from .encoding import MODES, encode, in_channels, input_activity
from .metrics import RoiAccumulator
from .model import RoiSNN


def split(ds, val_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds.frames))
    n_val = int(len(idx) * val_frac)
    return idx[n_val:], idx[:n_val]


def run(args):
    torch.manual_seed(args.seed)
    ds = load_npz(args.data) if args.data else make_synthetic(args.n_per_scene, T=args.T, seed=args.seed)
    tr_idx, va_idx = split(ds, seed=args.seed)
    frames = torch.from_numpy(ds.frames)
    masks = torch.from_numpy(ds.masks).float()

    model = RoiSNN(in_channels(args.mode), args.hidden, args.beta, learn_beta=args.learn_beta)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Recall-first loss: positives are rare (a few % of pixels) and misses cost more.
    pos_weight = torch.tensor(args.pos_weight)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.default_rng(args.seed + ep).permutation(tr_idx)
        losses = []
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            x = encode(frames[b], args.mode, args.thr) * args.gain
            y = masks[b].transpose(0, 1).unsqueeze(2)  # (T, B, 1, H, W)
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(
                out["logits"][args.skip_first:], y[args.skip_first:], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if args.verbose:
            print(f"[{args.mode} b={args.beta}] ep {ep + 1}/{args.epochs} loss {np.mean(losses):.4f}")

    model.eval()
    acc = RoiAccumulator(skip_first=args.skip_first, cover=args.cover, dilate=args.dilate)
    rate1, rate2, act = [], [], []
    with torch.no_grad():
        for i in range(0, len(va_idx), args.batch):
            b = va_idx[i:i + args.batch]
            x = encode(frames[b], args.mode, args.thr) * args.gain
            out = model(x)
            act.append(input_activity(x))
            rate1.append(out["spk1"].mean().item())
            rate2.append(out["spk2"].mean().item())
            acc.add(out["spk2"][:, :, 0].numpy(), masks[b].transpose(0, 1).numpy(), ds.scenes[b], b)

    res = dict(config=vars(args), train_sec=round(time.time() - t0, 1),
               scenes=acc.summary(),
               firing_rate=dict(lif1=float(np.mean(rate1)), lif2=float(np.mean(rate2))),
               input_activity=float(np.mean(act)),
               failures=acc.failures(args.n_failures))
    if args.learn_beta:
        res["learned_beta"] = dict(lif1=float(model.lif1.beta), lif2=float(model.lif2.beta))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag or f'{args.mode}_b{args.beta}'}.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, default="diff")
    p.add_argument("--beta", type=float, default=0.875)
    p.add_argument("--learn-beta", action="store_true")
    p.add_argument("--thr", type=float, default=0.05, help="diff threshold for thresh/polarity")
    p.add_argument("--gain", type=float, default=4.0,
                   help="input scale; folds into conv1 weights / threshold in HW")
    p.add_argument("--data", help="npz with frames/masks[/scenes]; synthetic if omitted")
    p.add_argument("--n-per-scene", type=int, default=40)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--pos-weight", type=float, default=5.0)
    p.add_argument("--skip-first", type=int, default=2)
    p.add_argument("--cover", type=float, default=0.5)
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--n-failures", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--tag")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


if __name__ == "__main__":
    r = run(build_parser().parse_args())
    print(json.dumps({k: r[k] for k in ("scenes", "firing_rate", "input_activity", "train_sec")}, indent=2, default=float))
