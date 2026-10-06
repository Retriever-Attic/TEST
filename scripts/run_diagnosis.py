"""Run the diagnosis matrix and write results/REPORT.md.

  4.1 upper bound : raw vs diff vs raw+diff (same model, same training)
  4.2 encodings   : absdiff, thresh, polarity
  4.3 beta sweep  : shift-friendly betas on the diff input

  python scripts/run_diagnosis.py                       # synthetic scenes
  python scripts/run_diagnosis.py --data scenes.npz     # your footage
  python scripts/run_diagnosis.py --seeds 0 1 2 --epochs 15

Extra args after `--` go to every training run (e.g. -- --hidden 16).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roi_snn.model import SHIFT_BETAS  # noqa: E402
from roi_snn.train import build_parser, run  # noqa: E402

UPPER = ("raw", "diff", "raw+diff")
ENCODINGS = ("absdiff", "thresh", "polarity")


def train(mode, beta, seed, common, out):
    argv = ["--mode", mode, "--beta", str(beta), "--seed", str(seed), "--out", out,
            "--tag", f"{mode}_b{beta}_s{seed}"] + common
    return run(build_parser().parse_args(argv))


def mean_over_seeds(rs):
    scenes = rs[0]["scenes"].keys()
    agg = {sc: {m: float(np.mean([r["scenes"][sc][m] for r in rs])) for m in ("recall", "precision", "roi_area", "recall_std")}
           for sc in scenes}
    return dict(scenes=agg,
                lif1=float(np.mean([r["firing_rate"]["lif1"] for r in rs])),
                lif2=float(np.mean([r["firing_rate"]["lif2"] for r in rs])),
                act=float(np.mean([r["input_activity"] for r in rs])),
                recall_seed_sd=float(np.std([r["scenes"]["ALL"]["recall"] for r in rs])),
                failures=sorted(sum((r["failures"] for r in rs), []), key=lambda p: p["recall"])[:8])


def table(rows, scene_names):
    head = "| config | " + " | ".join(scene_names) + " | ALL R / P | ROI area | LIF1 / LIF2 rate | input act |"
    sep = "|" + "---|" * (len(scene_names) + 5)
    lines = [head, sep]
    for name, r in rows:
        s = r["scenes"]
        cells = [f"{s[sc]['recall']:.2f}" for sc in scene_names]
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | **{s['ALL']['recall']:.2f}** / {s['ALL']['precision']:.2f}"
                     + (f" (±{r['recall_seed_sd']:.2f})" if r["recall_seed_sd"] else "")
                     + f" | {s['ALL']['roi_area']:.2f} | {r['lif1']:.3f} / {r['lif2']:.3f} | {r['act']:.2f} |")
    return "\n".join(lines)


def verdict(res):
    raw, diff = res["raw"]["scenes"]["ALL"], res["diff"]["scenes"]["ALL"]
    both = res["raw+diff"]["scenes"]["ALL"]
    gap = raw["recall"] - diff["recall"]
    lines = [f"- raw recall {raw['recall']:.2f} vs diff {diff['recall']:.2f} (gap {gap:+.2f}), raw+diff {both['recall']:.2f}"]
    dead = [m for m, r in res.items() if r["lif2"] < 0.01]
    if dead:
        lines.append(f"- **경고: 출력층 발화율 < 0.01 ({', '.join(dead)})**. 정보 부족이 아니라 학습 실패일 수 있음 → 해당 행은 판정에서 제외하고 gain/lr/epochs부터 조정.")
    if gap < -0.05:
        # One frame per step + 2 LIF layers: the net must build the temporal difference
        # from membranes alone, so raw is NOT an upper bound here.
        lines.append("- 판정: **raw가 상한선 역할을 못 함**. 얕은 SNN은 막전위만으로 프레임 간 차이를 만들기 어려워 원본에선 모양으로 맞춤. "
                     "'raw도 못 뽑음 → 모델 문제'로 읽으면 안 됨. raw+diff와 diff를 비교하고, diff가 약한 장면(실패 목록)부터 볼 것.")
        if both["recall"] > diff["recall"] + 0.05:
            lines.append("- raw+diff > diff → 차분이 버리는 정보(밝기, 내부 영역)가 도움이 됨. 인코딩 보강 쪽.")
    elif raw["recall"] < 0.8:
        lines.append("- 판정: **모델 쪽 문제 가능성 큼**. 원본을 줘도 못 뽑음 → 구조/손실/라벨/beta 점검.")
    elif gap > 0.05:
        lines.append("- 판정: **입력(차분 인코딩) 문제 가능성 큼**. 원본에선 뽑힘 → 4.2 인코딩 결과 참고.")
    else:
        lines.append("- 판정: 원본과 차분 차이 작음 → 입력보다 모델/라벨/장면 쪽을 먼저 의심.")
    if raw["precision"] < diff["precision"] - 0.05:
        lines.append(f"- 주의: raw 정밀도 {raw['precision']:.2f} < diff {diff['precision']:.2f}. "
                     "원본 입력은 움직임이 아니라 모양으로 맞추는 중(정지 물체 오검출). 진단용으로만 해석할 것.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--betas", type=float, nargs="+", default=list(SHIFT_BETAS))
    ap.add_argument("--base-beta", type=float, default=0.875)
    ap.add_argument("--out", default="results")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    common = ["--epochs", str(a.epochs)] + (["--data", a.data] if a.data else []) + [x for x in a.rest if x != "--"]
    runs_dir = str(Path(a.out) / "runs")

    res = {}
    for mode in UPPER + ENCODINGS:
        res[mode] = mean_over_seeds([train(mode, a.base_beta, s, common, runs_dir) for s in a.seeds])
        print(f"done {mode}: recall {res[mode]['scenes']['ALL']['recall']:.3f}", flush=True)
    sweep = {}
    for beta in a.betas:
        if beta == a.base_beta:
            sweep[beta] = res["diff"]
            continue
        sweep[beta] = mean_over_seeds([train("diff", beta, s, common, runs_dir) for s in a.seeds])
        print(f"done beta {beta}: recall {sweep[beta]['scenes']['ALL']['recall']:.3f}", flush=True)

    scene_names = [k for k in res["diff"]["scenes"] if k != "ALL"]
    best = max(UPPER + ENCODINGS, key=lambda m: res[m]["scenes"]["ALL"]["recall"])
    worst_scene = min(scene_names, key=lambda sc: res["diff"]["scenes"][sc]["recall"])
    md = [
        "# ROI SNN 진단 결과",
        "",
        f"- data: `{a.data or 'synthetic'}` · seeds {a.seeds} · epochs {a.epochs} · base beta {a.base_beta}",
        "- 표의 장면 열은 ROI Recall. ALL은 Recall / Precision (± seed 간 표준편차).",
        "",
        "## 4.1 상한선 (입력만 교체)",
        "",
        table([(m, res[m]) for m in UPPER], scene_names),
        "",
        verdict(res),
        "",
        "## 4.2 차분 인코딩",
        "",
        table([(m, res[m]) for m in ("diff",) + ENCODINGS], scene_names),
        "",
        f"- Recall 최고: **{best}** · diff 기준 최악 장면: **{worst_scene}**",
        "",
        "## 4.3 beta 스윕 (diff 입력, beta = 1 - 2^-k → 시프트 누설)",
        "",
        table([(f"beta {b}", sweep[b]) for b in a.betas], scene_names),
        "",
        "## 실패 목록 (diff, recall 낮은 순)",
        "",
        "| seq | scene | recall | precision | ROI area |",
        "|---|---|---|---|---|",
    ] + [f"| {p['seq']} | {p['scene']} | {p['recall']:.2f} | {p['precision']:.2f} | {p['roi_area']:.2f} |"
         for p in res["diff"]["failures"]]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "REPORT.md").write_text("\n".join(md) + "\n")
    (Path(a.out) / "summary.json").write_text(json.dumps(dict(modes=res, beta_sweep={str(k): v for k, v in sweep.items()}), indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
