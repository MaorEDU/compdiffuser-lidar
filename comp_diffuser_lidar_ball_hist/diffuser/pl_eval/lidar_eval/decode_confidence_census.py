"""Does argmin's own distance array say WHEN it is wrong? -- CPU only, no GPU.

`_decode_plan_xy` computes the scan distance from every plan frame to every free grid
pose, keeps the argmin, and throws the rest away. The rest carries a free per-frame
confidence: how much worse the best pose MORE THAN ONE CELL AWAY scores.

    margin = min_{i : |xy_i - xy*| > cell} d_i  -  min_i d_i

A small margin means a far-away lookalike nearly won -- exactly the near-tie that an
aliased frame teleports through. This uses no map graph and no connectivity: it reads
only the distance array argmin already has.

Three measurements:

1. CALIBRATION on ground truth. Real dataset paths (`decode_bench.build_paths`), scans
   rendered by the real scanner plus Gaussian beam noise. Truth is known, so "wrong" is
   defined: raw argmin more than one cell from the true pose. AUC of -margin for it.
2. GENERATED PLANS. Truth is unknown, so two proxies: (a) OUTLIER -- the raw argmin is
   more than a cell from what the shipped median-15 makes of it; (b) BREAK-ADJACENT --
   the frame sits on either side of a wall-crossing step of the shipped decode.
3. COUNTERFACTUAL. Drop the lowest-margin frames, refill them by linear interpolation in
   time from their confident neighbours, then run the shipped median / interp / 3-tap /
   pin. Endpoints are never dropped. The `keep=1.0` arm MUST reproduce the shipped
   `_decode_plan_xy` exactly -- asserted, not assumed. Read `breaks` and `free_frac`:
   a straight-line refill CAN cross a wall, and the break census will say so.

Usage:
    python -m diffuser.pl_eval.lidar_eval.decode_confidence_census --max_range 24
"""
import argparse
import json
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.append('./')

import numpy as np

from diffuser.datasets.lidar import (
    make_scanner_for_env, load_or_build_grid, GridLocalizer)
from diffuser.datasets.lidar.lidar_localization import scan_distance
from diffuser.pl_eval.lidar_eval.g2a_floor import _DecodeShim, g2a_metrics
from diffuser.pl_eval.lidar_eval.decode_bench import LIDAR_CONFIG, build_paths
from diffuser.pl_eval.lidar_eval.pull_plan_audit import break_census, drivable_prefix
from diffuser.pl_eval.lidar_eval.lidar_grid_common import los_clear
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
    OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as BallPlanner)

ENV = 'pointmaze-giant-stitch-v0'


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', default='logs/lidar_eval/r24_hist16/plans_dump_hist0.npz')
    p.add_argument('--max_range', type=float, default=24.0)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--median', type=int, default=15)
    p.add_argument('--los_margin', type=float, default=0.6)
    p.add_argument('--keeps', default='1.0,0.9,0.8,0.7,0.6,0.5',
                   help='fraction of frames KEPT (highest margin first)')
    p.add_argument('--n_truth', type=int, default=40)
    p.add_argument('--noises', default='0.0,0.5,1.0,2.0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    return p.parse_args()


def auc(score_bad, label):
    """P(score of a bad frame > score of a good frame), ties half. Rank-based."""
    label = np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    s = np.asarray(score_bad, np.float64)
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    ## average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def frame_conf(sub, free_scans, free_xy, cell):
    """Per-frame argmin index, best distance and margin to the best FAR pose."""
    d = scan_distance(sub, free_scans)                              # (n, M)
    k = np.argmin(d, axis=1)
    best = d[np.arange(len(k)), k]
    far = np.linalg.norm(free_xy[None, :, :] - free_xy[k][:, None, :], axis=-1) > cell
    d_far = np.where(far, d, np.inf).min(axis=1)
    return k, best, d_far - best


def gated_decode(H, idx, raw_xy, keep_mask, median):
    """The shipped pipeline after argmin, with dropped frames refilled in time."""
    kept = np.flatnonzero(keep_mask)
    xy = np.stack([np.interp(np.arange(len(raw_xy)), kept, raw_xy[kept, c])
                   for c in (0, 1)], axis=1)
    map_xy = BallPlanner._median_dejitter(xy, median)
    full = np.stack([np.interp(np.arange(H), idx, map_xy[:, 0]),
                     np.interp(np.arange(H), idx, map_xy[:, 1])], axis=1)
    if H >= 3:
        kk = np.ones(3) / 3.0
        pad = np.pad(full, ((1, 1), (0, 0)), mode='edge')
        full = np.stack([np.convolve(pad[:, 0], kk, 'valid'),
                         np.convolve(pad[:, 1], kk, 'valid')], axis=1)
        full[0], full[-1] = map_xy[0], map_xy[-1]
    return full, map_xy


def main():
    a = _parse()
    rng = np.random.default_rng(a.seed)
    d = np.load(a.npz, allow_pickle=True)
    plans, start_xy, goal_xy = d['plans'], d['start_xy'], d['goal_xy']
    if abs(a.max_range - float(d['max_range'])) > 1e-9:
        raise SystemExit(f'--max_range {a.max_range:g} contradicts the dump '
                         f'({float(d["max_range"]):g})')
    cfg = dict(LIDAR_CONFIG); cfg['max_range'] = float(a.max_range)
    scanner = make_scanner_for_env(ENV, cfg)
    grid = load_or_build_grid(ENV, cfg, xy_step=0.5, n_theta=1, verbose=False)
    loc = GridLocalizer(grid, sigma=1.0)
    cell = float(scanner.maze_unit)
    fs = np.asarray(grid.free_scans, np.float32)
    fxy = np.asarray(grid.free_xy, np.float64)
    H = plans.shape[1]
    idx = np.unique(np.concatenate([np.arange(0, H, a.stride), [H - 1]]))
    out = dict(npz=a.npz, max_range=a.max_range, cell=cell, n_plans=int(len(plans)))

    ## ---- 1. calibration on ground truth ------------------------------------
    paths = build_paths(a.n_truth, H, 0.5, rng)
    calib = {}
    for sg in [float(x) for x in a.noises.split(',')]:
        err, mar = [], []
        for p in paths:
            sc = scanner.scan_batch(p, np.zeros(len(p))).astype(np.float32)[idx]
            if sg > 0:
                sc = sc + rng.normal(0, sg, sc.shape).astype(np.float32)
            k, best, m = frame_conf(sc, fs, fxy, cell)
            err.append(np.linalg.norm(fxy[k] - p[idx], axis=1)); mar.append(m)
        err, mar = np.concatenate(err), np.concatenate(mar)
        bad = err > cell
        calib[sg] = dict(wrong_frac=float(bad.mean()), auc=auc(-mar, bad),
                         margin_p50_ok=float(np.median(mar[~bad])),
                         margin_p50_bad=float(np.median(mar[bad])) if bad.any() else None)
        print(f'[calib] noise {sg:3.1f}: wrong {bad.mean():6.3f}  AUC(-margin) '
              f'{calib[sg]["auc"]:.3f}  margin p50 ok {calib[sg]["margin_p50_ok"]:.3f} '
              f'bad {calib[sg]["margin_p50_bad"]}', flush=True)
    out['calibration'] = calib

    ## ---- 2 + 3. generated plans ---------------------------------------------
    shim = _DecodeShim(scanner, loc, 'argmin', a.stride, a.median, 1.0)
    keeps = [float(x) for x in a.keeps.split(',')]
    all_m, all_out, all_brk, all_seam = [], [], [], []
    rows = {kp: [] for kp in keeps}
    seam_mask = np.zeros(H, bool)
    L, ov = int(d['hzn']), int(d['len_ovlp'])
    for c in range(1, int(d['n_comp'])):
        s0 = c * (L - ov); seam_mask[s0:s0 + ov] = True
    for i, pl in enumerate(plans):
        sub = np.asarray(pl, np.float32)[idx]
        k, best, m = frame_conf(sub, fs, fxy, cell)
        raw = fxy[k]
        shipped = shim.decode(np.asarray(pl, np.float32))
        full0, map0 = gated_decode(H, idx, raw, np.ones(len(idx), bool), a.median)
        assert np.allclose(full0, shipped, atol=1e-9), \
            f'plan {i}: keep=1.0 does not reproduce the shipped decode'
        outlier = np.linalg.norm(raw - map0, axis=1) > cell
        st = np.linalg.norm(np.diff(map0, axis=0), axis=1)
        brk = np.zeros(len(idx), bool)
        for j in np.flatnonzero(st > 1.0):
            if not bool(los_clear(scanner, map0[j], map0[j + 1][None],
                                  margin=a.los_margin)[0]):
                brk[j] = brk[j + 1] = True
        all_m.append(m); all_out.append(outlier); all_brk.append(brk)
        all_seam.append(seam_mask[idx])
        gap = float(np.linalg.norm(goal_xy[i] - start_xy[i]))
        order = np.argsort(-m, kind='mergesort')
        for kp in keeps:
            keep = np.zeros(len(idx), bool)
            keep[order[:max(int(round(kp * len(idx))), 2)]] = True
            keep[0] = keep[-1] = True
            xy, _ = gated_decode(H, idx, raw, keep, a.median)
            free, jump, step = g2a_metrics(xy, scanner, cell)
            nb, barc = break_census(xy, scanner, a.los_margin)
            pre = drivable_prefix(xy, scanner, a.los_margin)
            closed = (gap - np.linalg.norm(goal_xy[i] - xy[pre])) / gap if gap > 0 else 0
            rows[kp].append(dict(breaks=nb, free=free, jump=jump, arc=float(step.sum()),
                                 brk_arc=barc / max(step.sum(), 1e-9), prefix=pre,
                                 closed=float(closed),
                                 end_err=float(np.linalg.norm(xy[-1] - goal_xy[i]))))
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(plans)} plans', flush=True)

    m, o, b, s = (np.concatenate(x) for x in (all_m, all_out, all_brk, all_seam))
    gen = dict(n_frames=int(len(m)), outlier_frac=float(o.mean()),
               break_adj_frac=float(b.mean()),
               auc_outlier=auc(-m, o), auc_break=auc(-m, b), auc_seam=auc(-m, s),
               margin_p50=float(np.median(m)),
               margin_p50_outlier=float(np.median(m[o])) if o.any() else None,
               margin_p50_break=float(np.median(m[b])) if b.any() else None,
               margin_p50_seam=float(np.median(m[s])),
               margin_p50_nonseam=float(np.median(m[~s])))
    ## how concentrated: share of outlier / break frames in the lowest-margin decile
    lo = m <= np.quantile(m, 0.1)
    gen['outlier_share_in_low10'] = float(o[lo].sum() / max(o.sum(), 1))
    gen['break_share_in_low10'] = float(b[lo].sum() / max(b.sum(), 1))
    out['generated'] = gen
    print('[generated] ' + json.dumps(gen, indent=1), flush=True)

    print(f'\n{"keep":>5} {"breaks_p50":>10} {"breaks_mean":>11} {"brk_arc":>8} {"free":>7} '
          f'{"jump":>7} {"arc_p50":>8} {"prefix_p50":>10} {"closed_p50":>10} {"endErr":>7}')
    summ = {}
    for kp in keeps:
        r = rows[kp]
        g = lambda k_: np.array([x[k_] for x in r], float)
        summ[kp] = dict(breaks_p50=float(np.median(g('breaks'))),
                        breaks_mean=float(g('breaks').mean()),
                        brk_arc_mean=float(g('brk_arc').mean()),
                        free_p50=float(np.median(g('free'))), free_min=float(g('free').min()),
                        jump_p50=float(np.median(g('jump'))),
                        arc_p50=float(np.median(g('arc'))),
                        prefix_p50=float(np.median(g('prefix'))),
                        closed_p50=float(np.median(g('closed'))),
                        end_err_p50=float(np.median(g('end_err'))),
                        breaks_per_plan=[int(x) for x in g('breaks')])
        q = summ[kp]
        print(f'{kp:5.2f} {q["breaks_p50"]:10.1f} {q["breaks_mean"]:11.2f} '
              f'{q["brk_arc_mean"]:8.3f} {q["free_p50"]:7.4f} {q["jump_p50"]:7.4f} '
              f'{q["arc_p50"]:8.1f} {q["prefix_p50"]:10.0f} {q["closed_p50"]:10.3f} '
              f'{q["end_err_p50"]:7.3f}')
    out['counterfactual'] = summ
    path = a.out or os.path.join(os.path.dirname(a.npz), 'decode_confidence_census.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
