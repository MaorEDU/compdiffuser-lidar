"""Beam/range aliasing census for the ball lidar recipe (giant maze).

Question it answers BEFORE any retrain GPU is spent: which sensing knob, if any,
collapses scan-space aliasing --
  * n_beams (angular resolution)  -> the log's step-2 candidate (60 beams)
  * max_range (seeing past 3 cells) -> untested alternative; locally-congruent
    corridor segments are IDENTICAL at any beam count if all walls in range match.

Pre-registered criterion (fixed before looking at results):
  confusable pose i  <=>  alias_rms(i) < tau
    alias_rms(i) = min RMS-per-beam scan distance to any pose >= 8 wu away
                   (8 wu = 2 cells = an unambiguous "different place";
                    G2a jumps are multi-cell)
    tau          = median over poses of the median scan distance to 0.5-wu
                   lattice neighbors (the cost of one LEGITIMATE half-unit move)
  i.e. "teleporting >=2 cells is cheaper in scan space than legally moving 0.5 wu".
  Secondary far radii 4 / 12 wu reported for robustness; the gate reads 8.

Pure numpy; mirrors LidarGrid poses exactly (giant, xy_step=0.5, n_theta=1,
theta0=0, yaw=0 point-mass convention). The (30, mr12) rebuild is verified
against the shipped cache data/ogb_maze/lidar_grid/..._xy0.5_th1_mr12.npz.
"""
import importlib.util
import json
import sys

import numpy as np

REPO = sys.argv[1] if len(sys.argv) > 1 else '.'
LG_PATH = f'{REPO}/diffuser/datasets/lidar/lidar_grid.py'
CACHE_30_12 = f'{REPO}/data/ogb_maze/lidar_grid/pointmaze-giant-stitch-v0_xy0.5_th1_mr12.npz'
ENV = 'pointmaze-giant-stitch-v0'
CONFIGS = [(30, 12.0), (60, 12.0), (120, 12.0), (30, 24.0), (60, 24.0), (120, 24.0)]
if len(sys.argv) > 3:  # e.g. "60:12" -> run only that config (per-call time budget)
    nb_s, mr_s = sys.argv[3].split(':')
    CONFIGS = [(int(nb_s), float(mr_s))]
FAR_RADII = (4.0, 8.0, 12.0)   # wu; 8.0 is the pre-registered gate radius
NEIGH_R = 0.51                  # wu; catches the 4-connected 0.5-wu lattice step

spec = importlib.util.spec_from_file_location('lidar_grid', LG_PATH)
lg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lg)


def build(nb, mr):
    g = lg.build_grid_for_env(
        ENV, lidar_config=dict(n_beams=nb, max_range=mr),
        xy_step=0.5, n_theta=1, verbose=False)
    return g.free_xy.astype(np.float64), g.free_scans.astype(np.float64)


def pdist2(A):
    sq = (A * A).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (A @ A.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def linf_alias(S, d2_xy, far_r, tau_ref_mask, block=256):
    """L-inf variant (exploratory, NOT the gate): does ANY single beam separate
    far look-alikes? Distinguishes 'sensor is blind' from 'L2 loss dilutes the
    few informative beams'. Returns (alias_linf per pose, tau_linf)."""
    M = len(S)
    S32 = S.astype(np.float32)
    alias = np.full(M, np.inf, dtype=np.float32)
    tau_meds = np.empty(M, dtype=np.float32)
    far2 = far_r ** 2
    for a in range(0, M, block):
        b = min(a + block, M)
        diff = np.abs(S32[a:b, None, :] - S32[None, :, :]).max(-1)  # (b-a, M)
        far = d2_xy[a:b] >= far2
        alias[a:b] = np.where(far, diff, np.inf).min(1)
        for k in range(a, b):
            tau_meds[k] = np.median(diff[k - a][tau_ref_mask[k]])
    return alias, float(np.median(tau_meds))


results = {}
for nb, mr in CONFIGS:
    xy, S = build(nb, mr)
    M = len(S)
    d2_xy = pdist2(xy)
    d2_s = pdist2(S)
    rms = np.sqrt(d2_s) / np.sqrt(nb)          # RMS-per-beam scan distance (wu)

    # legitimate-motion scale tau: median over poses of median neighbor distance
    neigh = (d2_xy > 1e-9) & (d2_xy <= NEIGH_R ** 2)
    step_med = np.array([np.median(rms[i, neigh[i]]) for i in range(M)])
    tau = float(np.median(step_med))

    row = dict(n_beams=nb, max_range=mr, n_poses=M, tau_rms_wu=tau)
    for R in FAR_RADII:
        far = d2_xy >= R ** 2
        rms_far = np.where(far, rms, np.inf)
        alias = rms_far.min(1)                  # nearest far look-alike distance
        n_look = (rms_far < tau).sum(1)         # far look-alikes cheaper than a step
        key = f'far{R:g}'
        row[key] = dict(
            confusable_frac=float((alias < tau).mean()),
            alias_over_tau_p10=float(np.percentile(alias / tau, 10)),
            alias_over_tau_p50=float(np.percentile(alias / tau, 50)),
            n_lookalikes_p50=int(np.percentile(n_look, 50)),
            n_lookalikes_p90=int(np.percentile(n_look, 90)),
        )
    # exploratory L-inf variant at the gate radius (8 wu)
    a_inf, tau_inf = linf_alias(S, d2_xy, 8.0, neigh)
    row['linf_far8'] = dict(
        tau_linf_wu=tau_inf,
        confusable_frac=float((a_inf < tau_inf).mean()),
        alias_over_tau_p50=float(np.percentile(a_inf / tau_inf, 50)),
    )
    results[f'nb{nb}_mr{mr:g}'] = row
    f8 = row['far8']
    li = row['linf_far8']
    print(f"[census] nb={nb:>3} mr={mr:>4} | poses={M} tau={tau:.3f} wu | "
          f"confusable@8wu={100*f8['confusable_frac']:5.1f}% | "
          f"alias/tau p10={f8['alias_over_tau_p10']:.2f} p50={f8['alias_over_tau_p50']:.2f} | "
          f"lookalikes p50/p90={f8['n_lookalikes_p50']}/{f8['n_lookalikes_p90']} | "
          f"Linf: conf={100*li['confusable_frac']:5.1f}% a/t p50={li['alias_over_tau_p50']:.2f}",
          flush=True)

# sanity: the (30, 12) rebuild must match the shipped cache the decode uses
if (30, 12.0) in CONFIGS:
    try:
        cache = np.load(CACHE_30_12)
        xy30, S30 = build(30, 12.0)
        ok_xy = np.allclose(cache['free_xy'], xy30.astype(np.float32))
        ok_s = np.allclose(cache['free_scans'], S30.astype(np.float32))
        print(f'[census] cache check vs {CACHE_30_12.split("/")[-1]}: '
              f'free_xy match={ok_xy} free_scans match={ok_s} '
              f'(cache M={len(cache["free_scans"])})')
        results['cache_check'] = dict(free_xy_match=bool(ok_xy),
                                      free_scans_match=bool(ok_s))
    except FileNotFoundError:
        print('[census] cache npz not found locally; skipped bit-match check')

out = sys.argv[2] if len(sys.argv) > 2 else 'ball_beam_aliasing_census.json'
json.dump(results, open(out, 'w'), indent=1)
print(f'[census] wrote {out}')
