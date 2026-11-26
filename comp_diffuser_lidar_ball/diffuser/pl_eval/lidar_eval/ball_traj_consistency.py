#!/usr/bin/env python3
"""BALL trajectory-consistency analysis (BALL.md Phase 8 / S4).

Goal: "Check the frequency of identical trajectories generated between the LIDAR
scans and the x,y coordinates, and verify if the model consistently chooses
these paths at all."

Given N plan resamples per problem from (a) the LiDAR ball planner and
optionally (b) the (x, y) o2d planner, this tool:
  * decodes LiDAR plans to xy with the grid (sequence-consistent filter),
  * maps every plan to its maze CELL-ROUTE (duplicate-collapsed cell sequence),
  * within-model consistency: route mode-share + route entropy per problem
    ("does the model consistently choose the same path?"),
  * cross-model agreement (when both given): mode-route identity rate, route
    Jaccard, and discrete Frechet distance between xy polylines ("frequency of
    identical trajectories between the LiDAR and x,y models").

Input format (per model): an .npz with
    plans      : (P, N, H, D) float  -- P problems x N resamples; D=30 (scans)
                 or D=2 (xy). Ragged inputs: pass a list via plans_0.npy ... or
                 pad with NaN rows (NaN frames are dropped per plan).
    problem_ids: (P,) int (optional; default 0..P-1)
Produce it by sampling the planner at a checkpoint (see lidar_planner_recon.py
for the sampling pattern) or from the rollout's saved-pkl plans.

Usage
-----
  python -m diffuser.pl_eval.lidar_eval.ball_traj_consistency \
      --env pointmaze-giant-stitch-v0 \
      --plans_lidar plans_lidar.npz [--plans_xy plans_o2d.npz] \
      --out logs/lidar_eval/ball_consistency.json
"""
import os, sys, json, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
try:
    from diffuser.pl_eval.lidar_eval.lidar_grid_common import (
        LG, LL, resolve_lidar_config, json_safe_lidar_config)
except Exception:
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    def _lm(name, rel):
        p = os.path.normpath(os.path.join(here, rel))
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m
    LG = _lm('lidar_grid', '../../datasets/lidar/lidar_grid.py')
    LL = _lm('lidar_localization', '../../datasets/lidar/lidar_localization.py')
    _C = _lm('lidar_grid_common', './lidar_grid_common.py')
    resolve_lidar_config = _C.resolve_lidar_config
    json_safe_lidar_config = _C.json_safe_lidar_config


def frechet(P, Q, max_pts=64):
    """Discrete Frechet distance (subsampled for O(n^2) sanity)."""
    def sub(A):
        A = np.asarray(A, dtype=np.float64)
        if len(A) > max_pts:
            A = A[np.linspace(0, len(A) - 1, max_pts).astype(int)]
        return A
    P, Q = sub(P), sub(Q)
    n, m = len(P), len(Q)
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    C = np.full((n, m), np.inf)
    C[0, 0] = D[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            prev = min(C[i-1, j] if i else np.inf,
                       C[i, j-1] if j else np.inf,
                       C[i-1, j-1] if i and j else np.inf)
            C[i, j] = max(prev, D[i, j])
    return float(C[-1, -1])


def to_cell_route(xy, scanner):
    """(H, 2) xy -> tuple of visited maze cells (consecutive dups collapsed)."""
    ok = np.isfinite(xy).all(axis=1)
    xy = xy[ok]
    i, j = scanner.world_to_cell(xy[:, 0], xy[:, 1])
    route, last = [], None
    for c in zip(np.asarray(i).tolist(), np.asarray(j).tolist()):
        if c != last:
            route.append(c); last = c
    return tuple(route)


def per_problem_consistency(routes):
    """routes: list of cell-route tuples (the N resamples of ONE problem)."""
    vals, counts = np.unique(np.array([hash(r) for r in routes]), return_counts=True)
    p = counts / counts.sum()
    mode_share = float(p.max())
    entropy = float(-(p * np.log(p)).sum())
    mode_route = max(set(routes), key=routes.count)
    return mode_share, entropy, mode_route


def load_plans(path, want_meta=False):
    d = np.load(path, allow_pickle=True)
    plans = d['plans']
    pids = d['problem_ids'] if 'problem_ids' in d.files else np.arange(plans.shape[0])
    out = (np.asarray(plans, dtype=np.float32), np.asarray(pids, dtype=int))
    if want_meta:
        return out + (str(d['meta']) if 'meta' in d.files else None,)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='pointmaze-giant-stitch-v0')
    ap.add_argument('--plans_lidar', required=True)
    ap.add_argument('--plans_xy', default='')
    ap.add_argument('--xy_step', type=float, default=0.5)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--decode_stride', type=int, default=2)
    ap.add_argument('--motion_sigma', type=float, default=1.0)
    ap.add_argument('--cache_root', default='data/ogb_maze/lidar_grid')
    ap.add_argument('--out', default='logs/lidar_eval/ball_consistency.json')
    ## sensing geometry -- resolved, never hardcoded. This tool used to pin
    ## mr12 here, which is the stray mr12 `[grid]` line that kept appearing
    ## AFTER `[G2] wrote` in the mr24 dumps (BALL_SESSIONS_LOG.md §4).
    ap.add_argument('--planner_logdir', default='',
                    help='override the run dir; default = the one in the dump meta')
    ap.add_argument('--n_beams', type=int, default=0, help='0 = from the checkpoint')
    ap.add_argument('--max_range', type=float, default=0.0, help='0 = from the checkpoint')
    args = ap.parse_args()

    plans_l, pids_l, meta_l = load_plans(args.plans_lidar, want_meta=True)
    lidar_cfg = resolve_lidar_config(logdir=args.planner_logdir, meta=meta_l,
                                     n_beams=args.n_beams, max_range=args.max_range)
    grid = LG.load_or_build_grid(args.env, lidar_cfg,
                                 xy_step=args.xy_step, n_theta=1,
                                 cache_root=args.cache_root)
    scanner = grid.scanner
    loc = LL.GridLocalizer(grid, sigma=args.sigma)

    def decode(plan):                                   # (H, 30) scans -> (H, 2) xy
        H = plan.shape[0]
        idx = np.unique(np.concatenate([np.arange(0, H, args.decode_stride), [H-1]]))
        r = loc.localize_sequence(plan[idx], motion_sigma_cells=args.motion_sigma,
                                  theta_blur=0)
        m = np.asarray(r['map_xy'], dtype=np.float64)
        return np.stack([np.interp(np.arange(H), idx, m[:, 0]),
                         np.interp(np.arange(H), idx, m[:, 1])], axis=1)

    def model_pass(tag, plans=None, pids=None, path=''):
        if plans is None:
            plans, pids = load_plans(path)
        P, N = plans.shape[0], plans.shape[1]
        out = {}
        for p in range(P):
            xys, routes = [], []
            for n in range(N):
                pl = plans[p, n]
                pl = pl[np.isfinite(pl).all(axis=1)]
                xy = decode(pl) if pl.shape[1] != 2 else pl
                xys.append(xy)
                routes.append(to_cell_route(xy, scanner))
            ms, ent, mode = per_problem_consistency(routes)
            out[int(pids[p])] = dict(mode_share=ms, entropy=ent,
                                     mode_route=mode, xys=xys)
        mss = [v['mode_share'] for v in out.values()]
        print(f'[consistency/{tag}] {P} problems x {N} resamples: '
              f'mode-share median {np.median(mss):.2f}, '
              f'entropy median {np.median([v["entropy"] for v in out.values()]):.2f}')
        return out

    lid = model_pass('lidar', plans=plans_l, pids=pids_l)
    res = dict(env=args.env,
               lidar_config=json_safe_lidar_config(lidar_cfg),
               lidar=dict(per_problem={str(k): dict(mode_share=v['mode_share'],
                                                    entropy=v['entropy'],
                                                    mode_route=[list(c) for c in v['mode_route']])
                                       for k, v in lid.items()}))
    if args.plans_xy:
        xym = model_pass('xy', path=args.plans_xy)
        agree, jac, fre = [], [], []
        for k in set(lid) & set(xym):
            a, b = lid[k]['mode_route'], xym[k]['mode_route']
            agree.append(a == b)
            sa, sb = set(a), set(b)
            jac.append(len(sa & sb) / max(len(sa | sb), 1))
            fre.append(frechet(np.asarray(lid[k]['xys'][0]),
                               np.asarray(xym[k]['xys'][0])))
        res['cross'] = dict(
            n_shared=len(agree),
            identical_mode_route=float(np.mean(agree)),
            route_jaccard_median=float(np.median(jac)),
            frechet_median_wu=float(np.median(fre)))
        print(f"[consistency/cross] identical mode-route "
              f"{res['cross']['identical_mode_route']*100:.0f}% | "
              f"route Jaccard median {res['cross']['route_jaccard_median']:.2f} | "
              f"Frechet median {res['cross']['frechet_median_wu']:.1f} wu")
        res['xy'] = dict(per_problem={str(k): dict(mode_share=v['mode_share'],
                                                   entropy=v['entropy'])
                                      for k, v in xym.items()})
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)
    print(f'[consistency] wrote {args.out}')


if __name__ == '__main__':
    main()
