"""Do REAL generated plans violate the filter's motion prior? -- CPU, reads the npz.

`decode_bench` (CPU, 2026-08-24) refuted the scan-blending seam hypothesis and found
the filter's actual fragility: it tolerates an implied inter-frame displacement of
~3x its motion sigma (1.0 lattice = 0.5 wu per decoded step) and collapses past ~5x,
while per-frame argmin is immune. That predicts something measurable about real plans.

This reads `dump_plans.py`'s npz and reports, with NO GPU and NO model:

  1. the plan's implied displacement per decoded step, vs the 0.5 wu prior
  2. how far off the scan manifold the generated frames are at all
  3. a motion-sigma SWEEP on the same plans, scored against the two frames whose
     truth is exact -- frame 0 (the inpainted current scan, truth = start_xy) and
     frame -1 (truth = goal_xy; the model reproduces the goal scan to 1e-07)

Usage:
    python -m diffuser.pl_eval.lidar_eval.plan_motion_census
    python -m diffuser.pl_eval.lidar_eval.plan_motion_census --npz logs/lidar_eval/plans_dump_hist0.npz
"""
import argparse, glob, json, os, sys
sys.path.append('./')

import numpy as np

from diffuser.datasets.lidar import (
    make_scanner_for_env, load_or_build_grid, GridLocalizer, scan_distance)

ENV = 'pointmaze-giant-stitch-v0'
## Capped at 16: `_conv1d_axis` broadcasts out of range once the Gaussian radius
## (3*sigma) exceeds the grid (n_row=73), so sigma>=24 raises. argmin below is the
## sigma->infinity limit anyway, and is measured directly.
SIGMAS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', default=None, help='default: every plans_dump_hist*.npz')
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--out', default='logs/lidar_eval/plan_motion_census.json')
    return p.parse_args()



def motion_calibration(lidar, n_paths=6, n_frames=2000, seed=0):
    """scan-delta -> true displacement, measured on REAL trajectories.

    Estimating the plan's implied motion from the argmin decode conflates real plan
    jumps with argmin ALIASING (a look-alike cell 30 wu away is a 30 wu "step").
    Scan-space delta has no such failure mode: it is what the model actually emitted.
    So calibrate ||scan_{t+k} - scan_t|| against known displacement on real data and
    read the plan's implied motion off that curve.

    Returns (disp_grid, scan_delta_at_that_disp) with scan_delta monotone increasing.
    """
    from diffuser.pl_eval.lidar_eval.decode_bench import build_paths
    rng = np.random.default_rng(seed)
    sc = make_scanner_for_env(ENV, lidar)
    paths = build_paths(n_paths, n_frames, 0.6, rng)
    dd, ss = [], []
    for p in paths:
        scans = sc.scan_batch(p, np.zeros(len(p)))
        for k in (1, 2, 4, 8, 16, 32, 64):
            a_, b_ = scans[:-k], scans[k:]
            dd.append(np.linalg.norm(p[k:] - p[:-k], axis=1))
            ss.append(np.sqrt(((b_ - a_) ** 2).mean(-1)))       # same metric as scan_distance
    dd, ss = np.concatenate(dd), np.concatenate(ss)
    edges = np.arange(0, 12.01, 0.25)
    g, v = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dd >= lo) & (dd < hi)
        if m.sum() >= 50:
            g.append(0.5 * (lo + hi)); v.append(float(np.median(ss[m])))
    g, v = np.asarray(g), np.asarray(v)
    ## Scan-space distance SATURATES: at max_range=12 a pose one cell away and four
    ## cells away are nearly equally distant (BALLV2 measured 4.00 vs 4.39). So the
    ## curve is only invertible up to the saturation knee -- truncate there and treat
    ## anything above it as a LOWER BOUND, never as a magnitude.
    keep = [0]
    for i in range(1, len(v)):
        if v[i] > v[keep[-1]] + 0.02:                           # strictly, meaningfully up
            keep.append(i)
        elif g[i] - g[keep[-1]] > 1.0:                           # plateau reached -> stop
            break
    keep = np.asarray(keep)
    return g[keep], v[keep]


def census(path, a):
    d = np.load(path)
    plans, st, gl = d['plans'], d['start_xy'], d['goal_xy']
    ep, cell = d['ep_id'], float(d['maze_unit'])
    lidar = dict(n_beams=int(d['n_beams']), max_range=float(d['max_range']),
                 fov=2.0 * np.pi, fov_centered=False, step=0.1)
    grid = load_or_build_grid(ENV, lidar, xy_step=0.5, n_theta=1, verbose=False)
    loc = GridLocalizer(grid, sigma=1.0)
    cal_d, cal_s = motion_calibration(lidar)
    N, H, _ = plans.shape
    idx = np.unique(np.concatenate([np.arange(0, H, a.stride), [H - 1]]))
    print(f'\n===== {os.path.basename(path)}  n={N} plans of {H} frames, '
          f"hist={int(d['hist'])} epoch={int(d['epoch'])} =====")

    rows = []
    for i in range(N):
        pl = plans[i]
        dist = scan_distance(pl[idx].astype(np.float32), grid.free_scans)   # (n_dec, M)
        k = np.argmin(dist, axis=1)
        am_xy = grid.free_xy[k].astype(np.float64)                          # argmin decode
        resid = dist[np.arange(len(k)), k]                                  # manifold residual
        step = np.linalg.norm(np.diff(am_xy, axis=0), axis=1)               # implied motion
        ## aliasing-robust: read implied displacement off the scan-delta calibration
        sd = np.sqrt(((pl[idx][1:] - pl[idx][:-1]) ** 2).mean(-1))
        cstep = np.interp(sd, cal_s, cal_d)
        r = dict(i=int(i), ep=int(ep[i]), block=int(ep[i]) // 20,
                 resid_p50=float(np.percentile(resid, 50)),
                 resid_p90=float(np.percentile(resid, 90)),
                 step_p50=float(np.percentile(step, 50)),
                 step_p90=float(np.percentile(step, 90)),
                 step_p99=float(np.percentile(step, 99)),
                 step_max=float(step.max()),
                 cstep_p50=float(np.percentile(cstep, 50)),
                 cstep_p90=float(np.percentile(cstep, 90)),
                 cstep_p99=float(np.percentile(cstep, 99)),
                 frac_cstep_gt_1wu=float((cstep > 1.0).mean()),
                 frac_step_gt_1wu=float((step > 1.0).mean()),
                 frac_step_gt_1cell=float((step > cell).mean()),
                 argmin_head_err=float(np.linalg.norm(am_xy[0] - st[i])),
                 argmin_end_err=float(np.linalg.norm(am_xy[-1] - gl[i])))
        for s in SIGMAS:                                                    # the sweep
            m = np.asarray(loc.localize_sequence(pl[idx].astype(np.float32),
                                                 motion_sigma_cells=s,
                                                 theta_blur=0)['map_xy'], dtype=np.float64)
            r[f'filt{s}_head_err'] = float(np.linalg.norm(m[0] - st[i]))
            r[f'filt{s}_end_err'] = float(np.linalg.norm(m[-1] - gl[i]))
            gp = np.linalg.norm(m - am_xy, axis=1)
            r[f'filt{s}_gap_p50'] = float(np.percentile(gp, 50))
            bad = gp > cell
            r[f'filt{s}_first_div'] = int(idx[int(np.argmax(bad))]) if bad.any() else -1
        rows.append(r)
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{N}', flush=True)

    q = lambda k: float(np.percentile([r[k] for r in rows], 50))
    print(f"\n-- 1. IMPLIED MOTION (argmin decode, per decoded step at stride {a.stride}) --")
    print(f"   motion prior sigma = 1.0 lattice = 0.50 wu  |  decode_bench: filter")
    print(f"   degrades past ~1.5 wu p90 and collapses past ~2.7 wu p90")
    print(f"   [scan-delta calibrated, aliasing-robust]")
    print(f"   step p50={q('cstep_p50'):5.2f}  p90={q('cstep_p90'):5.2f}  "
          f"p99={q('cstep_p99'):5.2f} wu    frac >1 wu = {100*q('frac_cstep_gt_1wu'):.1f}%")
    print(f"   [argmin decode, inflated by aliasing -- for contrast only]")
    print(f"   step p50={q('step_p50'):5.2f}  p90={q('step_p90'):5.2f}  "
          f"p99={q('step_p99'):5.2f}  max={q('step_max'):6.2f} wu   "
          f">1 cell = {100*q('frac_step_gt_1cell'):.1f}%")

    print(f"\n-- 2. MANIFOLD RESIDUAL (min scan distance to any free pose) --")
    print(f"   resid p50={q('resid_p50'):.3f}  p90={q('resid_p90'):.3f}   "
          f"(decode_bench: rendered truth ~0.23, blended seams ~0.68)")

    print(f"\n-- 3. MOTION-SIGMA SWEEP, exact-truth endpoints --")
    print(f"   {'sigma':>7} {'sigma_wu':>9} {'head_err':>9} {'head<=1c':>9} "
          f"{'end_err':>9} {'end<=1c':>9} {'gap_p50':>9} {'1st_div':>8}")
    best = None
    for s in SIGMAS:
        he = [r[f'filt{s}_head_err'] for r in rows]
        ee = [r[f'filt{s}_end_err'] for r in rows]
        w = float(np.mean(np.asarray(ee) <= cell))
        print(f"   {s:>7} {0.5*s:>9.2f} {np.percentile(he,50):>9.2f} "
              f"{100*np.mean(np.asarray(he)<=cell):>8.1f}% {np.percentile(ee,50):>9.2f} "
              f"{100*w:>8.1f}% {np.percentile([r[f'filt{s}_gap_p50'] for r in rows],50):>9.2f} "
              f"{np.percentile([r[f'filt{s}_first_div'] for r in rows],50):>8.0f}")
        if best is None or w > best[1]:
            best = (s, w)
    he = [r['argmin_head_err'] for r in rows]
    ee = [r['argmin_end_err'] for r in rows]
    aw = float(np.mean(np.asarray(ee) <= cell))
    print(f"   {'argmin':>7} {'-':>9} {np.percentile(he,50):>9.2f} "
          f"{100*np.mean(np.asarray(he)<=cell):>8.1f}% {np.percentile(ee,50):>9.2f} "
          f"{100*aw:>8.1f}% {0.0:>9.2f} {'-':>8}")

    print(f"\n-- per block, at the current sigma=1.0 vs the sweep best vs argmin --")
    print(f"   {'blk':>4} {'n':>3} {'cstep_p90':>10} {'resid_p90':>10} "
          f"{'end@1.0':>9} {'end@'+str(best[0]):>9} {'end_argmin':>11}")
    for b in sorted({r['block'] for r in rows}):
        br = [r for r in rows if r['block'] == b]
        m = lambda k: float(np.percentile([r[k] for r in br], 50))
        print(f"   {b:>4} {len(br):>3} {m('cstep_p90'):>10.2f} {m('resid_p90'):>10.3f} "
              f"{m('filt1.0_end_err'):>9.2f} {m(f'filt{best[0]}_end_err'):>9.2f} "
              f"{m('argmin_end_err'):>11.2f}")

    print(f"\n   VERDICT: best filter sigma={best[0]} ({100*best[1]:.1f}% endpoints within a "
          f"cell) vs argmin {100*aw:.1f}%")
    if aw >= best[1] - 0.02:
        print("   -> widening the motion prior does NOT beat argmin. Ship argmin (F4).")
    else:
        print(f"   -> the filter at sigma={best[0]} beats argmin. Ship the rescaled prior (F3).")
    return dict(path=path, n=len(rows), best_sigma=best[0], best_within=best[1],
                argmin_within=aw, rows=rows)


def main():
    a = _parse()
    paths = ([a.npz] if a.npz else
             sorted(glob.glob('logs/lidar_eval/plans_dump_hist*.npz')))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print('no plans_dump_hist*.npz found -- run dump_plans.sbatch first')
        return 1
    out = [census(p, a) for p in paths]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'\nwrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
