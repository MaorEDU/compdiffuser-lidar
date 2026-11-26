"""Are the generated plans' decode jumps ISOLATED or SUSTAINED? -- CPU only.

`g2a_floor` (2026-08-25) established that G2a's `jump_frac <= 0.01` bar is sound:
walkable-by-construction paths score 0.0009 through the SHIPPED decoder while all 100
generated problems score 0.0280. Zero overlap. So G2a's failure is a property of the
plans, not of the instrument.

It also ruled out the obvious explanation. The ground-truth paths carry the known 42.6%
single-frame aliasing at `max_range=12` and still score 0.0009, because the width-15
rolling median kills ISOLATED spikes. That leaves a sharp, falsifiable alternative:

  H  the surviving jumps are SUSTAINED excursions -- runs of consecutive displaced
     frames long enough to defeat the median. A median of width w is defeated by a run
     strictly longer than w//2, so with LIDAR_DECODE_MEDIAN=15 the threshold is 8.

If the run-length mass sits at 1-3, the median is simply under-width and this is still
an aliasing story with a cheap fix. If it sits at >=8, no de-jitter width can help and
the planner is emitting genuinely discontinuous plans -- which is a model problem the
`max_range` retrain may or may not touch.

Reads the 40 real generated plans already on disk (`dump_plans.py`). No GPU.

Also splits jumps by SEAM membership. `decode_bench` cleared scan-space blending for the
filter's ENDPOINT error, but nobody has checked whether jump_frac concentrates in the
392 blended frames of 888 -- a different question about the same 44% of the plan.

Usage:
    python -m diffuser.pl_eval.lidar_eval.jump_census
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
from diffuser.pl_eval.lidar_eval.decode_bench import seam_spans, ENV, LIDAR_CONFIG
from diffuser.pl_eval.lidar_eval.g2a_floor import _DecodeShim, g2a_metrics


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', default='logs/lidar_eval/plans_dump_hist0.npz')
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--median', type=int, default=15)
    p.add_argument('--ref_median', type=int, default=61,
                   help='wide median used only as the robust REFERENCE track')
    p.add_argument('--n_comp', type=int, default=8)
    p.add_argument('--hzn', type=int, default=160)
    p.add_argument('--len_ovlp', type=int, default=56)
    p.add_argument('--out', default='logs/lidar_eval/jump_census.json')
    return p.parse_args()


def runs_of(mask):
    """Lengths of maximal runs of True in a 1-D boolean array."""
    if not mask.any():
        return np.zeros(0, dtype=int)
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    return (idx[1::2] - idx[0::2]).astype(int)


def run_starts(mask):
    if not mask.any():
        return np.zeros(0, dtype=int)
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    return idx[0::2].astype(int)


def main():
    a = _parse()
    d = np.load(a.npz, allow_pickle=True)
    plans = d['plans']                                  # (P, H, n_beams)
    ep_id = d['ep_id']
    cell = float(d['maze_unit'])
    P, H, _ = plans.shape
    print(f'[jump_census] {P} plans, H={H}, cell={cell:g} wu, '
          f'median={a.median} (defeated by runs > {a.median // 2})', flush=True)

    scanner = make_scanner_for_env(ENV, LIDAR_CONFIG)
    grid = load_or_build_grid(ENV, LIDAR_CONFIG, xy_step=0.5, n_theta=1, verbose=False)
    loc = GridLocalizer(grid, sigma=1.0)

    ## seam membership on the FULL plan, exactly as blend_traj_lists writes it
    seam = np.zeros(H, dtype=bool)
    for lo, hi in seam_spans(a.n_comp, a.hzn, a.len_ovlp):
        seam[lo:hi] = True

    rows = []
    all_exc, all_jrun = [], []
    for i_p in range(P):
        scans = plans[i_p]

        ## shipped decode (median ON) -- what G2a actually scores
        shim = _DecodeShim(scanner, loc, 'argmin', a.stride, a.median, 1.0)
        xy_med = shim.decode(scans)
        _, jump_med, step_med = g2a_metrics(xy_med, scanner, cell)

        ## same decode with the de-jitter OFF, to see what the median was handed
        shim_raw = _DecodeShim(scanner, loc, 'argmin', a.stride, 0, 1.0)
        xy_raw = shim_raw.decode(scans)
        _, jump_raw, _ = g2a_metrics(xy_raw, scanner, cell)

        ## EXCURSIONS: raw strided argmin vs a wide robust reference of itself.
        idx = np.unique(np.concatenate([np.arange(0, H, a.stride), [H - 1]]))
        from diffuser.pl_eval.lidar_eval.g2a_floor import _DecodeShim as _S
        raw_strided = _S._median_dejitter(
            np.asarray(xy_raw[idx], dtype=np.float64), 0)
        ref = _S._median_dejitter(raw_strided, a.ref_median)
        dev = np.linalg.norm(raw_strided - ref, axis=1)
        exc_mask = dev > cell
        exc = runs_of(exc_mask)
        all_exc.append(exc)

        ## where the SURVIVING (post-median) jumps sit
        jmask = step_med > cell
        jrun = runs_of(jmask)
        all_jrun.append(jrun)
        js = run_starts(jmask)
        seam_hits = int(seam[np.clip(js, 0, H - 1)].sum())

        rows.append(dict(
            i_p=int(i_p), ep_id=int(ep_id[i_p]), block=int(ep_id[i_p]) // 20,
            jump_frac_med=float(jump_med), jump_frac_raw=float(jump_raw),
            n_exc=int(len(exc)),
            exc_len_p50=float(np.percentile(exc, 50)) if len(exc) else 0.0,
            exc_len_max=int(exc.max()) if len(exc) else 0,
            exc_frac_frames=float(exc_mask.mean()),
            exc_surviving=int((exc > a.median // 2).sum()),
            n_jump_runs=int(len(jrun)),
            jump_run_max=int(jrun.max()) if len(jrun) else 0,
            jump_starts_in_seam=seam_hits,
            n_jump_starts=int(len(js)),
        ))

    exc = np.concatenate(all_exc) if any(len(x) for x in all_exc) else np.zeros(0, int)
    jrun = np.concatenate(all_jrun) if any(len(x) for x in all_jrun) else np.zeros(0, int)
    thr = a.median // 2

    ## seam exposure: the share of frames that are blended, i.e. the null rate
    seam_null = float(seam.mean())
    tot_starts = sum(r['n_jump_starts'] for r in rows)
    tot_seam = sum(r['jump_starts_in_seam'] for r in rows)

    print('\n=================== EXCURSIONS (raw argmin vs wide reference) ==========')
    print(f'  excursion events           : {len(exc)} over {P} plans '
          f'({len(exc)/P:.1f} per plan)')
    if len(exc):
        print(f'  run length p50 / p90 / max : {np.percentile(exc,50):.0f} / '
              f'{np.percentile(exc,90):.0f} / {exc.max()}  (strided frames)')
        print(f'  runs > {thr} (survive med{a.median}) : '
              f'{int((exc>thr).sum())}/{len(exc)}  '
              f'= {100*float((exc>thr).mean()):.1f}%')
        for lo, hi in [(1,1),(2,3),(4,7),(8,15),(16,10**6)]:
            n = int(((exc>=lo)&(exc<=hi)).sum())
            lbl = f'{lo}-{hi}' if hi < 10**6 else f'>={lo}'
            print(f'    len {lbl:>7s} : {n:5d}  ({100*n/len(exc):5.1f}%)'
                  + ('   <- survives the median' if lo > thr else ''))
    print('\n=================== SURVIVING JUMPS (post-med15, what G2a sees) ========')
    print(f'  jump_frac  raw {np.mean([r["jump_frac_raw"] for r in rows]):.4f} '
          f'-> med{a.median} {np.mean([r["jump_frac_med"] for r in rows]):.4f}')
    if len(jrun):
        print(f'  jump runs                  : {len(jrun)} '
              f'({len(jrun)/P:.1f} per plan), p50 {np.percentile(jrun,50):.0f} '
              f'max {jrun.max()}')
    print(f'  jump starts inside a seam  : {tot_seam}/{tot_starts} '
          f'= {100*tot_seam/max(tot_starts,1):.1f}%   (seam frames are '
          f'{100*seam_null:.1f}% of the plan)')
    print('\n=================== PER BLOCK ==========================================')
    print(f'  {"blk":>3s} {"n":>3s} {"jump_med":>9s} {"jump_raw":>9s} '
          f'{"exc/plan":>9s} {"exc>thr":>8s}')
    per_block = {}
    for b in sorted({r['block'] for r in rows}):
        rr = [r for r in rows if r['block'] == b]
        pb = dict(
            n=len(rr),
            jump_frac_med=float(np.mean([r['jump_frac_med'] for r in rr])),
            jump_frac_raw=float(np.mean([r['jump_frac_raw'] for r in rr])),
            exc_per_plan=float(np.mean([r['n_exc'] for r in rr])),
            exc_surviving=float(np.mean([r['exc_surviving'] for r in rr])),
        )
        per_block[b] = pb
        print(f'  {b:3d} {pb["n"]:3d} {pb["jump_frac_med"]:9.4f} '
              f'{pb["jump_frac_raw"]:9.4f} {pb["exc_per_plan"]:9.1f} '
              f'{pb["exc_surviving"]:8.1f}')
    print('========================================================================\n')

    summary = dict(
        npz=a.npz, n_plans=int(P), H=int(H), cell=cell,
        median=a.median, median_defeat_len=int(thr), ref_median=a.ref_median,
        n_exc=int(len(exc)),
        exc_len_p50=float(np.percentile(exc, 50)) if len(exc) else 0.0,
        exc_len_p90=float(np.percentile(exc, 90)) if len(exc) else 0.0,
        exc_len_max=int(exc.max()) if len(exc) else 0,
        exc_surviving_frac=float((exc > thr).mean()) if len(exc) else 0.0,
        jump_frac_raw=float(np.mean([r['jump_frac_raw'] for r in rows])),
        jump_frac_med=float(np.mean([r['jump_frac_med'] for r in rows])),
        jump_starts_in_seam=int(tot_seam), n_jump_starts=int(tot_starts),
        seam_frame_frac=seam_null,
        per_block={str(k): v for k, v in per_block.items()},
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(dict(summary=summary, per_plan=rows), fh, indent=1)
    print(f'wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
