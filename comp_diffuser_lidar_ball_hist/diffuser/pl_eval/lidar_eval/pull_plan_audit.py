"""Is the plan string-pull safe to bench? -- CPU only, no GPU, no checkpoint.

`_pull_plan` is implemented, wired (`LIDAR_BALL_PULL_WIN`) and unit-tested, and
`ball_pull_win` is **0 in every rollout ever run**, because `plan_ball_bench100.sbatch`
never exported it. It is the corrected, PLAN-LEVEL form of the FIX 5 cursor shortcut
that the benchmark runs refuted (that one pulled from the BALL's position, so the monotone
cursor teleported past the route; this one pulls plan-point to plan-point and stays a
connected route).

It is not benched yet because of an unresolved measurement. RETRAIN 8.8.1 scored the
pulled path at `free_frac = 0.727` -- **27% of it inside walls** -- while its arc and
endpoint both looked healthy, and traced that to the plan's own wall-crossing BREAKS
surviving the pull as `i+1` fallbacks and then being resampled into dense straight lines
straight through the wall. The wall-point drop and the zero-length-segment guard were
added afterwards and **the number was never re-measured**. This file re-measures it.

GATE: `free_frac >= 0.95` on the pulled path. Below that the pull is manufacturing
wall points and must not be benched -- use `--split_breaks`, which pulls each
LOS-CONNECTED SPAN independently and never bridges one, and cannot introduce a wall
crossing by construction.

Everything is scored through the SHIPPED `_decode_plan_xy` (called unbound on
`g2a_floor._DecodeShim`, so stride / median / interp / 3-tap / endpoint-pin / pull are
the rollout's, not a copy). The `off` arm is the control and MUST reproduce the shipped
decode exactly; that is asserted, not assumed.

Note `jump_frac` is NOT the reader here, for the same reason it is not under viterbi:
resampling at a fixed 0.4 wu spacing makes small inter-frame steps true by construction.
Read `free_frac` and `breaks` (consecutive points >1 wu apart that `los_clear` says are
not mutually visible) -- a break is what actually severs the route.

Usage:
    python -m diffuser.pl_eval.lidar_eval.pull_plan_audit \
        --npz logs/lidar_eval/r24_hist16/plans_dump_hist0.npz --max_range 24
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
from diffuser.pl_eval.lidar_eval.g2a_floor import _DecodeShim, g2a_metrics
from diffuser.pl_eval.lidar_eval.decode_bench import LIDAR_CONFIG
from diffuser.pl_eval.lidar_eval.lidar_grid_common import los_clear

ENV = 'pointmaze-giant-stitch-v0'
FREE_FRAC_GATE = 0.95


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', default='logs/lidar_eval/r24_hist16/plans_dump_hist0.npz')
    p.add_argument('--max_range', type=float, default=None,
                   help='MUST match the checkpoint that produced the dump (r24 -> 24), '
                        'else the grid and the scanner silently disagree with it')
    p.add_argument('--stride', type=int, default=2, help='ball_decode_stride')
    p.add_argument('--median', type=int, default=15, help='LIDAR_DECODE_MEDIAN')
    p.add_argument('--pull_stride', type=int, default=4)
    p.add_argument('--pull_resample_wu', type=float, default=0.4)
    p.add_argument('--los_margin', type=float, default=0.6)
    p.add_argument('--wins', default='0,200,888', help='pull windows to score')
    p.add_argument('--n_plans', type=int, default=0, help='0 = all')
    p.add_argument('--out', default=None, help='default: <npz dir>/pull_plan_audit.json')
    return p.parse_args()


def break_census(xy, scanner, margin, cell_gap=1.0):
    """Wall-crossing breaks: consecutive points more than `cell_gap` apart that
    `los_clear` says are NOT mutually visible. This is the quantity that severs a
    route -- 24.8 per plan (p50 24) on the shipped decode, 61.7% of the arc."""
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cand = np.flatnonzero(d > cell_gap)
    if not len(cand):
        return 0, 0.0
    ok = np.array([bool(los_clear(scanner, xy[i], xy[i + 1][None], margin=margin)[0])
                   for i in cand])
    brk = cand[~ok]
    return int(len(brk)), float(d[brk].sum())


def drivable_prefix(xy, scanner, margin, cell_gap=1.0):
    """Index of the FIRST break -- how much of the plan the ball can actually drive."""
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    for i in np.flatnonzero(d > cell_gap):
        if not bool(los_clear(scanner, xy[i], xy[i + 1][None], margin=margin)[0]):
            return int(i)
    return int(len(xy) - 1)


def main():
    a = _parse()
    d = np.load(a.npz, allow_pickle=True)
    plans = d['plans']
    if a.n_plans:
        plans = plans[:a.n_plans]
    start_xy, goal_xy = d['start_xy'], d['goal_xy']

    lidar_config = dict(LIDAR_CONFIG)
    mr = a.max_range if a.max_range is not None else float(d['max_range'])
    lidar_config['max_range'] = float(mr)
    if a.max_range is not None and abs(a.max_range - float(d['max_range'])) > 1e-9:
        raise SystemExit(
            f'--max_range {a.max_range:g} contradicts the dump\'s {float(d["max_range"]):g}. '
            f'The grid would be built for a different sensor than produced these plans.')
    scanner = make_scanner_for_env(ENV, lidar_config)
    grid = load_or_build_grid(ENV, lidar_config, xy_step=0.5, n_theta=1, verbose=False)
    loc = GridLocalizer(grid, sigma=1.0)
    cell = float(scanner.maze_unit)
    wins = [int(w) for w in a.wins.split(',')]
    print(f'[pull_audit] {len(plans)} plans  max_range={mr:g}  cell={cell:g} wu  '
          f'wins={wins}  gate free_frac >= {FREE_FRAC_GATE}', flush=True)

    def shim(win):
        return _DecodeShim(scanner, loc, 'argmin', a.stride, a.median, 1.0,
                           pull_win=win, pull_stride=a.pull_stride,
                           pull_resample_wu=a.pull_resample_wu, los_margin=a.los_margin)

    rows = {w: [] for w in wins}
    base = None
    for i, pl in enumerate(plans):
        for w in wins:
            xy = shim(w).decode(np.asarray(pl, dtype=np.float32))
            if w == 0 and base is None:
                base = xy.copy()
            free, jump, step = g2a_metrics(xy, scanner, cell)
            nbrk, brk_arc = break_census(xy, scanner, a.los_margin)
            arc = float(step.sum())
            pre = drivable_prefix(xy, scanner, a.los_margin)
            gap = float(np.linalg.norm(goal_xy[i] - start_xy[i]))
            rows[w].append(dict(
                free_frac=free, jump_frac=jump, arc=arc, n=len(xy),
                breaks=nbrk, break_arc_frac=(brk_arc / arc if arc > 0 else 0.0),
                prefix_arc=float(np.linalg.norm(
                    np.diff(xy[:pre + 1], axis=0), axis=1).sum()),
                prefix_frac_of_plan=float(pre) / max(len(xy) - 1, 1),
                end_err=float(np.linalg.norm(xy[-1] - goal_xy[i])),
                fits_budget=bool(arc <= 200.0), gap=gap))
        if (i + 1) % 20 == 0:
            print(f'  {i + 1}/{len(plans)}', flush=True)

    ## CONTROL: win=0 must be the shipped decode, pass-through, bit-exact.
    ctl = _DecodeShim(scanner, loc, 'argmin', a.stride, a.median, 1.0).decode(
        np.asarray(plans[0], dtype=np.float32))
    assert np.array_equal(ctl, base), (
        'pull_win=0 is not a pass-through -- the control arm does not reproduce the '
        'shipped decode, so nothing measured against it means anything.')

    def agg(rs, k):
        return float(np.median([r[k] for r in rs]))

    out = dict(npz=a.npz, max_range=mr, n_plans=len(plans), stride=a.stride,
               median=a.median, pull_stride=a.pull_stride,
               pull_resample_wu=a.pull_resample_wu, gate_free_frac=FREE_FRAC_GATE,
               arms={})
    print(f'\n{"win":>5} {"free_p50":>9} {"free_min":>9} {"jump_p50":>9} {"arc_p50":>8} '
          f'{"breaks_p50":>11} {"brkarc":>7} {"pfx_arc":>8} {"endErr":>7} {"fits":>6} '
          f'{"GATE":>5}')
    for w in wins:
        rs = rows[w]
        fmin = float(np.min([r['free_frac'] for r in rs]))
        passed = agg(rs, 'free_frac') >= FREE_FRAC_GATE
        out['arms'][str(w)] = dict(
            free_frac_p50=agg(rs, 'free_frac'), free_frac_min=fmin,
            jump_frac_p50=agg(rs, 'jump_frac'), arc_p50=agg(rs, 'arc'),
            breaks_p50=agg(rs, 'breaks'), break_arc_frac_p50=agg(rs, 'break_arc_frac'),
            prefix_arc_p50=agg(rs, 'prefix_arc'),
            prefix_frac_of_plan_p50=agg(rs, 'prefix_frac_of_plan'),
            end_err_p50=agg(rs, 'end_err'),
            n_fits_budget=int(sum(r['fits_budget'] for r in rs)),
            gate_pass=bool(passed), per_plan=rs)
        print(f'{w:>5} {agg(rs,"free_frac"):>9.4f} {fmin:>9.4f} '
              f'{agg(rs,"jump_frac"):>9.4f} {agg(rs,"arc"):>8.1f} '
              f'{agg(rs,"breaks"):>11.1f} {agg(rs,"break_arc_frac"):>7.3f} '
              f'{agg(rs,"prefix_arc"):>8.1f} {agg(rs,"end_err"):>7.2f} '
              f'{sum(r["fits_budget"] for r in rs):>4}/{len(rs):<2} '
              f'{"PASS" if passed else "FAIL":>5}')

    print('\nRead free_frac and breaks, NOT jump_frac: resampling at a fixed '
          f'{a.pull_resample_wu} wu spacing makes small inter-frame steps true by '
          'construction, exactly as a hard step cap does under viterbi.')
    dest = a.out or os.path.join(os.path.dirname(a.npz) or '.', 'pull_plan_audit.json')
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    with open(dest, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'[pull_audit] -> {dest}')


if __name__ == '__main__':
    main()
