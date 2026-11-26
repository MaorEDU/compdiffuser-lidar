"""What jump_frac does a PHYSICALLY WALKABLE plan actually score? -- CPU only.

G2a pre-registers `free_frac >= 0.95` and `jump_frac <= 0.01`. The free-cell half
passes everywhere. The jump half has NEVER passed, under any decoder, on any run:
0.0123 (block-0 filter), 0.0169 (all-blocks filter), 0.0280 (argmin). Both numbers
were picked before the first gate ran, and neither comes from the paper -- the
CompDiffuser release has no plan-validity metric at all, because its `o2d` branch
emits (x, y) directly and has no decoder to be measured.

A bar no configuration has ever cleared is either a real standing defect or a
miscalibrated threshold, and G2a cannot tell us which, because it scores
PLAN + DECODER jointly and never sees a plan known to be good.

So: take paths that are walkable BY CONSTRUCTION -- real dataset trajectories,
chained through `decode_bench.build_paths` -- render their scans with the real
scanner, and push them through the SHIPPED `_decode_plan_xy` (called unbound on a
shim, so the stride / median / interp / 3-tap / endpoint-pin pipeline is
bit-identical to the rollout's, not a re-implementation). Score them with G2a's
exact expressions.

That splits the 0.028 into two parts that the gate currently confounds:

  truth     the kinematic floor -- what a real walkable path scores with NO decoder.
            Anything above this is instrument, not planner.
  argmin /  the DECODER floor -- what the shipped decoder scores on a plan that is
  filter    perfect. If this is ~0.025, the 0.01 bar is unreachable by construction
            and must be re-registered against it. If it is ~0.002, our generated
            plans really are discontinuous and 0.028 is a live defect.

Usage:
    python -m diffuser.pl_eval.lidar_eval.g2a_floor
    python -m diffuser.pl_eval.lidar_eval.g2a_floor --n_paths 60
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
from diffuser.pl_eval.lidar_eval.decode_bench import build_paths, ENV, LIDAR_CONFIG
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
    OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as BallPlanner)

## G2a's pre-registered thresholds, imported rather than restated so this cannot
## drift away from the gate it is calibrating.
from diffuser.pl_eval.lidar_eval.plan_decode_gate import G2A_FREE_FRAC, G2A_JUMP_FRAC


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n_paths', type=int, default=40)
    p.add_argument('--hzn', type=int, default=160)
    p.add_argument('--n_comp', type=int, default=8)
    p.add_argument('--len_ovlp', type=int, default=56)
    p.add_argument('--stride', type=int, default=2,
                   help='ball_decode_stride; the shipped default is 2')
    p.add_argument('--median', type=int, default=15,
                   help='LIDAR_DECODE_MEDIAN; the shipped default is 15')
    p.add_argument('--motion_sigma', type=float, default=1.0)
    p.add_argument('--join_tol', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_range', type=float, default=None,
                   help='override the imported LIDAR_CONFIG range; the deployed '
                        'arm is r24, while decode_bench.LIDAR_CONFIG is r12')
    p.add_argument('--vit_max_step', type=float, default=2.0)
    p.add_argument('--vit_beam', type=int, default=2048)
    p.add_argument('--vit_w_step', type=float, default=0.0)
    p.add_argument('--vit_sigma_e', type=float, default=1.0)
    p.add_argument('--out', default='logs/lidar_eval/g2a_floor.json')
    return p.parse_args()


class _DecodeShim:
    """The attribute surface `_decode_plan_xy` touches, and nothing else.

    Calling the real method unbound is deliberate: a re-implementation here would
    be one more thing that can silently disagree with the rollout, which is the
    exact class of bug this file exists to measure.
    """

    ## borrowed off the class so the de-jitter is the shipped one, not a copy
    _median_dejitter = staticmethod(BallPlanner._median_dejitter)
    ## likewise the viterbi branch: an unbound call keeps the anchors, the graph reuse
    ## and the no-median rule identical to the rollout's
    _decode_viterbi = BallPlanner._decode_viterbi
    ## ...and the plan string-pull, which `_decode_plan_xy` returns THROUGH. Without
    ## it bound here the unbound call raises `AttributeError: _pull_plan`, which is
    ## what every offline decode gate did between `_pull_plan` landing (2026-09-03)
    ## and this line (2026-09-04) -- unnoticed because no gate was run in between.
    ## Defaults leave `_pull_win = 0`, i.e. a pass-through, so every number these
    ## gates produced before 2026-09-03 is reproduced exactly.
    _pull_plan = BallPlanner._pull_plan

    def __init__(self, scanner, loc, mode, stride, median, motion_sigma,
                 max_step=3.0, beam=1024, w_step=0.0, sigma_e=1.0, anchor_xy=None,
                 pull_win=0, pull_stride=4, pull_resample_wu=0.4, los_margin=0.6):
        self.scanner = scanner
        self.loc = loc
        self.n_beams = int(scanner.n_beams)
        self._decode_mode = mode
        self._decode_stride = int(stride)
        self._decode_median = int(median)
        self._decode_motion_sigma = float(motion_sigma)
        self._decode_stats = []
        ## viterbi knobs, mirroring the planner's attribute names
        self._vit_max_step = float(max_step)
        self._vit_beam = int(beam)
        self._vit_w_step = float(w_step)
        self._vit_sigma_e = float(sigma_e)
        self._vit_graph = None
        self._decode_anchor_xy = anchor_xy
        ## string-pull knobs, mirroring the planner's attribute names. 0 = off.
        self._pull_win = int(pull_win)
        self._pull_stride = max(int(pull_stride), 1)
        self._pull_resample_wu = float(pull_resample_wu)
        self._los_margin = float(los_margin)
        self._pull_stats = []

    def decode(self, scans):
        return BallPlanner._decode_plan_xy(self, scans)


def g2a_metrics(xy, scanner, cell):
    """G2a exactly as `plan_decode_gate` computes it, on a full-length (H, 2) plan."""
    ii, jj = scanner.world_to_cell(xy[:, 0], xy[:, 1])
    free_frac = float((~scanner._is_wall(ii, jj)).mean())
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    jump_frac = float((step > cell).mean()) if len(step) else 0.0
    return free_frac, jump_frac, step


def main():
    a = _parse()
    tot = a.n_comp * a.hzn - (a.n_comp - 1) * a.len_ovlp
    rng = np.random.default_rng(a.seed)

    lidar_config = dict(LIDAR_CONFIG)
    if a.max_range is not None:
        lidar_config['max_range'] = float(a.max_range)
    scanner = make_scanner_for_env(ENV, lidar_config)
    grid = load_or_build_grid(ENV, lidar_config, xy_step=0.5, n_theta=1, verbose=False)
    loc = GridLocalizer(grid, sigma=1.0)
    cell = float(scanner.maze_unit)
    print(f'[g2a_floor] tot_hzn={tot}  cell={cell:g} wu  stride={a.stride} '
          f'median={a.median}  n_paths={a.n_paths}  '
          f'max_range={lidar_config["max_range"]:g}  '
          f'viterbi(max_step={a.vit_max_step:g}, beam={a.vit_beam})', flush=True)

    paths = build_paths(a.n_paths, tot, a.join_tol, rng)
    print(f'[g2a_floor] chained {len(paths)} ground-truth paths of {tot} frames',
          flush=True)

    arms = ['truth', 'argmin', 'filter', 'viterbi']
    rows = {k: [] for k in arms}

    for i_p, path_xy in enumerate(paths):
        truth_xy = np.asarray(path_xy[:tot], dtype=np.float64)
        scans = scanner.scan_batch(truth_xy, np.zeros(tot))

        f, j, step = g2a_metrics(truth_xy, scanner, cell)
        rows['truth'].append(dict(i_p=i_p, free_frac=f, jump_frac=j,
                                  arc=float(step.sum()),
                                  step_p50=float(np.percentile(step, 50)),
                                  step_p99=float(np.percentile(step, 99)),
                                  strided_jump_frac=float(
                                      (np.linalg.norm(np.diff(truth_xy[::a.stride], axis=0),
                                                      axis=1) > cell).mean())))

        for mode in ('argmin', 'filter', 'viterbi'):
            ## the start anchor is the path's OWN first pose, which is what the rollout
            ## holds at this moment (`_pos()`); giving it the whole truth would make the
            ## accuracy number meaningless.
            shim = _DecodeShim(scanner, loc, mode, a.stride, a.median, a.motion_sigma,
                               max_step=a.vit_max_step, beam=a.vit_beam,
                               w_step=a.vit_w_step, sigma_e=a.vit_sigma_e,
                               anchor_xy=truth_xy[0])
            xy = shim.decode(scans)
            f, j, step = g2a_metrics(xy, scanner, cell)
            end_err = float(np.linalg.norm(xy[-1] - truth_xy[-1]))
            ## the planner's OWN stat, computed on the STRIDED decode, not the
            ## interpolated plan -- the gate and the rollout json do not agree here
            ## and the difference is a factor of 2 in step magnitude
            strided = float(shim._decode_stats[-1]['frac_cell_jumps'])
            err = np.linalg.norm(xy - truth_xy, axis=1)
            rows[mode].append(dict(i_p=i_p, free_frac=f, jump_frac=j,
                                   strided_jump_frac=strided,
                                   arc=float(step.sum()),
                                   step_p50=float(np.percentile(step, 50)),
                                   step_p99=float(np.percentile(step, 99)),
                                   err_p50=float(np.percentile(err, 50)),
                                   within_1cell=float((err <= cell).mean()),
                                   end_err=end_err))

        if (i_p + 1) % 5 == 0 or i_p == 0:
            t, g = rows['truth'][-1], rows['argmin'][-1]
            print(f'[g2a_floor] path {i_p:3d}  truth jump={t["jump_frac"]:.4f}  '
                  f'argmin jump={g["jump_frac"]:.4f} free={g["free_frac"]:.3f} '
                  f'err_p50={g["err_p50"]:.2f} wu', flush=True)

    def agg(name):
        r = rows[name]
        out = dict(arm=name, n=len(r))
        for k in r[0]:
            if k == 'i_p':
                continue
            v = np.array([x[k] for x in r], dtype=np.float64)
            out[k] = float(v.mean())
            out[k + '_p90'] = float(np.percentile(v, 90))
        return out

    summary = dict(
        env=ENV, tot_hzn=tot, cell=cell, n_paths=int(len(paths)),
        stride=a.stride, median=a.median, motion_sigma=a.motion_sigma,
        seed=a.seed,
        g2a_free_bar=G2A_FREE_FRAC, g2a_jump_bar=G2A_JUMP_FRAC,
        arms={k: agg(k) for k in arms},
    )

    print('\n================ G2a FLOOR (walkable-by-construction paths) ============')
    print(f'  bars: free_frac >= {G2A_FREE_FRAC}   jump_frac <= {G2A_JUMP_FRAC}\n')
    print(f'  {"arm":8s} {"free_frac":>10s} {"jump_frac":>10s} {"jump p90":>9s} '
          f'{"strided_j":>10s} {"arc":>8s} {"err_p50":>8s}')
    for k in arms:
        s = summary['arms'][k]
        print(f'  {k:8s} {s["free_frac"]:10.4f} {s["jump_frac"]:10.4f} '
              f'{s["jump_frac_p90"]:9.4f} {s["strided_jump_frac"]:10.4f} '
              f'{s["arc"]:8.1f} '
              f'{s.get("err_p50", float("nan")):8.2f}')
    print()
    for k in arms:
        s = summary['arms'][k]
        verdict = ('PASS' if (s['free_frac'] >= G2A_FREE_FRAC
                              and s['jump_frac'] <= G2A_JUMP_FRAC) else 'FAIL')
        print(f'  {k:8s} -> G2a {verdict}')
    print('========================================================================\n')

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(dict(summary=summary, per_path=rows), fh, indent=1)
    print(f'wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
