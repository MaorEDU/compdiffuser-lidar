"""G2 -- the offline plan-decode gate. Run it on every checkpoint; run NOTHING
closed-loop until it passes.

It answers the question that has to be settled before a single rollout GPU-hour is
spent: conditioned on (start scan, goal scan[, history]), does the composed planner
produce a scan sequence that decodes to a connected, collision-free route ending at
the goal? The previous fork found out its planner was producing locally plausible but
globally incoherent plans only after weeks of closed-loop debugging.

Metrics, per problem, over ``--n_samp`` resamples:

  G2a  validity   -- share of decoded frames in FREE cells, and share of consecutive
                     frames whose step exceeds one maze cell ("jumps"). A healthy plan
                     is a path the ball could physically follow.
  G2b  anchoring  -- distance from the decoded ENDPOINT to the true goal, and the
                     share of endpoints within one cell of it.
  G2c  diversity  -- spread of endpoints across resamples of the same problem, i.e.
                     whether the planner is a sampler of one route or of many.

Pre-registered pass: **G2a >= 95% free-cell frames with ~0 jumps, and G2b >= 80% of
endpoints within one cell of the goal.** G2b failing while G2a passes means the
planner generates plausible-but-unanchored scans -- a conditioning problem, not an
executor one, and not something a rollout can fix.

CAVEAT, stated because the previous fork was burned by exactly this: the decode is the
forward FILTER, and a filter's per-step argmax can hop between posterior modes. A G2
number therefore measures *plan + decoder*, never the plan alone. Never compare G2
numbers computed with different decoders.

``--hist 1`` conditions each plan on a real history window taken from a validation
episode that passes through the start pose, which is the closest offline analogue of
what the rollout supplies at a replan. Running the gate both ways on the same
checkpoint is the cheapest read on whether the history channel does anything.

Usage:
    python -m diffuser.pl_eval.lidar_eval.plan_decode_gate \
        --config config/ogb_pnt_maze/og_pntM_Gi_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512.py \
        --n_prob 20 --n_samp 8 --hist 0 --out logs/lidar_eval/g2_hist0.json
"""

import argparse
import json
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.append('./')

import numpy as np


## pre-registered thresholds -- change these only with a written reason
G2A_FREE_FRAC = 0.95
G2A_JUMP_FRAC = 0.01
G2B_ANCHOR = 0.80


def _parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--n_prob', type=int, default=100,
                   help='eval problems (of 100). NOT 20: ev_probs is 5 blocks of 20 and '
                        'each block is ONE start-goal pair jittered, so n_prob=20 measures '
                        'a single route -- house rule #1.')
    p.add_argument('--ep_st_idx', type=int, default=0)
    p.add_argument('--decode', default=None, choices=['argmin', 'filter', 'viterbi'],
                   help='override the plan decoder; default = the planner default. '
                        'BALLV2 section 5: never compare numbers across decoders.')
    p.add_argument('--n_samp', type=int, default=8, help='resamples per problem')
    p.add_argument('--hist', type=int, default=0,
                   help='1 = condition on a real history window from the val set')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='')
    return p.parse_args()


def _build_planner(cfg_path, hist_on):
    """Construct the LiDAR ball planner exactly as the rollout does.

    Reusing the planner (rather than re-implementing model loading) is what keeps the
    gate honest: it samples through the same policy, decodes through the same grid,
    and normalizes through the same normalizer as the run it is gating."""
    import diffuser.utils as utils
    from diffuser.ogb_task.ogb_maze_v1.plan_ogb_stgl_sml import Parser, setup_eval_args
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1,
    )

    ## the gate never steps the env, so pin the arm to the oracle position source and
    ## the paper's replan trigger; only the history knob is under test here.
    os.environ['LIDAR_LOC_SOURCE'] = 'env'
    os.environ['LIDAR_BALL_HIST'] = str(int(hist_on))
    os.environ.setdefault('LIDAR_REPL_EVERY', '0')

    argv_bak = sys.argv
    sys.argv = ['plan_decode_gate', '--config', cfg_path]
    try:
        args_train = Parser().parse_args('diffusion')
        args = Parser().parse_args('plan')
    finally:
        sys.argv = argv_bak

    assert args_train.dataset_config.get('lidar_config', None) is not None, (
        f'{cfg_path} is not a LiDAR config -- G2 decodes SCAN plans')
    args.pl_seeds = utils.parse_seeds_str(args.pl_seeds)
    args.n_batch_acc_probs = 4
    setup_eval_args(args_train, args)
    args.env_n_max_steps = args.repl_ada_dist_cfg['n_max_steps']

    latest_e = utils.get_latest_epoch((args.logbase, args.dataset, args_train.exp_name))
    args_train.diffusion_epoch = latest_e
    args.diffusion_epoch = latest_e

    planner = OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1(args_train, args=args)
    planner.setup_load(ld_config=dict())
    return planner, args, latest_e


def _val_history(planner, start_xy, hist_len):
    """A real ``hist_len``-frame scan history ending at ``start_xy``.

    Taken from the validation episodes: find the frame nearest the start pose and
    re-render the scans of the ``hist_len`` frames before it. This is the offline
    stand-in for what the rollout's deque holds at a replan -- real trajectory frames,
    not a synthesized approach."""
    from diffuser.datasets.lidar.hist_window import left_pad_history

    cache = getattr(planner, '_gate_val_xy', None)
    if cache is None:
        path = os.path.expanduser(f'~/.ogbench/data/{planner.env.name}-val.npz')
        d = np.load(path)
        cache = (np.asarray(d['observations'], dtype=np.float64)[:, :2],
                 np.asarray(d['terminals']).reshape(-1).astype(bool))
        planner._gate_val_xy = cache
    xy_all, term = cache

    i = int(np.argmin(np.linalg.norm(xy_all - np.asarray(start_xy)[None], axis=1)))
    ## do not walk back across an episode boundary: the history must be one trajectory
    lo = i
    for k in range(1, hist_len + 1):
        if i - k < 0 or term[i - k]:
            break
        lo = i - k
    frames = np.stack([planner.scanner.scan(float(x), float(y), 0.0)
                       for x, y in xy_all[lo:i]], axis=0) if lo < i \
        else np.empty((0, planner.n_beams), dtype=np.float32)
    fill = planner.scanner.scan(float(start_xy[0]), float(start_xy[1]), 0.0)
    return left_pad_history(frames, hist_len, fill=fill)


def main():
    a = _parse()
    import diffuser.utils as utils

    planner, args, epoch = _build_planner(a.config, a.hist)
    if a.decode:
        planner._decode_mode = a.decode
    print(f'[gate] plan decoder = {planner._decode_mode}')
    utils.set_seed(a.seed)

    probs = planner.problems_dict
    scanner, loc = planner.scanner, planner.loc
    cell = float(scanner.maze_unit)
    hist_len = planner.hist_len if a.hist else 0
    if a.hist:
        assert hist_len > 0, (
            '--hist 1 but this checkpoint has no history encoder')

    rows = []
    for i_ep in range(a.ep_st_idx, a.ep_st_idx + a.n_prob):
        st_xy = np.asarray(probs['start_state'][i_ep], dtype=np.float64)[:2]
        gl_xy = np.asarray(probs['goal_pos'][i_ep], dtype=np.float64)[:2]
        st_scan = scanner.scan(float(st_xy[0]), float(st_xy[1]), 0.0).astype(np.float32)
        gl_scan = scanner.scan(float(gl_xy[0]), float(gl_xy[1]), 0.0).astype(np.float32)

        g_cond = {'st_gl': np.stack([st_scan, gl_scan])[:, None, :].astype(np.float32)}
        if a.hist:
            hw = _val_history(planner, st_xy, hist_len)
            ## `planner._gen_cond_stgl` WRAPS policy.gen_cond_stgl and overwrites
            ## g_cond['hist'] from the planner's rollout deque. Offline there is no
            ## episode, so that deque is empty and the wrapper replaced this window
            ## with a (1, 0) array -- the gate's own history never reached the model.
            ## Seed the deque instead: the wrapper then rebuilds exactly `hw`, and the
            ## gate keeps going through the same code path the rollout uses.
            planner._hist.clear()
            planner._hist.extend(hw)
            g_cond['hist'] = hw[None]

        ends, frees, jumps = [], [], []
        for _ in range(a.n_samp):
            ## planner._gen_cond_stgl decodes into planner._plan_xy as a side effect,
            ## through the same code path the rollout uses
            planner.policy.gen_cond_stgl(g_cond=dict(g_cond), b_s=args.b_size_per_prob)
            xy = planner._plan_xy
            if xy is None:
                continue
            ii, jj = scanner.world_to_cell(xy[:, 0], xy[:, 1])
            frees.append(float((~scanner._is_wall(ii, jj)).mean()))
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            jumps.append(float((step > cell).mean()) if len(step) else 0.0)
            ends.append(xy[-1].copy())

        if not ends:
            rows.append(dict(i_ep=i_ep, decode_failed=True))
            continue
        ends = np.asarray(ends)
        end_err = np.linalg.norm(ends - gl_xy[None], axis=1)
        rows.append(dict(
            i_ep=i_ep,
            n_samp=int(len(ends)),
            free_frac=float(np.mean(frees)),
            jump_frac=float(np.mean(jumps)),
            end_err_p50=float(np.percentile(end_err, 50)),
            anchored_frac=float((end_err <= cell).mean()),
            ## G2c: how far apart the resampled endpoints are from each other
            end_spread=float(np.mean(np.linalg.norm(ends - ends.mean(0)[None], axis=1))),
        ))
        print(f'[G2] ep {i_ep:3d}  free={rows[-1]["free_frac"]:.3f} '
              f'jump={rows[-1]["jump_frac"]:.3f} '
              f'endErr={rows[-1]["end_err_p50"]:6.2f} wu '
              f'anchored={rows[-1]["anchored_frac"]:.2f} '
              f'spread={rows[-1]["end_spread"]:6.2f}', flush=True)

    ok = [r for r in rows if not r.get('decode_failed')]
    agg = dict(
        config=a.config, epoch=int(epoch), decode=str(planner._decode_mode), hist=bool(a.hist), hist_len=int(hist_len),
        n_prob=len(rows), n_decoded=len(ok), n_samp=a.n_samp, seed=a.seed,
        g2a_free_frac=float(np.mean([r['free_frac'] for r in ok])) if ok else 0.0,
        g2a_jump_frac=float(np.mean([r['jump_frac'] for r in ok])) if ok else 1.0,
        g2b_anchored=float(np.mean([r['anchored_frac'] for r in ok])) if ok else 0.0,
        g2b_end_err_p50=float(np.median([r['end_err_p50'] for r in ok])) if ok else 1e9,
        g2c_end_spread=float(np.mean([r['end_spread'] for r in ok])) if ok else 0.0,
    )
    agg['g2a_pass'] = bool(agg['g2a_free_frac'] >= G2A_FREE_FRAC
                           and agg['g2a_jump_frac'] <= G2A_JUMP_FRAC)
    agg['g2b_pass'] = bool(agg['g2b_anchored'] >= G2B_ANCHOR)
    agg['pass'] = bool(agg['g2a_pass'] and agg['g2b_pass'])

    print('\n================ G2 ================')
    print(f"  G2a validity  free={agg['g2a_free_frac']:.3f} (>= {G2A_FREE_FRAC}) "
          f"jump={agg['g2a_jump_frac']:.3f} (<= {G2A_JUMP_FRAC})  "
          f"-> {'PASS' if agg['g2a_pass'] else 'FAIL'}")
    print(f"  G2b anchoring anchored={agg['g2b_anchored']:.3f} (>= {G2B_ANCHOR}) "
          f"endErr_p50={agg['g2b_end_err_p50']:.2f} wu  "
          f"-> {'PASS' if agg['g2b_pass'] else 'FAIL'}")
    print(f"  G2c diversity endpoint spread = {agg['g2c_end_spread']:.2f} wu")
    if not agg['pass']:
        print('  ** GATE FAILED -- do not start closed-loop rollouts. **')
        if agg['g2a_pass'] and not agg['g2b_pass']:
            print('  G2a passed, G2b did not: the plans are walkable but not aimed. '
                  'That is a conditioning problem; a rollout cannot fix it.')
    print('====================================\n')

    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(dict(summary=agg, per_problem=rows), f, indent=2)
        print(f'wrote {a.out}')
    return 0 if agg['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
