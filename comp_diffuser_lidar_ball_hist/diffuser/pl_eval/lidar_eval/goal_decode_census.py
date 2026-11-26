"""Is the goal SCAN able to identify the goal CELL at all?

G2b failed hard on the giant checkpoint: plans are walkable (99.2% free-cell frames)
but only 25.6% of endpoints land within a cell of the goal, median endpoint error
35.45 wu. That is either

  (a) the planner failing to aim at a target it could see, or
  (b) the target not being identifiable from the conditioning it is given -- 30 beams
      clamped at max_range 12 in a giant maze, where BALLV2's pre-registered non-choice
      #2 already records 63.6% aliasing.

This settles it WITHOUT the model, on CPU, which is the whole point: the planner is
conditioned on the goal scan, so if the goal scan does not localize, no amount of
planner work can anchor the plan and the fix is a sensing decision, not a training one.

Per eval problem it reports, for the goal scan alone (and the start scan, which is
what the L3 belief seed rides on):

  * decoded cell and its error to the true pose, in world units
  * the RANK of the true cell under the scan likelihood -- rank 1 means the scan picks
    the right cell outright; a large rank means look-alikes beat it
  * whether the true cell is inside the top-k

Run at several ``--max_range`` values in one go to price the sensing fix directly.
BALLV2 declines max_range 12 -> 24 as a non-choice for comparability; this is the
"revisit only with evidence" instrument.

Usage:
    python -m diffuser.pl_eval.lidar_eval.goal_decode_census --max_range 12,24
"""
import argparse, json, os, sys
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.append('./')

import numpy as np

ENV = 'pointmaze-giant-stitch-v0'
BASE_LIDAR = dict(n_beams=30, max_range=12.0, fov=2.0 * np.pi, step=0.1,
                  obs_layout='lidar', use_cache=True)


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--max_range', default='12,24', help='comma list of max_range values')
    p.add_argument('--n_prob', type=int, default=100)
    p.add_argument('--xy_step', type=float, default=0.5)
    p.add_argument('--n_theta', type=int, default=1)
    p.add_argument('--sigma', type=float, default=1.0)
    p.add_argument('--out', default='logs/lidar_eval/goal_decode_census.json')
    return p.parse_args()


def census(env_name, lidar_cfg, probs, n_prob, xy_step, n_theta, sigma):
    from diffuser.datasets.lidar.lidar_grid import load_or_build_grid
    from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env
    from diffuser.datasets.lidar.lidar_localization import GridLocalizer, scan_distance

    grid = load_or_build_grid(env_name, lidar_config=lidar_cfg,
                              xy_step=xy_step, n_theta=n_theta, verbose=True)
    loc = GridLocalizer(grid, sigma=sigma)
    scanner = make_scanner_for_env(env_name, lidar_cfg)
    mu = float(grid.maze_unit)
    fxy = grid.free_xy.astype(np.float64)

    def one(xy):
        """decode error (wu) and rank of the true CELL for the scan taken at xy."""
        scan = scanner.scan(float(xy[0]), float(xy[1]), 0.0).astype(np.float32)
        d = scan_distance(scan, grid.free_scans)[0]                 # (M,)
        k = int(np.argmin(d))
        dec_xy = fxy[k]
        err = float(np.linalg.norm(dec_xy - np.asarray(xy, dtype=np.float64)))
        ## rank of the true cell: collapse poses to cells, rank cells by best distance
        cell_ij = np.floor(fxy / mu).astype(np.int64)
        true_ij = np.floor(np.asarray(xy, dtype=np.float64) / mu).astype(np.int64)
        same = (cell_ij[:, 0] == true_ij[0]) & (cell_ij[:, 1] == true_ij[1])
        if not same.any():
            return err, -1, float(d[k])
        best_true = float(d[same].min())
        ## how many DISTINCT cells score strictly better than the true cell
        better = d < best_true
        if better.any():
            bij = cell_ij[better]
            n_better = len(np.unique(bij[:, 0] * 100000 + bij[:, 1]))
        else:
            n_better = 0
        return err, int(n_better) + 1, float(d[k])

    rows = []
    for i in range(min(n_prob, len(probs['goal_pos']))):
        g = np.asarray(probs['goal_pos'][i], dtype=np.float64)[:2]
        s = np.asarray(probs['start_state'][i], dtype=np.float64)[:2]
        ge, gr, gd = one(g)
        se, sr, sd = one(s)
        rows.append(dict(i=i, goal_err=ge, goal_rank=gr, start_err=se, start_rank=sr))

    ge = np.array([r['goal_err'] for r in rows])
    gr = np.array([r['goal_rank'] for r in rows])
    se = np.array([r['start_err'] for r in rows])
    sr = np.array([r['start_rank'] for r in rows])

    ## aliasing census over the whole free space, the number BALLV2 quotes
    dm = loc.distinctiveness_map(min_cell_gap=3.0, reduce='mean')
    fine = dm[~np.isnan(dm)]
    alias_frac = float((fine < sigma).mean())

    return dict(
        max_range=float(lidar_cfg['max_range']), n_prob=len(rows),
        n_free_xy=int(grid.n_free_xy), maze_unit=mu,
        goal_err_p50=float(np.percentile(ge, 50)), goal_err_p90=float(np.percentile(ge, 90)),
        goal_within_1cell=float((ge <= mu).mean()), goal_rank1=float((gr == 1).mean()),
        goal_rank_p50=float(np.percentile(gr, 50)), goal_rank_p90=float(np.percentile(gr, 90)),
        start_err_p50=float(np.percentile(se, 50)),
        start_within_1cell=float((se <= mu).mean()), start_rank1=float((sr == 1).mean()),
        alias_frac=alias_frac,
        per_problem=rows,
    )


def main():
    a = _parse()
    import diffuser.utils as utils
    probs = utils.load_stgl_lh_ev_probs_hdf5(
        h5path=utils.get_ogb_maze_ev_probs_fname(ENV))
    print(f'loaded {len(probs["goal_pos"])} eval problems for {ENV}')

    out = []
    for mr in [float(x) for x in a.max_range.split(',')]:
        cfg = dict(BASE_LIDAR); cfg['max_range'] = mr
        print(f'\n---------------- max_range = {mr:g} wu ----------------', flush=True)
        r = census(ENV, cfg, probs, a.n_prob, a.xy_step, a.n_theta, a.sigma)
        out.append(r)
        print(f"  free (x,y) lattice points : {r['n_free_xy']}  (1 cell = {r['maze_unit']:g} wu)")
        print(f"  GOAL  scan -> decoded cell: err_p50={r['goal_err_p50']:6.2f} wu   "
              f"err_p90={r['goal_err_p90']:6.2f} wu   within 1 cell: {r['goal_within_1cell']*100:5.1f}%")
        print(f"  GOAL  rank of TRUE cell   : rank1={r['goal_rank1']*100:5.1f}%   "
              f"p50={r['goal_rank_p50']:.0f}   p90={r['goal_rank_p90']:.0f}")
        print(f"  START scan -> decoded cell: err_p50={r['start_err_p50']:6.2f} wu   "
              f"within 1 cell: {r['start_within_1cell']*100:5.1f}%   rank1={r['start_rank1']*100:5.1f}%")
        print(f"  aliasing (free cells with a far look-alike within sigma): "
              f"{r['alias_frac']*100:5.1f}%")

    print('\n================== READ ==================')
    b = out[0]
    print(f"  G2b measured 25.6% of PLAN endpoints within a cell of the goal.")
    print(f"  The goal SCAN alone puts {b['goal_within_1cell']*100:.1f}% within a cell "
          f"at max_range={b['max_range']:g}.")
    if b['goal_within_1cell'] < 0.5:
        print("  -> the target is not identifiable from its scan. This is a SENSING")
        print("     limit, not planner work; G2b cannot exceed it by construction.")
    else:
        print("  -> the goal scan DOES localize; the planner is failing to aim at a")
        print("     target it can see. That is planner/conditioning work.")
    if len(out) > 1:
        for r in out[1:]:
            print(f"  max_range {r['max_range']:g}: goal within 1 cell "
                  f"{r['goal_within_1cell']*100:.1f}%  rank1 {r['goal_rank1']*100:.1f}%  "
                  f"aliasing {r['alias_frac']*100:.1f}%")
    print('==========================================')

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
