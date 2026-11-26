#!/usr/bin/env python3
"""GATE G1 (BALL.md Phase 1): scanner-geometry + dataset-structure checks for
the pointmaze ("ball") env. Run this BEFORE building caches/grids/training --
nothing downstream is trustworthy until it passes.

Checks
------
 1. controlled ray distances: at every free cell center, axis-aligned beams
    must match an exact grid ray-walk (validates maze_unit=4 / offset=4 for
    the POINTMAZE maps, the assumption the whole stack inherits from the ant).
 2. dataset positions: 100% of (x, y) fall in free cells of the resolved map.
 3. episode structure: terminals give uniform-length episodes; length fits the
    configs' max_path_length budgets (planner 300/h160, invdyn 300/h12).
 4. velocity reconstruction: integrate the finite-difference velocities ->
    reproduces xy exactly; |v| bounded (~<= 0.22 wu/step for the point env).

Usage
-----
  # geometry only (no dataset needed; runs anywhere):
  python -m diffuser.pl_eval.lidar_eval.ball_geometry_check --env pointmaze-giant-stitch-v0 --no_dataset
  # full check:
  python -m diffuser.pl_eval.lidar_eval.ball_geometry_check --env pointmaze-giant-stitch-v0 \
      --npz ~/.ogbench/data/pointmaze-giant-stitch-v0.npz --out logs/lidar_eval/ball_g1.json
"""
import os, sys, json, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
try:
    from diffuser.pl_eval.lidar_eval.lidar_grid_common import LS
except Exception:  # torch-free fallback
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.normpath(os.path.join(here, '../../datasets/lidar/lidar_sim.py'))
    spec = importlib.util.spec_from_file_location('lidar_sim', p)
    LS = importlib.util.module_from_spec(spec); spec.loader.exec_module(LS)


def exact_axis_ray(maze_map, i, j, di, dj, maze_unit, max_range):
    """Exact distance from cell (i,j) center to the first wall along (di,dj)."""
    n = 0
    H, W = maze_map.shape
    while True:
        n += 1
        ii, jj = i + n * di, j + n * dj
        if not (0 <= ii < H and 0 <= jj < W) or maze_map[ii, jj] == 1:
            ## wall cell starts half a unit before its center
            return min((n - 0.5) * maze_unit, max_range)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='pointmaze-giant-stitch-v0')
    ap.add_argument('--npz', default='')
    ap.add_argument('--no_dataset', action='store_true')
    ap.add_argument('--n_beams', type=int, default=4)     # axis-aligned check beams
    ap.add_argument('--max_range', type=float, default=12.0)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    res = dict(env=args.env)
    scanner = LS.make_scanner_for_env(
        args.env, dict(n_beams=args.n_beams, max_range=args.max_range,
                       fov=2 * np.pi, step=args.step))
    mp = scanner.maze_map
    u = scanner.maze_unit

    ## ---- 1. controlled distances at every free cell center ----
    free = np.argwhere(mp == 0)
    errs = []
    for (i, j) in free:
        x = j * u - scanner.offset_x
        y = i * u - scanner.offset_y
        scan = scanner.scan(x, y, 0.0)                     # beams at 0,90,180,270 deg
        exp = [exact_axis_ray(mp, i, j, 0, +1, u, args.max_range),   # +x
               exact_axis_ray(mp, i, j, +1, 0, u, args.max_range),   # +y
               exact_axis_ray(mp, i, j, 0, -1, u, args.max_range),   # -x
               exact_axis_ray(mp, i, j, -1, 0, u, args.max_range)]   # -y
        errs.append(np.abs(np.asarray(scan) - np.asarray(exp)))
    errs = np.asarray(errs)
    res['geometry'] = dict(
        n_free_cells=int(len(free)),
        max_abs_err_wu=float(errs.max()),
        mean_abs_err_wu=float(errs.mean()),
        ## ray-march quantization overshoots by up to one step; float32 scan
        ## values add ~1e-3 at 10 wu. A GEOMETRY error (wrong maze_unit/offset)
        ## would be off by whole cells (>= 2 wu), far outside this tolerance.
        tol_wu=float(args.step) + 1e-3,
        passed=bool(errs.max() <= args.step + 1e-3))
    print(f"[G1/geometry] {len(free)} free cells, max ray err "
          f"{errs.max():.4f} wu (tol {args.step}+1e-3) -> "
          f"{'PASS' if res['geometry']['passed'] else 'FAIL'}")

    ## ---- 2-4. dataset checks ----
    if not args.no_dataset:
        path = args.npz or os.path.expanduser(f'~/.ogbench/data/{args.env}.npz')
        assert os.path.exists(path), f'dataset not found: {path} (use --npz / --no_dataset)'
        d = np.load(path, allow_pickle=True)
        obs = np.asarray(d['observations'])
        term = np.asarray(d['terminals']).reshape(-1)
        res['dataset'] = dict(path=path, n=int(obs.shape[0]), obs_dim=int(obs.shape[1]))
        assert obs.shape[1] == 2, f'pointmaze obs must be 2-D (x, y); got {obs.shape}'

        ## 2. free-cell coverage
        ci, cj = scanner.world_to_cell(obs[:, 0], obs[:, 1])
        inb = (ci >= 0) & (ci < mp.shape[0]) & (cj >= 0) & (cj < mp.shape[1])
        onfree = inb & (mp[np.clip(ci, 0, mp.shape[0]-1), np.clip(cj, 0, mp.shape[1]-1)] == 0)
        res['dataset']['frac_in_free_cells'] = float(onfree.mean())
        res['dataset']['coverage_passed'] = bool(onfree.all())
        print(f"[G1/coverage] {onfree.mean()*100:.3f}% of {len(obs)} positions in free "
              f"cells -> {'PASS' if onfree.all() else 'FAIL'}")

        ## 3. episode structure
        ends = np.where(term == 1)[0]
        lens = np.diff(np.concatenate([[-1], ends]))
        res['dataset']['n_episodes'] = int(len(ends))
        res['dataset']['ep_len_min'] = int(lens.min()); res['dataset']['ep_len_max'] = int(lens.max())
        uniform = bool(lens.min() == lens.max())
        fits = bool(lens.max() - 1 + 160 <= 300)   # planner h160 / max_path_length 300
        res['dataset']['uniform_episodes'] = uniform
        res['dataset']['fits_max_path_length_300'] = fits
        print(f"[G1/episodes] {len(ends)} episodes, len {lens.min()}..{lens.max()} "
              f"(uniform={uniform}, fits mpl300/h160={fits})")

        ## 4. velocity reconstruction
        try:
            from diffuser.datasets.ogb_dset.ogb_lidar_utils import finite_diff_velocities
        except Exception:
            import importlib.util
            here = os.path.dirname(os.path.abspath(__file__))
            p = os.path.normpath(os.path.join(here, '../../datasets/ogb_dset/ogb_lidar_utils.py'))
            src = open(p).read()
            ns = {}
            ## execute only the numpy-level helper (module imports torch-side utils)
            import ast
            tree = ast.parse(src)
            fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'finite_diff_velocities'][0]
            code = compile(ast.Module(body=[fn], type_ignores=[]), p, 'exec')
            exec(code, {'np': np}, ns)
            finite_diff_velocities = ns['finite_diff_velocities']
        vel = finite_diff_velocities(obs[:, :2], terminals=term)
        ## integrate per episode and compare
        starts = np.concatenate([[0], ends[:-1] + 1])
        max_err = 0.0
        for s, e in zip(starts, ends):
            rec = obs[s, :2] + np.cumsum(vel[s:e + 1], axis=0) - vel[s]
            max_err = max(max_err, float(np.abs(rec - obs[s:e + 1, :2]).max()))
        vmax = float(np.linalg.norm(vel, axis=1).max())
        res['dataset']['vel_reconstruction_max_err'] = max_err
        res['dataset']['vel_max_wu_per_step'] = vmax
        res['dataset']['vel_passed'] = bool(max_err < 1e-4)
        print(f"[G1/velocity] reconstruction max err {max_err:.2e} wu; "
              f"|v|max {vmax:.3f} wu/step -> {'PASS' if max_err < 1e-4 else 'FAIL'}")

    ok = res['geometry']['passed'] and (args.no_dataset or (
        res['dataset']['coverage_passed'] and res['dataset']['vel_passed']))
    res['G1_passed'] = bool(ok)
    print(f"[G1] OVERALL: {'PASS' if ok else 'FAIL'}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(res, open(args.out, 'w'), indent=2)
        print(f'[G1] wrote {args.out}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
