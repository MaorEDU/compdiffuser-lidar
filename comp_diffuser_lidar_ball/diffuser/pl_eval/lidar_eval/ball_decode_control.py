#!/usr/bin/env python3
"""Ground-truth decode control — is G2a measuring the MODEL or the LOCALIZER?

BALL.md's G2a (jump-free decoded plans) is only a statement about the planner if
the grid decode is itself clean on GROUND-TRUTH scans. That was verified once, at
max_range=12: 95% jump-free, median err 0.17 wu, GridLocalizer(sigma=1.0),
motion_sigma_cells=1.0. **No mr24 equivalent existed**, and sigma is an ABSOLUTE
per-beam tolerance in world units — doubling the scan range halves the relative
tolerance, which can sharpen the posterior enough to manufacture MAP jumps on
imperfect scans. If that happens, "mr24 G2a = 0%" is a localizer artifact and the
mr24-vs-mr12 continuity comparison is void.

This script decodes N ground-truth windows straight from the LiDAR npz and sweeps
sigma, so one run answers both "is the decode clean at this range?" and "is sigma
mis-scaled for this range?".

Deliberately standalone and pure numpy: no dataset build, no OGBench env, no
MuJoCo, no torch beyond the optional ckpt-config read -> runs on a GPU-less login
node. ``load_or_build_grid`` takes an env NAME and ``get_maze_map``
is a string->array lookup, so nothing here touches a simulator.

  python -m diffuser.pl_eval.lidar_eval.ball_decode_control \
      --env pointmaze-giant-stitch-v0 \
      --npz data/ogb_maze/lidar/pointmaze-giant-stitch-v0_lidar_r24.npz \
      --planner_logdir logs/pointmaze-giant-stitch-v0/diffusion/og_pntM_Gi_lidar30r24_... \
      --decode 20
"""
import os, sys, json, argparse, importlib.util
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))

C_OK, C_BAD, C_INFO, C_OFF = '\033[92m', '\033[91m', '\033[96m', '\033[0m'

## mr12 reference, validated 2026-07-18 (BALL_SESSIONS_LOG §1)
REF_MR12 = dict(jump_free=0.95, med_err_wu=0.17, sigma=1.0, motion_sigma=1.0)
DEFAULT_LIDAR_CONFIG = dict(n_beams=30, max_range=12.0, fov=2.0 * np.pi, step=0.1)


def _load(mod_name, rel_path):
    """Import our numpy-only lib modules without dragging in the torch package."""
    try:
        return __import__(mod_name, fromlist=['*'])
    except Exception:  # noqa: BLE001
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, rel_path))
        spec = importlib.util.spec_from_file_location(os.path.basename(rel_path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


LG = _load('diffuser.datasets.lidar.lidar_grid', '../../datasets/lidar/lidar_grid.py')
LL = _load('diffuser.datasets.lidar.lidar_localization', '../../datasets/lidar/lidar_localization.py')


def resolve_geometry(logdir, max_range, n_beams):
    """Geometry from the CHECKPOINT. The trainer saves the dataset Config as
    ``utils.Config(..., dataset_config=args.dataset_config)``, so the scan config
    lives at ``cfg['dataset_config']['lidar_config']`` — NOT ``cfg['lidar_config']``.
    Reading the wrong key falls back to the mr12 default silently; that is exactly
    how the 2026-07-29 dump printed "MODULE DEFAULT (unverified!)"."""
    cfg, src = None, ''
    if logdir:
        try:
            from diffuser.utils import load_config
            c = load_config(logdir, 'dataset_config.pkl')
            try:
                cfg, src = dict(c['dataset_config']['lidar_config']), 'ckpt dataset_config.pkl'
            except Exception:  # noqa: BLE001
                cfg, src = dict(c['lidar_config']), 'ckpt dataset_config.pkl (flat)'
        except Exception as e:  # noqa: BLE001
            print(f'{C_BAD}[audit] could not read lidar_config from {logdir}: {e!r}{C_OFF}')
    if cfg is None:
        cfg, src = dict(DEFAULT_LIDAR_CONFIG), 'MODULE DEFAULT (unverified!)'
    cfg = {**DEFAULT_LIDAR_CONFIG, **cfg}
    if n_beams:
        cfg['n_beams'], src = int(n_beams), src + ' + CLI n_beams'
    if max_range:
        cfg['max_range'], src = float(max_range), src + ' + CLI max_range'
    cfg['n_beams'], cfg['max_range'] = int(cfg['n_beams']), float(cfg['max_range'])
    ok = 'DEFAULT' not in src
    print(f"{C_OK if ok else C_BAD}[audit] lidar_config from {src}: n_beams={cfg['n_beams']} "
          f"max_range={cfg['max_range']} -> grid cache must say mr{int(cfg['max_range'])}{C_OFF}")
    return cfg


def ep_bounds(terminals):
    ends = np.where(np.asarray(terminals).reshape(-1).astype(bool))[0]
    out, prev = [], 0
    for e in ends:
        out.append((prev, e + 1)); prev = e + 1
    if prev < len(terminals):
        out.append((prev, len(terminals)))
    return out


def decode_windows(grid, lidar, xy, wins, sigma, motion_sigma, stride):
    loc = LL.GridLocalizer(grid, sigma=sigma)
    unit = grid.scanner.maze_unit
    njs, errs = [], []
    for (s0, H) in wins:
        idx = np.unique(np.concatenate([np.arange(0, H, stride), [H - 1]]))
        sl = slice(s0, s0 + H)
        r = loc.localize_sequence(lidar[sl][idx].astype(np.float32),
                                  motion_sigma_cells=motion_sigma, theta_blur=0)
        m = np.asarray(r['map_xy'], dtype=np.float64)
        njs.append(int((np.linalg.norm(np.diff(m, axis=0), axis=1) > unit).sum()))
        errs.append(float(np.median(np.linalg.norm(m - xy[sl][idx], axis=1))))
    njs = np.asarray(njs)
    return dict(n=len(njs), jump_free=float((njs == 0).mean()),
                med_jumps=float(np.median(njs)), med_err_wu=float(np.median(errs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='pointmaze-giant-stitch-v0')
    ap.add_argument('--npz', required=True, help='LiDAR npz from gen_lidar_dataset (its max_range must match the ckpt)')
    ap.add_argument('--planner_logdir', default='', help='ckpt run dir — geometry source')
    ap.add_argument('--max_range', type=float, default=0.0, help='override (0 = read from ckpt)')
    ap.add_argument('--n_beams', type=int, default=0)
    ap.add_argument('--decode', type=int, default=20, help='ground-truth windows to decode')
    ap.add_argument('--horizon', type=int, default=160)
    ap.add_argument('--stride', type=int, default=2, help='must match g2_check (2)')
    ap.add_argument('--sigmas', default='0.5,1.0,2.0,4.0', help='measurement sigma sweep')
    ap.add_argument('--motion_sigma', type=float, default=1.0)
    ap.add_argument('--xy_step', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='logs/lidar_eval/ball_decode_control.json')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = resolve_geometry(args.planner_logdir, args.max_range, args.n_beams)

    d = np.load(os.path.expanduser(args.npz))
    lidar, xy = d['lidar'].astype(np.float32), d['xy'].astype(np.float32)
    obs_max = float(lidar.max())
    assert abs(obs_max - cfg['max_range']) < 0.5, (
        f'npz max scan {obs_max:.2f} wu != model max_range {cfg["max_range"]:.1f} — wrong npz, '
        f'regenerate with gen_lidar_dataset --max_range {cfg["max_range"]:.0f}')
    assert lidar.shape[-1] == cfg['n_beams'], f'npz has {lidar.shape[-1]} beams, ckpt wants {cfg["n_beams"]}'
    print(f'{C_OK}[audit] npz OK: max scan {obs_max:.2f} wu == max_range {cfg["max_range"]:.1f}{C_OFF}')

    ## per-step scan roughness — the quantity that competes with pose aliasing
    bounds = ep_bounds(d['terminals'])
    st = np.concatenate([np.linalg.norm(np.diff(lidar[s:e].astype(np.float64), axis=0), axis=1)
                         for s, e in bounds])
    print(f'{C_INFO}[roughness] scan step p50/p99.9/max = {np.median(st):.3f}/'
          f'{np.quantile(st, 0.999):.3f}/{st.max():.3f} wu   (mr12 ref: 1.47/13.3/17.6){C_OFF}')

    grid = LG.load_or_build_grid(args.env, cfg, xy_step=args.xy_step, n_theta=1)

    H = args.horizon
    ok = [(s, e) for s, e in bounds if e - s >= H]
    wins = [(int(rng.integers(s, e - H + 1)), H)
            for s, e in [ok[i] for i in rng.choice(len(ok), size=min(args.decode, len(ok)), replace=False)]]

    res = dict(npz=args.npz, lidar_config={k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                                           for k, v in cfg.items()},
               n_windows=len(wins), horizon=H, stride=args.stride,
               motion_sigma=args.motion_sigma, ref_mr12=REF_MR12, sweep={})
    print(f'{C_INFO}[decode] {len(wins)} ground-truth windows x {H} frames, stride {args.stride}, '
          f'motion_sigma={args.motion_sigma}{C_OFF}')
    best = None
    for sig in [float(s) for s in args.sigmas.split(',') if s.strip()]:
        r = decode_windows(grid, lidar, xy, wins, sig, args.motion_sigma, args.stride)
        res['sweep'][f'{sig:g}'] = r
        good = r['jump_free'] > 0.9
        star = ' <-- the setting G2 used' if abs(sig - 1.0) < 1e-9 else ''
        print(f"{C_OK if good else C_BAD}[decode] sigma={sig:<5g} jump-free {100 * r['jump_free']:3.0f}% | "
              f"median jumps {r['med_jumps']:.1f} | median err {r['med_err_wu']:.2f} wu{star}{C_OFF}")
        if best is None or (r['jump_free'], -r['med_err_wu']) > (best[1]['jump_free'], -best[1]['med_err_wu']):
            best = (sig, r)

    at1 = res['sweep'].get('1', res['sweep'].get('1.0'))
    res['best_sigma'], res['verdict_clean_at_sigma1'] = best[0], bool(at1 and at1['jump_free'] > 0.9)
    if at1 and at1['jump_free'] > 0.9:
        v = ('DECODE CLEAN at sigma=1.0 -> G2a measures the MODEL. The mr24 continuity result '
             'stands: longer range did not reduce scan-space teleports.')
        col = C_OK
    elif best[1]['jump_free'] > 0.9:
        v = (f"DECODE DIRTY at sigma=1.0 but CLEAN at sigma={best[0]:g} -> G2a is a LOCALIZER "
             f"artifact at this range, not a model result. Re-score G2 offline with sigma={best[0]:g} "
             f"(ball_dump_plans --g2_npz, CPU) BEFORE any mr24 conclusion.")
        col = C_BAD
    else:
        v = ('DECODE DIRTY AT EVERY SIGMA on ground-truth scans -> the grid decode does not work at '
             'this range; G2a is uninterpretable. Investigate motion_sigma / xy_step / grid build '
             'before trusting any mr24 number.')
        col = C_BAD
    print(f'{col}[VERDICT] {v}{C_OFF}')
    res['verdict'] = v
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'{C_INFO}[decode] wrote {args.out}{C_OFF}')


if __name__ == '__main__':
    main()
