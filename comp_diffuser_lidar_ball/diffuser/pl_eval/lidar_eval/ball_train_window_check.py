#!/usr/bin/env python3
"""Post-G2 triage: is the plan incoherence coming from the TRAINING DATA?

G2 verdict on the benched ckpt (2026-07-18): 0% jump-free plans, jumps spread
uniformly inside segments (24% near seams = the uniform baseline), G2b 24%,
route mode-share 0.03 (= every resample a different route). The generated scan
sequences are locally plausible but globally incoherent. Candidate sources:

  A. the lidar npz itself (transform-level): within-episode scan/xy step stats
  B. the training buffer (segmentation / terminals threading / PadBuf): step
     stats inside every buffer path + alignment of buffer paths to raw episodes
     + sampled __getitem__ windows (catches get_item-time padding bugs)
  C. the grid decode inside the training distribution (--decode N): ground-truth
     windows must decode jump-free (offline round-trip claimed 94% same-cell)

A+B+C clean  => data pipeline exonerated => the MODEL failed to learn global
structure (training length / objective / composition config) — see analysis md.
B dirty      => fix ogb_utils segmentation/padding, retrain (medium smoke first).
C dirty      => decode/localizer settings implicated too (tighten motion sigma).

CPU only. Run from repo root on the cluster:

  python -m diffuser.pl_eval.lidar_eval.ball_train_window_check \
      --npz data/ogb_maze/lidar/pointmaze-giant-stitch-v0_lidar.npz \
      --planner_logdir <the run dir the dump resolved> \
      --decode 20 \
      --out logs/lidar_eval/ball_window_check.json
"""
import os, sys, json, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
import diffuser.utils as utils
from diffuser.utils import load_config
## sensing geometry -- resolved from the checkpoint, never hardcoded here
## (this tool used to pin mr12; see lidar_grid_common.resolve_lidar_config).
from diffuser.pl_eval.lidar_eval.lidar_grid_common import resolve_lidar_config


def ep_bounds(terminals):
    ends = np.where(terminals)[0]
    bounds, prev = [], 0
    for e in ends:
        bounds.append((prev, e + 1)); prev = e + 1
    if prev < len(terminals):
        bounds.append((prev, len(terminals)))
    return bounds


def steps(a):
    return np.linalg.norm(np.diff(np.asarray(a, np.float64), axis=0), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--planner_logdir', required=True)
    ap.add_argument('--env', default='pointmaze-giant-stitch-v0')
    ap.add_argument('--n_items', type=int, default=400, help='__getitem__ windows to sample')
    ap.add_argument('--decode', type=int, default=0, help='grid-decode N ground-truth windows')
    ap.add_argument('--horizon', type=int, default=160)
    ap.add_argument('--xy_step', type=float, default=0.5)
    ap.add_argument('--n_beams', type=int, default=0, help='0 = from the checkpoint')
    ap.add_argument('--max_range', type=float, default=0.0, help='0 = from the checkpoint')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='logs/lidar_eval/ball_window_check.json')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    res = {}

    ## ================= A. raw npz =================
    d = np.load(args.npz)
    lidar, xy = d['lidar'].astype(np.float32), d['xy'].astype(np.float32)
    terminals = d['terminals'].astype(bool)
    bounds = ep_bounds(terminals)
    sc_steps = np.concatenate([steps(lidar[s:e]) for s, e in bounds])
    xy_steps = np.concatenate([steps(xy[s:e]) for s, e in bounds])
    lens = np.array([e - s for s, e in bounds])
    p999 = float(np.quantile(sc_steps, 0.999))
    thr = 1.5 * p999                       ## teleport threshold in scan space
    res['A_raw'] = dict(n_eps=len(bounds), ep_len_min=int(lens.min()), ep_len_max=int(lens.max()),
                        scan_step_p50=float(np.median(sc_steps)), scan_step_p999=p999,
                        scan_step_max=float(sc_steps.max()),
                        xy_step_p50=float(np.median(xy_steps)), xy_step_max=float(xy_steps.max()),
                        teleport_thr=thr)
    utils.print_color(f"[A raw] {len(bounds)} eps (len {lens.min()}-{lens.max()}) | "
                      f"scan step p50/p99.9/max = {np.median(sc_steps):.3f}/{p999:.3f}/{sc_steps.max():.3f} wu | "
                      f"xy step p50/max = {np.median(xy_steps):.3f}/{xy_steps.max():.3f} wu | "
                      f"teleport thr = {thr:.3f}",
                      c=('g' if xy_steps.max() < 0.5 else 'r'))

    ## ================= B. training buffer =================
    dset_cfg = load_config(args.planner_logdir, 'dataset_config.pkl')
    dset = dset_cfg()
    obs_f = dset.fields.observations            ## (P, L, 30) unnormed
    plens = np.asarray(dset.fields.path_lengths)
    P = obs_f.shape[0]
    bad_paths, n_bad_steps, rel_pos = 0, 0, []
    for p in range(P):
        st = steps(obs_f[p, :plens[p]])
        bad = np.where(st > thr)[0]
        if len(bad):
            bad_paths += 1; n_bad_steps += len(bad)
            rel_pos.extend((bad / max(plens[p] - 1, 1)).tolist())
    res['B_buffer'] = dict(n_paths=int(P), path_len_uniq=sorted(set(plens.tolist()))[:5],
                           bad_paths=int(bad_paths), bad_steps=int(n_bad_steps))
    utils.print_color(f"[B buffer] {P} paths (len {sorted(set(plens.tolist()))[:5]}) vs {len(bounds)} raw eps | "
                      f"paths with scan teleports: {bad_paths} ({100 * bad_paths / P:.1f}%), "
                      f"{n_bad_steps} teleport steps"
                      + (f", rel-positions p50 {np.median(rel_pos):.2f}" if rel_pos else ''),
                      c=('g' if bad_paths == 0 else 'r'))

    ## buffer paths <-> raw episodes alignment (match on last frame, front-padded buffers)
    tail_map = {}
    for i, (s, e) in enumerate(bounds):
        tail_map[lidar[e - 1].tobytes()] = i
    n_match = sum(1 for p in range(min(P, 200))
                  if obs_f[p, plens[p] - 1].astype(np.float32).tobytes() in tail_map)
    res['B_align'] = dict(checked=min(P, 200), matched=int(n_match))
    utils.print_color(f"[B align] {n_match}/{min(P, 200)} buffer paths end-frame-match a raw episode",
                      c=('g' if n_match == min(P, 200) else 'y'))

    ## sampled __getitem__ windows (normalized -> unnormalize -> step check)
    n_bad_win = 0
    for _ in range(args.n_items):
        b = dset[int(rng.integers(0, len(dset)))]
        w = dset.normalizer.unnormalize(np.asarray(b[0]), 'observations')
        if (steps(w) > thr).any():
            n_bad_win += 1
    res['B_getitem'] = dict(n=args.n_items, bad=int(n_bad_win))
    utils.print_color(f"[B getitem] {n_bad_win}/{args.n_items} sampled training windows contain a scan teleport",
                      c=('g' if n_bad_win == 0 else 'r'))

    ## ================= C. decode ground-truth windows =================
    if args.decode > 0:
        from diffuser.pl_eval.lidar_eval.lidar_grid_common import LG, LL
        lidar_cfg = resolve_lidar_config(logdir=args.planner_logdir,
                                         n_beams=args.n_beams, max_range=args.max_range)
        res['lidar_config'] = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                               for k, v in lidar_cfg.items()}
        grid = LG.load_or_build_grid(args.env, lidar_cfg, xy_step=args.xy_step, n_theta=1)
        loc = LL.GridLocalizer(grid, sigma=1.0)
        unit = grid.scanner.maze_unit
        H = args.horizon
        ok_eps = [i for i, (s, e) in enumerate(bounds) if e - s >= H]
        njs, errs = [], []
        for i in rng.choice(ok_eps, size=min(args.decode, len(ok_eps)), replace=False):
            s, e = bounds[i]
            st0 = int(rng.integers(s, e - H + 1))
            sl = slice(st0, st0 + H)
            idx = np.unique(np.concatenate([np.arange(0, H, 2), [H - 1]]))
            r = loc.localize_sequence(lidar[sl][idx], motion_sigma_cells=1.0, theta_blur=0)
            m = np.asarray(r['map_xy'])
            njs.append(int((np.linalg.norm(np.diff(m, axis=0), axis=1) > unit).sum()))
            errs.append(float(np.median(np.linalg.norm(m - xy[sl][idx], axis=1))))
        res['C_decode'] = dict(n=len(njs), jump_free=float(np.mean([n == 0 for n in njs])),
                               med_jumps=float(np.median(njs)), med_err_wu=float(np.median(errs)))
        utils.print_color(f"[C decode] {len(njs)} ground-truth windows: jump-free {100 * res['C_decode']['jump_free']:.0f}% | "
                          f"median jumps {np.median(njs):.1f} | median err {np.median(errs):.2f} wu",
                          c=('g' if res['C_decode']['jump_free'] > 0.9 else 'r'))

    ## ================= verdict =================
    data_clean = (bad_paths == 0 and n_bad_win == 0 and xy_steps.max() < 0.5)
    dec_clean = res.get('C_decode', {}).get('jump_free', 1.0) > 0.9
    if data_clean and dec_clean:
        v = ('DATA + DECODE CLEAN -> the model itself is the problem: it never learned '
             'globally-coherent scan sequences at this ckpt. Next: trainer recon heatmaps, '
             'lidar_planner_recon RMSE, medium-maze smoke train, check training actually ran to 2M.')
    elif not data_clean:
        v = ('TRAINING DATA DIRTY -> fix segmentation/padding in ogb_utils/ogb_lidar_utils, '
             'regenerate cache, retrain (medium smoke first). The model was trained on teleports.')
    else:
        v = 'DATA CLEAN, DECODE DIRTY on ground truth -> revisit localizer settings (sigma/motion sigma).'
    utils.print_color(f'[VERDICT] {v}', c='y')
    res['verdict'] = v
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)
    utils.print_color(f'[window check] wrote {args.out}', c='c')


if __name__ == '__main__':
    main()
