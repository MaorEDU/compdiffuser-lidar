#!/usr/bin/env python3
"""BALL likelihood-map census (BALL.md Phase 7 / design notes S2-S3).

For each goal (the eval-problem goals, plus optionally a dense sweep of every
free lattice point), compute the measurement-likelihood map of its WORLD-FRAME
goal scan over the (x, y) reference grid (n_theta=1) and answer, offline and
before any training completes:

  * how many identical peaks the likelihood map has (the project's "how many
    identical peaks exist in the space"),
  * whether the true goal is the argmax, and its top-K rank (the ceiling for
    check-and-move-on / verify-and-hop),
  * the decoy atlas: which cell pairs alias each other (and how strongly),
  * the per-goal posterior mass at the true cell (the calibration feature the
    failure-vs-uncertainty analysis buckets on).

Pure numpy + the repo's own grid/localization modules (torch-free import via
lidar_grid_common). h5py is needed only for --probs.

Usage
-----
  python -m diffuser.pl_eval.lidar_eval.ball_likelihood_census \
      --env pointmaze-giant-stitch-v0 \
      --probs data/ogb_maze/ev_probs/ogb_pointM_Gi_ev_prob_numEp20_eSdSt0.hdf5 \
      --sweep_all 1 --out logs/lidar_eval/ball_census_giant
"""
import os, sys, json, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
try:
    from diffuser.pl_eval.lidar_eval.lidar_grid_common import LG, LL, LS
    HAVE_PLOT = True
except Exception:
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    def _lm(name, rel):
        p = os.path.normpath(os.path.join(here, rel))
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m
    LS = _lm('lidar_sim', '../../datasets/lidar/lidar_sim.py')
    LG = _lm('lidar_grid', '../../datasets/lidar/lidar_grid.py')
    LL = _lm('lidar_localization', '../../datasets/lidar/lidar_localization.py')
    HAVE_PLOT = False


def cell_posterior(loc, grid, scanner, scan, sigma):
    """scan -> per-maze-cell posterior dict {(i, j): mass} + argmax lattice xy."""
    d = LL.scan_distance(scan, grid.free_scans)[0]
    like = LL.measurement_likelihood(d, sigma)
    b = like / max(like.sum(), 1e-30)
    fxy = np.asarray(grid.free_xy)
    ci, cj = scanner.world_to_cell(fxy[:, 0], fxy[:, 1])
    keys = (np.asarray(ci, dtype=np.int64) + 1000) * 100000 + (np.asarray(cj, dtype=np.int64) + 1000)
    uniq, inv = np.unique(keys, return_inverse=True)
    mass = np.zeros(len(uniq)); np.add.at(mass, inv, b)
    cells = [((int(k) // 100000) - 1000, (int(k) % 100000) - 1000) for k in uniq]
    k_best = int(np.argmax(b))
    return dict(zip(cells, mass.tolist())), tuple(fxy[k_best].tolist())


def count_peaks(cell_mass, maze_map, beta=0.5, nms_cells=1):
    """Local maxima of the per-cell mass, >= beta * global max, NMS by radius."""
    H, W = maze_map.shape
    M = np.zeros((H, W))
    for (i, j), m in cell_mass.items():
        if 0 <= i < H and 0 <= j < W:
            M[i, j] = m
    mx = M.max()
    cand = []
    for i in range(H):
        for j in range(W):
            if maze_map[i, j] == 1 or M[i, j] < beta * mx or M[i, j] <= 0:
                continue
            nb = M[max(i-1, 0):i+2, max(j-1, 0):j+2]
            if M[i, j] >= nb.max() - 1e-12:
                cand.append((float(M[i, j]), i, j))
    cand.sort(reverse=True)
    peaks = []
    for m, i, j in cand:                       # greedy NMS
        if all(max(abs(i - pi), abs(j - pj)) > nms_cells for _, pi, pj in peaks):
            peaks.append((m, i, j))
    return [(int(i), int(j), float(m)) for m, i, j in peaks], M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='pointmaze-giant-stitch-v0')
    ap.add_argument('--probs', default='', help='ev_probs hdf5 (goal_pos used)')
    ap.add_argument('--sweep_all', type=int, default=1,
                    help='1: also census EVERY free cell center (the decoy atlas)')
    ap.add_argument('--xy_step', type=float, default=0.5)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--n_beams', type=int, default=30)
    ap.add_argument('--max_range', type=float, default=12.0)
    ap.add_argument('--peak_beta', type=float, default=0.5)
    ap.add_argument('--nms_cells', type=int, default=1)
    ap.add_argument('--topk', type=int, default=6)
    ap.add_argument('--cache_root', default='data/ogb_maze/lidar_grid')
    ap.add_argument('--out', default='logs/lidar_eval/ball_census')
    ap.add_argument('--plots', type=int, default=0)
    args = ap.parse_args()

    lidar_cfg = dict(n_beams=args.n_beams, max_range=args.max_range,
                     fov=2 * np.pi, step=0.1)
    grid = LG.load_or_build_grid(args.env, lidar_cfg, xy_step=args.xy_step,
                                 n_theta=1, cache_root=args.cache_root)
    scanner = grid.scanner
    loc = LL.GridLocalizer(grid, sigma=args.sigma)
    mp = scanner.maze_map
    u = scanner.maze_unit
    print(f'[census] grid: {grid.n_poses} poses (xy_step={args.xy_step}, n_theta=1)')

    goals = []
    if args.probs:
        import h5py
        with h5py.File(args.probs, 'r') as f:
            gp = np.asarray(f['goal_pos'])
        goals += [('prob', k, float(g[0]), float(g[1])) for k, g in enumerate(gp[:, :2])]
        print(f'[census] {len(goals)} eval-problem goals from {args.probs}')
    if args.sweep_all:
        for (i, j) in np.argwhere(mp == 0):
            goals.append(('cell', -1, float(j * u - scanner.offset_x),
                          float(i * u - scanner.offset_y)))
        print(f'[census] + {int((mp == 0).sum())} free-cell-center goals (sweep)')
    assert goals, 'no goals: give --probs and/or --sweep_all 1'

    rows, decoy_pairs = [], {}
    for src, pid, gx, gy in goals:
        scan = scanner.scan(gx, gy, 0.0).astype(np.float32)
        cm, argmax_xy = cell_posterior(loc, grid, scanner, scan, args.sigma)
        peaks, M = count_peaks(cm, mp, beta=args.peak_beta, nms_cells=args.nms_cells)
        ti, tj = scanner.world_to_cell(np.asarray([gx]), np.asarray([gy]))
        true_cell = (int(ti[0]), int(tj[0]))
        ranked = sorted(cm.items(), key=lambda kv: -kv[1])
        rank = next((n for n, (c, _) in enumerate(ranked) if c == true_cell), -1)
        mass_true = float(cm.get(true_cell, 0.0))
        for (pi, pj, pm) in peaks:                       # decoy atlas edges
            if (pi, pj) != true_cell:
                key = tuple(sorted([true_cell, (pi, pj)]))
                decoy_pairs[key] = decoy_pairs.get(key, 0) + 1
        rows.append(dict(src=src, prob_id=int(pid), goal_xy=[gx, gy],
                         true_cell=list(true_cell), n_peaks=len(peaks),
                         peaks=[[i, j, round(m, 4)] for i, j, m in peaks[:8]],
                         true_rank=int(rank), mass_true=round(mass_true, 4),
                         argmax_is_true=bool(rank == 0)))

    R = [r for r in rows]
    n_peaks = np.asarray([r['n_peaks'] for r in R])
    ranks = np.asarray([r['true_rank'] for r in R])
    topk_curve = {k: float(((ranks >= 0) & (ranks < k)).mean())
                  for k in range(1, args.topk + 1)}
    summary = dict(
        env=args.env, n_goals=len(R),
        grid=dict(xy_step=args.xy_step, n_theta=1, sigma=args.sigma,
                  peak_beta=args.peak_beta, nms_cells=args.nms_cells),
        n_peaks_hist={str(k): int((n_peaks == k).sum()) for k in
                      range(0, int(n_peaks.max()) + 1)},
        frac_unimodal=float((n_peaks <= 1).mean()),
        argmax_is_true=float(np.mean([r['argmax_is_true'] for r in R])),
        true_in_topk=topk_curve,
        mass_true_median=float(np.median([r['mass_true'] for r in R])),
        worst_goals=[r for r in sorted(R, key=lambda r: r['mass_true'])[:10]],
        decoy_pairs_top=[[list(a), list(b), int(c)] for (a, b), c in
                         sorted(decoy_pairs.items(), key=lambda kv: -kv[1])[:20]],
    )
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(summary=summary, rows=rows),
              open(os.path.join(args.out, 'census.json'), 'w'), indent=1)
    print(f"[census] peaks per goal: "
          f"{ {k: v for k, v in summary['n_peaks_hist'].items()} }")
    print(f"[census] unimodal: {summary['frac_unimodal']*100:.1f}% | "
          f"argmax==true: {summary['argmax_is_true']*100:.1f}% | "
          f"true in top{args.topk}: {topk_curve[args.topk]*100:.1f}%")
    print(f"[census] wrote {os.path.join(args.out, 'census.json')}")

    if args.plots and HAVE_PLOT:
        from diffuser.pl_eval.lidar_eval import lidar_grid_common as GC
        import matplotlib.pyplot as plt
        worst = sorted(range(len(rows)), key=lambda k: rows[k]['mass_true'])[:6]
        for k in worst:
            gx, gy = rows[k]['goal_xy']
            scan = scanner.scan(gx, gy, 0.0).astype(np.float32)
            cm, _ = cell_posterior(loc, grid, scanner, scan, args.sigma)
            _, M = count_peaks(cm, mp, beta=0.0, nms_cells=args.nms_cells)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.imshow(M, origin='lower', extent=GC.maze_world_extent(scanner),
                      cmap='viridis')
            GC.draw_walls(ax, scanner)
            ax.plot([gx], [gy], 'r*', ms=14)
            ax.set_title(f"goal cell-posterior (mass@true={rows[k]['mass_true']:.3f}, "
                         f"peaks={rows[k]['n_peaks']})")
            fig.savefig(os.path.join(args.out, f'worst_{k}_map.png'), dpi=130,
                        bbox_inches='tight')
            plt.close(fig)
        print(f'[census] plots -> {args.out}/worst_*_map.png')


if __name__ == '__main__':
    main()
