#!/usr/bin/env python3
"""BALL ambiguity map + 45-degree target-scan sweep.

Two figures, both offline / CPU / numpy:

  A. AMBIGUITY MAP -- the maze with **each cell colored** by how ambiguous it is.
     The per-cell scalar is the project's PRE-REGISTERED aliasing statistic, lifted
     out of ``ball_beam_aliasing_census.py`` (which computes it per pose and then
     collapses it to a single ``confusable_frac``):

         alias_rms(i) = min per-beam RMS scan distance from pose i to any pose
                        at least ``--far_r`` wu away          (8 wu = 2 cells)
         tau          = median over poses of the median scan distance to the
                        0.5-wu lattice neighbours (the cost of one LEGITIMATE move)
         confusable   <=>  alias_rms < tau
           i.e. "teleporting >=2 cells is cheaper in scan space than moving 0.5 wu"

     Colours are centred on ``tau``, so below-the-break == confusable.

  B. 45-DEGREE SWEEP -- for each eval-block goal, the target's scan is held at
     yaw 0 (as the ball actually observes it) and every cell is re-scanned at
     0/45/.../315 degrees; each panel shows how close that rotated scan is to the
     target scan. Plus the best-orientation (min) map and the which-orientation
     (argmin) map.

     *** READ THIS BEFORE INTERPRETING PANEL B ***
     The ball's yaw is fixed at 0 by construction (the reference grid is
     n_theta=1, and _wrap_zero_yaw_synth re-renders everything at yaw 0), so the
     8 rotated scans are COUNTERFACTUAL: they measure rotational self-similarity
     of the MAZE, not anything the agent ever sees. 45 deg is not a multiple of
     the 12 deg beam spacing at n_beams=30, so the union of the 8 rotations spans
     240 distinct directions -- the min-over-theta map is effectively a 240-BEAM
     sensor and will look far less ambiguous than the real one. That improvement
     is the extra beams, not a perception result. See BALL_SESSIONS_LOG.md
     latest+15, where ant-style multi-yaw goal scans were already closed negative
     for the ball, and the "confusable_frac is not comparable across n_beams" trap.

Outputs (under --out):
    ambiguity_cells.png / ambiguity_lattice.png / ambiguity.json
    goal{b}_theta{deg}.png x 8, goal{b}_min_over_theta.png,
    goal{b}_argmin_theta.png, goal{b}_cells_8panel.png, scan45.json

Usage
-----
  python -m diffuser.pl_eval.lidar_eval.ball_ambiguity_map \
      --env pointmaze-giant-stitch-v0 --xy_step 0.5 --n_beams 30 --max_range 12 \
      --probs data/ogb_maze/ev_probs/ogb_pointM_Gi_ev_prob_numEp20_eSdSt0.hdf5 \
      --n_theta_scan 8 --far_r 8.0 \
      --out logs/lidar_eval/ball_ambiguity_giant
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_grid_common as C  # noqa: E402  (torch-free bootstrap, same dir)

LG, LL = C.LG, C.LL

CAVEAT = (
    'The ball yaw is fixed at 0 (reference grid n_theta=1), so the rotated scans are '
    'COUNTERFACTUAL: they measure rotational self-similarity of the maze, not anything the '
    'agent observes. 45 deg is not a multiple of the 12 deg beam spacing at n_beams=30, so the '
    'union of the 8 rotations spans 240 distinct directions -- min-over-theta is effectively a '
    '240-beam sensor, and its apparent gain is the extra beams, not perception. Multi-yaw goal '
    'scans were already closed negative for the ball (BALL_SESSIONS_LOG.md latest+15), and '
    'confusable_frac is not comparable across n_beams.'
)


# ---------------------------------------------------------------------------#
# A. pre-registered aliasing statistic (ball_beam_aliasing_census.py recipe)
# ---------------------------------------------------------------------------#
def pdist2(A):
    """Squared euclidean gram trick -- same helper the census uses."""
    sq = (A * A).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (A @ A.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def alias_stats(free_xy, free_scans, n_beams, far_r, neigh_r):
    """Per-pose alias_rms + the legitimate-motion scale tau.

    Verbatim recipe from ball_beam_aliasing_census.py (its `for nb, mr` loop):
    per-beam RMS scan distance, tau = median over poses of the median distance to
    0.5-wu lattice neighbours, alias = nearest far look-alike.
    """
    xy = np.asarray(free_xy, dtype=np.float64)
    S = np.asarray(free_scans, dtype=np.float64)
    M = len(S)
    d2_xy = pdist2(xy)
    rms = np.sqrt(pdist2(S)) / np.sqrt(n_beams)

    neigh = (d2_xy > 1e-9) & (d2_xy <= neigh_r ** 2)
    step_med = np.array([np.median(rms[i, neigh[i]]) for i in range(M)])
    tau = float(np.median(step_med))

    far = d2_xy >= far_r ** 2
    rms_far = np.where(far, rms, np.inf)
    alias = rms_far.min(1)                    # nearest far look-alike distance
    n_look = (rms_far < tau).sum(1)           # far look-alikes cheaper than a step
    return alias, tau, n_look


def to_cells(scanner, xy, values, reduce='min'):
    """Aggregate a per-pose vector onto maze cells -> (H, W) array, NaN elsewhere."""
    H, W = scanner.maze_map.shape
    ci, cj = scanner.world_to_cell(np.asarray(xy)[:, 0], np.asarray(xy)[:, 1])
    ci = np.asarray(ci, dtype=np.int64)
    cj = np.asarray(cj, dtype=np.int64)
    out = np.full((H, W), np.nan)
    ok = (ci >= 0) & (ci < H) & (cj >= 0) & (cj < W)
    fn = dict(min=np.min, median=np.median, mean=np.mean, max=np.max)[reduce]
    flat = ci * W + cj
    for key in np.unique(flat[ok]):
        sel = ok & (flat == key)
        out[key // W, key % W] = fn(np.asarray(values)[sel])
    return out


# ---------------------------------------------------------------------------#
# goals
# ---------------------------------------------------------------------------#
def load_block_goals(scanner, probs_path, block_size=20):
    """The 5 eval-block goals.

    ev_probs is 5 blocks of `block_size` CONSECUTIVE same-goal problems, but
    "same goal" is at CELL level, not exact xy: the 20 goals in a block are
    jittered inside one maze cell (this is the same invariant ball_dump_plans.py
    L432-438 checks, via `abs(gcells[p] - gcells[q]).max() <= 1`). We take the
    block CENTROID as the representative target and report the within-block
    spread so the reader can see how much that choice could matter.
    """
    import h5py
    with h5py.File(probs_path, 'r') as f:
        gp = np.asarray(f['goal_pos'])[:, :2].astype(np.float64)
    n_blocks = len(gp) // block_size
    goals = []
    for b in range(n_blocks):
        blk = gp[b * block_size:(b + 1) * block_size]
        ci, cj = scanner.world_to_cell(blk[:, 0], blk[:, 1])
        cells = np.stack([np.asarray(ci, dtype=np.int64),
                          np.asarray(cj, dtype=np.int64)], 1)
        uniq = np.unique(cells, axis=0)
        assert len(uniq) == 1, (
            f'ev_probs block {b} spans {len(uniq)} goal cells {uniq.tolist()}; this tool '
            'assumes one cell per block (5 blocks of 20). The problem set has changed.')
        gx, gy = float(blk[:, 0].mean()), float(blk[:, 1].mean())
        goals.append(dict(block=b, gx=gx, gy=gy,
                          cell=(int(uniq[0, 0]), int(uniq[0, 1])),
                          n_problems=int(len(blk)),
                          spread_wu=round(float(np.abs(blk - blk.mean(0)).max()), 3)))
    return goals


# ---------------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='pointmaze-giant-stitch-v0')
    ap.add_argument('--xy_step', type=float, default=0.5)
    ap.add_argument('--n_beams', type=int, default=30)
    ap.add_argument('--max_range', type=float, default=12.0)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--probs',
                    default='data/ogb_maze/ev_probs/ogb_pointM_Gi_ev_prob_numEp20_eSdSt0.hdf5')
    ap.add_argument('--block_size', type=int, default=20)
    ap.add_argument('--n_theta_scan', type=int, default=8,
                    help='orientations for panel B; 8 == every 45 deg')
    ap.add_argument('--far_r', type=float, default=8.0,
                    help='wu; pre-registered gate radius (8 wu = 2 cells)')
    ap.add_argument('--neigh_r', type=float, default=0.51,
                    help='wu; catches the 4-connected 0.5-wu lattice step')
    ap.add_argument('--cache_root', default='data/ogb_maze/lidar_grid')
    ap.add_argument('--out', default='logs/lidar_eval/ball_ambiguity')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lidar_cfg = dict(n_beams=args.n_beams, max_range=args.max_range,
                     fov=2 * np.pi, step=0.1)

    # ---------------------------------------------------------------- part A --#
    # Operational geometry: yaw == 0, n_theta=1. This is what the ball really sees.
    grid1 = LG.load_or_build_grid(args.env, lidar_cfg, xy_step=args.xy_step,
                                  n_theta=1, cache_root=args.cache_root)
    scanner = grid1.scanner
    H, W = scanner.maze_map.shape
    print(f'[amb] grid n_theta=1: {grid1.n_poses} poses '
          f'(xy_step={args.xy_step}), maze {H}x{W}', flush=True)

    alias, tau, n_look = alias_stats(grid1.free_xy, grid1.free_scans,
                                     args.n_beams, args.far_r, args.neigh_r)
    confusable_frac = float((alias < tau).mean())
    print(f'[amb] tau={tau:.4f} wu | confusable@{args.far_r:g}wu='
          f'{100 * confusable_frac:.1f}%  (census reference: 63.6% at 30/mr12/8wu)',
          flush=True)

    # Aggregating the pose-level gate onto cells. NOTE ON THE CHOICE:
    # `min` over a cell's ~50 lattice poses SATURATES -- it flags 85/86 free cells
    # (30 of them at exactly 0.0, i.e. exact far duplicates, which the beam-layout
    # census measured at 3.9% of poses for 30 beams). A near-uniform map cannot
    # answer "which cells are ambiguous", so the headline is the per-cell
    # CONFUSABLE FRACTION: the share of the cell's poses whose nearest far
    # look-alike is cheaper than one legal 0.5 wu move. Same pre-registered gate,
    # aggregated without saturating. min/median alias_rms are kept as side views.
    conf_frac = to_cells(scanner, grid1.free_xy, (alias < tau).astype(np.float64),
                         reduce='mean')
    amb_min = to_cells(scanner, grid1.free_xy, alias, reduce='min')
    amb_med = to_cells(scanner, grid1.free_xy, alias, reduce='median')
    look_med = to_cells(scanner, grid1.free_xy, n_look, reduce='median')

    png = C.plot_cell_heatmap(
        conf_frac, None, scanner, os.path.join(args.out, 'ambiguity_cells.png'),
        f'{args.env}\nAMBIGUITY PER CELL: fraction of the cell that is confusable\n'
        f'confusable = a look-alike >={args.far_r:g} wu away is closer in scan space\n'
        f'(alias_rms < tau={tau:.3f} wu) than one legal 0.5 wu move\n'
        f'bright = looks like somewhere else in the maze',
        cmap='inferno', label='confusable fraction of cell', annotate=True,
        clim=(0.0, 1.0), fmt='{:.2f}')
    C.plot_cell_heatmap(
        amb_med, None, scanner, os.path.join(args.out, 'ambiguity_cells_alias_rms.png'),
        f'{args.env}: ambiguity per cell -- alias_rms, median over the cell\n'
        f'blue = below tau={tau:.3f} wu = CONFUSABLE',
        cmap='RdBu', label=f'alias_rms (wu), tau={tau:.3f}',
        vcenter=tau, annotate=True)
    C.plot_cell_heatmap(
        amb_min, None, scanner, os.path.join(args.out, 'ambiguity_cells_worst_pose.png'),
        f'{args.env}: worst pose in each cell -- alias_rms, min over the cell\n'
        f'SATURATES (85/86 cells below tau={tau:.3f}); kept for completeness, '
        f'read ambiguity_cells.png instead',
        cmap='RdBu', label=f'alias_rms (wu), tau={tau:.3f}',
        vcenter=tau, annotate=True)
    C.plot_cell_heatmap(
        look_med, None, scanner, os.path.join(args.out, 'lookalike_count_cells.png'),
        f'{args.env}: far look-alikes cheaper than one legal step (median over cell)',
        cmap='magma', label='# look-alikes', annotate=True, fmt='{:.0f}')
    C.plot_xy_map(
        grid1, to_lattice(grid1, alias), os.path.join(args.out, 'ambiguity_lattice.png'),
        f'{args.env}: alias_rms at lattice resolution (xy_step={args.xy_step})',
        cmap='viridis', label='alias_rms (wu)')

    amb_json = dict(
        env=args.env, kind='pre-registered alias_rms (ball_beam_aliasing_census.py recipe)',
        config=dict(xy_step=args.xy_step, n_theta=1, n_beams=args.n_beams,
                    max_range=args.max_range, far_r=args.far_r, neigh_r=args.neigh_r),
        n_poses=int(len(alias)), tau_rms_wu=tau,
        confusable_frac=confusable_frac,
        census_reference_confusable_frac=0.636,
        alias_over_tau_p10=float(np.percentile(alias / tau, 10)),
        alias_over_tau_p50=float(np.percentile(alias / tau, 50)),
        n_lookalikes_p50=int(np.percentile(n_look, 50)),
        n_lookalikes_p90=int(np.percentile(n_look, 90)),
        headline='confusable_fraction_of_cell (min-over-cell saturates at 85/86 cells)',
        cells=[dict(cell=[int(i), int(j)],
                    confusable_fraction=round(float(conf_frac[i, j]), 4),
                    alias_rms_median=round(float(amb_med[i, j]), 4),
                    alias_rms_min=round(float(amb_min[i, j]), 4),
                    n_lookalikes_median=int(look_med[i, j]),
                    confusable_median=bool(amb_med[i, j] < tau))
               for i in range(H) for j in range(W) if np.isfinite(amb_min[i, j])],
    )
    with open(os.path.join(args.out, 'ambiguity.json'), 'w') as f:
        json.dump(amb_json, f, indent=1)
    print(f'[amb] wrote ambiguity.json ({len(amb_json["cells"])} free cells)', flush=True)

    # ---------------------------------------------------------------- part B --#
    nT = int(args.n_theta_scan)
    # load_or_build_grid -> build_grid_for_env, whose theta0 default is 0.0, so the
    # bins are exactly 0, 360/nT, ... degrees. Asserted rather than assumed.
    gridT = LG.load_or_build_grid(args.env, lidar_cfg, xy_step=args.xy_step,
                                  n_theta=nT, cache_root=args.cache_root)
    assert float(gridT.theta0) == 0.0, f'expected theta0=0, got {gridT.theta0}'
    assert np.allclose(np.degrees(gridT.theta_vals),
                       np.arange(nT) * (360.0 / nT)), gridT.theta_vals
    # the nT-theta grid must reduce to the operational one at theta=0
    ti0 = np.where(np.asarray(gridT.free_ti) == 0)[0]
    assert np.allclose(np.asarray(gridT.free_scans)[ti0], np.asarray(grid1.free_scans)), \
        'n_theta=%d grid theta-0 slice does not match the shipped n_theta=1 grid' % nT
    print(f'[amb] grid n_theta={nT}: {gridT.n_poses} poses; theta-0 slice matches '
          f'the operational grid', flush=True)

    loc = LL.GridLocalizer(gridT, sigma=args.sigma)
    degs = [round(float(np.degrees(t))) for t in np.asarray(gridT.theta_vals)]
    fxy = np.asarray(gridT.free_xy)
    fti = np.asarray(gridT.free_ti)

    goals = load_block_goals(scanner, args.probs, block_size=args.block_size)
    print(f'[amb] {len(goals)} eval-block goals from {args.probs}', flush=True)

    rows = []
    for g in goals:
        b, gx, gy, gcell = g['block'], g['gx'], g['gy'], g['cell']
        target_scan = scanner.scan(gx, gy, 0.0).astype(np.float32)   # yaw 0, FIXED
        mark = [dict(xy=(gx, gy), c='lime', m='*', s=260, label=f'goal blk{b}')]

        # per-orientation lattice maps, via the registered API
        lat = []
        for ti in range(nT):
            dmap, _ = loc.distance_map_to_target(target_scan, reduce=ti)
            lat.append(dmap)
            C.plot_xy_map(
                gridT, dmap, os.path.join(args.out, f'goal{b}_theta{degs[ti]:03d}.png'),
                f'blk{b} goal ({gx:.1f}, {gy:.1f}): scan distance to target, '
                f'cells rotated {degs[ti]}deg (target held at yaw 0)\n'
                f'dark = looks like the target -- COUNTERFACTUAL, see scan45.json caveat',
                cmap='viridis', label='scan distance (wu)', marks=mark)
        lat = np.stack(lat, 0)                                  # (nT, n_row, n_col)

        dmin, info = loc.distance_map_to_target(target_scan, reduce='min')
        # argmin over theta, NaN at lattice points that are wall at every theta
        # (plain nanargmin raises on an all-NaN column)
        valid = np.isfinite(lat).any(0)
        amin = np.full(lat.shape[1:], np.nan)
        amin[valid] = np.argmin(np.where(np.isfinite(lat), lat, np.inf), 0)[valid]
        C.plot_xy_map(
            gridT, dmin, os.path.join(args.out, f'goal{b}_min_over_theta.png'),
            f'blk{b} goal ({gx:.1f}, {gy:.1f}): BEST-orientation distance to target\n'
            f'union of {nT} rotations = {nT * args.n_beams} directions -- reads as a '
            f'{nT * args.n_beams}-beam sensor, NOT a perception result',
            cmap='viridis', label='min scan distance (wu)', marks=mark)
        C.plot_xy_map(
            gridT, amin, os.path.join(args.out, f'goal{b}_argmin_theta.png'),
            f'blk{b} goal ({gx:.1f}, {gy:.1f}): WHICH rotation matches the target best',
            cmap='twilight', label='best rotation (index)', marks=mark)

        # per-cell tensor, straight from the pose-level distances
        d_pose = LL.scan_distance(target_scan, gridT.free_scans)[0]
        cell_t = np.stack([to_cells(scanner, fxy[fti == ti], d_pose[fti == ti], 'min')
                           for ti in range(nT)], 0)             # (nT, H, W)

        # the summary figure: 8 per-cell panels in one image
        import matplotlib.pyplot as plt
        ncol = min(4, nT)
        nrow = int(np.ceil(nT / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 0.40 * W, nrow * 0.44 * H + 1.6))
        vmax = float(np.nanmax(cell_t))
        im = None
        for ti, ax in enumerate(np.atleast_1d(axes).ravel()[:nT]):
            im = C.plot_cell_heatmap(
                cell_t[ti], None, scanner, None, f'{degs[ti]} deg', cmap='viridis',
                label='scan dist (wu)', ax=ax, marks=(mark if ti == 0 else
                                                     [dict(xy=(gx, gy), c='lime', m='*', s=260)]),
                clim=(0.0, vmax), cbar=False)
            ax.set_xlabel(''); ax.set_ylabel('')
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            f'blk{b} goal ({gx:.1f}, {gy:.1f}) cell {gcell}: how identical is each cell to the '
            f'target scan, scanned every {360 // nT} deg?   dark = identical\n'
            f'Target held at yaw 0. COUNTERFACTUAL: the ball yaw is ALWAYS 0, and the union of '
            f'these {nT} rotations is a {nT * args.n_beams}-beam sensor -- see scan45.json caveat.',
            fontsize=11)
        fig.tight_layout(rect=(0, 0.02, 0.93, 0.94))
        cax = fig.add_axes([0.945, 0.10, 0.012, 0.78])
        fig.colorbar(im, cax=cax, label='scan distance to target (wu)')
        fig.savefig(os.path.join(args.out, f'goal{b}_cells_8panel.png'), dpi=120)
        plt.close(fig)

        # Self-match sanity. The target is the block centroid, not a lattice point,
        # so its own cell's nearest lattice pose sits up to xy_step/2 away and the
        # distance is small-but-nonzero. It must still be (a) far below tau and
        # (b) the global minimum of the 0-deg map.
        self_vec = [round(float(cell_t[ti][gcell]), 4) for ti in range(nT)]
        assert self_vec[0] < 0.5 * tau, (
            f'blk{b}: own cell at 0 deg is {self_vec[0]:.3f} wu from the target, '
            f'which is not << tau={tau:.3f}')
        assert np.nanargmin(cell_t[0]) == gcell[0] * W + gcell[1], (
            f'blk{b}: the 0-deg map does not bottom out at the goal cell {gcell}')

        far_below_tau = []
        for ti in range(nT):
            m = cell_t[ti]
            cnt = 0
            for i in range(H):
                for j in range(W):
                    if not np.isfinite(m[i, j]):
                        continue
                    if max(abs(i - gcell[0]), abs(j - gcell[1])) * scanner.maze_unit < args.far_r:
                        continue
                    cnt += int(m[i, j] < tau)
            far_below_tau.append(cnt)

        rows.append(dict(
            block=b, goal_xy=[round(gx, 4), round(gy, 4)], goal_cell=list(gcell),
            n_problems_in_block=g['n_problems'],
            target='centroid of the block\'s goals (they are jittered within one cell)',
            within_block_spread_wu=g['spread_wu'],
            theta_deg=degs,
            self_cell_distance_by_theta=self_vec,
            far_cells_below_tau_by_theta=far_below_tau,
            min_over_theta_at_own_cell=round(float(np.nanmin(cell_t[:, gcell[0], gcell[1]])), 4),
            nearest_lookalike_xy=[round(float(v), 3) for v in info['nearest_xy']],
            nearest_lookalike_distance=round(float(info['nearest_distance']), 4),
            cells=[dict(cell=[int(i), int(j)],
                        dist_by_theta=[round(float(cell_t[ti][i, j]), 4) for ti in range(nT)])
                   for i in range(H) for j in range(W) if np.isfinite(cell_t[0][i, j])],
        ))
        print(f'[amb] blk{b} goal ({gx:.1f}, {gy:.1f}) cell {gcell}: '
              f'self-vec {self_vec} | far cells below tau per theta {far_below_tau}',
              flush=True)

    with open(os.path.join(args.out, 'scan45.json'), 'w') as f:
        json.dump(dict(env=args.env, caveat=CAVEAT,
                       config=dict(xy_step=args.xy_step, n_theta=nT, theta0=0.0,
                                   theta_step_deg=360.0 / nT, n_beams=args.n_beams,
                                   max_range=args.max_range, sigma=args.sigma,
                                   target_yaw=0.0, far_r=args.far_r),
                       tau_rms_wu=tau, blocks=rows), f, indent=1)
    print(f'[amb] wrote scan45.json ({len(rows)} blocks) -> {args.out}', flush=True)


def to_lattice(grid, per_pose):
    """(M,) per-pose values -> (n_row, n_col) lattice map (n_theta=1 grids)."""
    assert grid.n_theta == 1, 'lattice view expects the n_theta=1 grid'
    out = np.full((grid.n_row, grid.n_col), np.nan)
    rc = np.asarray(grid.free_rc)
    out[rc[:, 0], rc[:, 1]] = np.asarray(per_pose)
    return out


if __name__ == '__main__':
    main()
