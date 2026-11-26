#!/usr/bin/env python3
"""Sample N plan resamples per eval problem and dump the plans .npz that
``ball_traj_consistency.py`` (BALL.md Phase 8) consumes. Optionally also runs
the G2 offline-decode gate (BALL.md §7.3) on the sampled plans — the check the
ladder runs skipped.

Needs GPU + the trained planner run dir + the full LiDAR npz (Phase 1,
``gen_lidar_dataset.py``). Run on the cluster:

  python -m diffuser.pl_eval.lidar_eval.ball_dump_plans \
      --planner_logdir logs/pointmaze-giant-stitch-v0/diffusion/og_pntM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512 \
      --npz data/ogb_maze/lidar/pointmaze-giant-stitch-v0_lidar.npz \
      --epoch 1200000 --g2 \
      --out logs/lidar_eval/ball_plans_lidar.npz

  # optional second dump from the upstream (x,y) o2d planner for the cross-model half:
  #   --o2d_logdir logs/pointmaze-giant-stitch-v0/diffusion/og_pntM_Gi_o2d_... \
  #   --o2d_out logs/lidar_eval/ball_plans_o2d.npz

Then:

  python -m diffuser.pl_eval.lidar_eval.ball_traj_consistency \
      --env pointmaze-giant-stitch-v0 \
      --plans_lidar logs/lidar_eval/ball_plans_lidar.npz \
      --plans_xy    logs/lidar_eval/ball_plans_o2d.npz \
      --out logs/lidar_eval/ball_consistency.json

Output npz format (what ball_traj_consistency.load_plans expects):
  plans       (P, N, H, D) float32   D=30 scans (lidar) or 2 (o2d)
  problem_ids (P,) int

Sampling matches the bench eval path: EMA weights, cond_w=2, DDIM 50 (eta 1),
interleave composition, exp/beta-2 blending; ``ev_top_n = N`` so ALL N blended
candidates are kept instead of the top-5 cut (we want the plan distribution,
not the best-of ranking).
"""
import os, sys, json, argparse, copy
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
import diffuser.utils as utils
from diffuser.pl_eval.lidar_eval.lidar_planner_recon import (
    load_planner_ema, make_lidar_normalizer)
from diffuser.models.cd_stgl_sml_dfu.stgl_sml_policy_v1 import Stgl_Sml_Policy_V1
from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env
from diffuser.utils.cp_utils.cp_serial import load_stgl_lh_ev_probs_hdf5
## ONE shared sensing-geometry resolver for every CLI on the measurement path
## (see lidar_grid_common.resolve_lidar_config for the 2026-07-29 incident).
from diffuser.pl_eval.lidar_eval.lidar_grid_common import (
    DEFAULT_LIDAR_CONFIG, resolve_lidar_config, assert_npz_matches,
    json_safe_lidar_config as _json_safe, maze_bfs_wu,
    maze_shortest_path_cells, make_waypoints)


def ep_bounds(terminals):
    ends = np.where(np.asarray(terminals).reshape(-1).astype(bool))[0]
    out, prev = [], 0
    for e in ends:
        out.append((prev, e + 1)); prev = e + 1
    if prev < len(terminals):
        out.append((prev, len(terminals)))
    return out


def make_bridge_problems(d, lidar, xy, n_probs, gap, seed):
    """IN-DISTRIBUTION bridge problems — the model's own TRAINING task.

    ``get_conditions`` (ogb_dataset_v2.py L246) conditions each training window on
    ``observations[0]`` and ``observations[horizon-1]`` of the SAME window. So a
    problem built as (scan at frame t, scan at frame t+gap) with gap = tot_hzn-1
    from a real episode is *literally* what the model was trained to bridge:
    the goal is reachable in exactly the horizon, the pair is on-manifold, and a
    ground-truth answer exists.

    Every G2 number so far was measured on ev_probs instead — start/goal pairs
    ~35 wu apart scored against an 888-frame composed plan. That confounds three
    things at once (goal-conditioning, composition over 8 segments, and the goal
    constraint propagating back through 7 overlaps). This isolates the first.

    Conditioning scans are taken STRAIGHT FROM THE NPZ, not re-scanned, so they
    are bit-identical to the tensors training saw.
    """
    rng = np.random.default_rng(seed)
    bounds = [(s, e) for s, e in ep_bounds(d['terminals']) if e - s > gap]
    assert len(bounds) >= n_probs, \
        f'only {len(bounds)} episodes are longer than gap={gap}; lower --n_comp or --bridge_gap'
    pick = rng.choice(len(bounds), size=n_probs, replace=False)
    t0 = [int(rng.integers(bounds[i][0], bounds[i][1] - gap)) for i in pick]
    starts = np.stack([xy[t] for t in t0]).astype(np.float64)
    goals = np.stack([xy[t + gap] for t in t0]).astype(np.float64)
    st_sc = np.stack([lidar[t] for t in t0]).astype(np.float32)
    gl_sc = np.stack([lidar[t + gap] for t in t0]).astype(np.float32)
    gt_xy = np.stack([xy[t:t + gap + 1] for t in t0]).astype(np.float32)
    sep = np.linalg.norm(goals - starts, axis=1)
    utils.print_color(
        f'[bridge] {n_probs} in-distribution problems, gap={gap} frames '
        f'(= the training horizon): straight-line start-goal separation '
        f'p10/p50/p90 = {np.quantile(sep, 0.1):.1f}/{np.median(sep):.1f}/'
        f'{np.quantile(sep, 0.9):.1f} wu. Goals are reachable BY CONSTRUCTION '
        f'(a real trajectory does it in exactly {gap} steps).', c='c')
    return starts, goals, st_sc, gl_sc, gt_xy


def measure_speed(d, xy, gap, n_sample=2000, seed=0):
    """Median data arc-length PER FRAME, measured on real ``gap``-frame windows.

    The pace calibration for ``make_spatial_bridge_problems``: a composed plan of
    length ``L`` frames represents ``speed * L`` world units of travel, so a
    pace-matched goal must sit that far away ALONG THE MAZE, not in a straight
    line (a straight-line target makes the plan detour and is not the same task).
    """
    rng = np.random.default_rng(seed)
    bounds = [(s, e) for s, e in ep_bounds(d['terminals']) if e - s > gap]
    assert bounds, f'no episode longer than gap={gap}'
    arcs = []
    for i in rng.choice(len(bounds), size=min(n_sample, len(bounds)), replace=False):
        s, e = bounds[i]
        t = int(rng.integers(s, e - gap))
        seg = xy[t:t + gap + 1]
        arcs.append(float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()))
    return float(np.median(arcs)) / gap, float(np.median(arcs))


def measure_progress_speed(d, xy, scanner, gap, n_sample=600, seed=0):
    """Median BFS-MAZE-PATH progress per frame on real ``gap``-frame windows.

    THE UNIT THAT ``make_spatial_bridge_problems`` ACTUALLY NEEDS, and the one it
    got wrong until 2026-07-30. That function places the goal at a target *maze
    BFS distance*, but calibrated the target with ``measure_speed`` -- which is
    ARC LENGTH per frame. Those are not the same quantity and are not close: the
    data is a *stitch* dataset whose demonstrations wander, so a real 159-frame
    window walks 30.5 wu of arc while advancing only 12.0 wu along the maze
    (dawdle factor arc/BFS = 2.53). Calibrating a BFS target with an arc speed
    therefore asks for a goal ~2.5x farther than the data ever reaches in the
    frames provided, and every "pace-matched" spatial problem ever run was in
    fact 2.4-2.65x UNDER-PROVISIONED. See the 2026-07-30 session entry.

    Returns ``(progress_speed, arc_speed, med_path, med_arc, med_tort)`` where
    ``tort`` is straight-line separation / BFS path -- the tortuosity measure the
    spatial arm must match (real windows: 0.75). Note the earlier "spatial
    problems are straighter, 0.62 vs 0.33" confound compared sep/ARC against
    sep/BFS; on one consistent measure the gap is 0.61 vs 0.75, i.e. secondary.
    """
    rng = np.random.default_rng(seed)
    bounds = [(s, e) for s, e in ep_bounds(d['terminals']) if e - s > gap]
    assert bounds, f'no episode longer than gap={gap}'
    ci, cj = scanner.world_to_cell(xy[:, 0], xy[:, 1])
    ci, cj = np.asarray(ci, int), np.asarray(cj, int)
    cache = {}

    def bfs_from(i, j):
        k = (int(i), int(j))
        if k not in cache:
            cache[k] = maze_bfs_wu(scanner, k)
        return cache[k]

    arcs, paths, torts = [], [], []
    for i in rng.choice(len(bounds), size=min(n_sample, len(bounds)), replace=False):
        s, e = bounds[i]
        t = int(rng.integers(s, e - gap))
        seg = xy[t:t + gap + 1]
        p = float(bfs_from(ci[t], cj[t])[ci[t + gap], cj[t + gap]])
        if not np.isfinite(p) or p <= 0:
            continue
        arcs.append(float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()))
        paths.append(p)
        torts.append(float(np.linalg.norm(xy[t + gap] - xy[t]) / p))
    assert paths, 'no window with a finite BFS path; is the maze map right?'
    med_path, med_arc = float(np.median(paths)), float(np.median(arcs))
    return (med_path / gap, med_arc / gap, med_path, med_arc, float(np.median(torts)))


## maze_bfs_wu now lives in lidar_grid_common (imported above) so the anatomy
## tool can use it without pulling in torch; re-exported here for callers.


def make_spatial_bridge_problems(d, lidar, xy, scanner, n_probs, gap, seed,
                                 tol=0.2, max_tries=400, pace='progress',
                                 tort=None, tort_tol=0.15):
    """PACE-MATCHED composed-horizon problems, at the cost of the ground truth.

    Why this exists: every episode in the LiDAR npz is exactly 200 frames, so
    ``make_bridge_problems`` caps at gap=159 and there is NO ground-truth route
    for the 263/887-frame horizons that --n_comp 2/8 produce. (That is intrinsic
    to a *stitch* dataset -- the benchmark deliberately contains no trajectory
    long enough to solve its own tasks.) So a composition-depth ladder must
    either keep the solved gap=159 problems and let the plan dawdle (Arm A,
    ``--bridge_gap 159``), or drop the GT and match the PACE (this function).

    Construction: draw a start frame, BFS the maze for cells whose path distance
    from it is ``speed * gap`` (+-tol), and take the goal from a real data frame
    in one of those cells -- so both conditioning scans are real npz rows, the
    goal is at a maze distance the horizon can actually cover, and the goals
    spread over many distinct cells (a valid permutation null).

    Reachability is statistical, not by construction: a route exists (the maze is
    connected and BFS found the cell) but no single demonstration performs it.
    Metrics: G2a/G2b/null only -- there is no gt_xy, so no deviation/coverage.
    """
    rng = np.random.default_rng(seed)
    ## PACE. 'progress' calibrates the BFS target with BFS-per-frame (the same
    ## unit as the target). 'arc' reproduces the pre-2026-07-30 behaviour, which
    ## calibrated a BFS target with an ARC speed and so over-shot by the data's
    ## dawdle factor (2.53x here) -- kept only to reproduce the old ladder.
    prog_speed, arc_speed, med_path, arc159, data_tort = measure_progress_speed(
        d, xy, scanner, min(gap, 159), seed=seed)
    speed = prog_speed if pace == 'progress' else arc_speed
    target = speed * gap
    n_frames = len(xy)
    if pace == 'arc':
        utils.print_color(
            f'[bridge/spatial] PACE MODE "arc" -- this is the known-WRONG calibration '
            f'(2026-07-30): it sets a maze-BFS target using an arc-length speed, so it asks '
            f'for a goal {arc_speed / prog_speed:.2f}x farther than the data reaches in {gap} '
            f'frames. Use --bridge_pace progress unless you are reproducing the old ladder.',
            c='r')

    ## The maze has a finite BFS diameter (giant: 124 wu). A composed horizon can
    ## demand MORE travel than the maze contains -- n_comp=8 asks for
    ## 887*0.192 = 170 wu -- in which case no pace-matched problem exists at all
    ## and the horizon is over-provisioned BY CONSTRUCTION. Clamp and say so
    ## loudly rather than failing the whole ladder.
    diam = 0.0
    mp = np.asarray(scanner.maze_map)
    for i, j in np.argwhere(mp != 1):
        dd = maze_bfs_wu(scanner, (i, j))
        f = dd[np.isfinite(dd)]
        if len(f):
            diam = max(diam, float(f.max()))
    over = 1.0
    if target > diam * (1 - tol):
        over = target / max(diam, 1e-9)
        utils.print_color(
            f'[bridge/spatial] HORIZON EXCEEDS THE MAZE: a {gap + 1}-frame plan at data '
            f'speed {speed:.4f} wu/frame represents {target:.0f} wu of travel, but the maze '
            f'BFS DIAMETER is only {diam:.0f} wu ({over:.2f}x). No pace-matched problem '
            f'exists at this depth -- the horizon is over-provisioned BY CONSTRUCTION and the '
            f'model MUST loiter. Clamping the target to the longest available maze path; '
            f'this arm is "longest-available", NOT pace-matched. Report it as such.', c='r')
        target = diam * (1 - tol)
    ## cell index for every data frame, once
    ci, cj = scanner.world_to_cell(xy[:, 0], xy[:, 1])
    ci, cj = np.asarray(ci, int), np.asarray(cj, int)
    cell_key = ci * 10000 + cj
    order = np.argsort(cell_key, kind='stable')
    keys_sorted = cell_key[order]

    starts, goals, st_sc, gl_sc, seps, paths = [], [], [], [], [], []
    tries = 0
    while len(starts) < n_probs and tries < max_tries * n_probs:
        tries += 1
        t0 = int(rng.integers(0, n_frames))
        dist = maze_bfs_wu(scanner, (ci[t0], cj[t0]))
        cand = np.argwhere(np.abs(dist - target) <= tol * target)
        if not len(cand):
            continue
        rng.shuffle(cand)
        picked = None
        for a, b in cand:
            k = int(a) * 10000 + int(b)
            lo = int(np.searchsorted(keys_sorted, k, 'left'))
            hi = int(np.searchsorted(keys_sorted, k, 'right'))
            if hi <= lo:
                continue
            t1c = int(order[rng.integers(lo, hi)])
            ## Match the DATA's tortuosity (straight-line separation / BFS path).
            ## Without this the spatial arm drifts straighter than training windows
            ## (0.47-0.61 vs the data's 0.75), which is the second confound stated
            ## on 2026-07-29 -- smaller than the pace error but still uncontrolled.
            if tort is not None:
                pd = float(dist[a, b])
                if pd <= 0:
                    continue
                if abs(float(np.linalg.norm(xy[t1c] - xy[t0])) / pd - tort) > tort_tol:
                    continue
            picked = (t1c, float(dist[a, b]))
            break
        if picked is None:
            continue
        t1, pathd = picked
        starts.append(xy[t0]); goals.append(xy[t1])
        st_sc.append(lidar[t0]); gl_sc.append(lidar[t1])
        seps.append(float(np.linalg.norm(xy[t1] - xy[t0]))); paths.append(pathd)
    assert len(starts) == n_probs, (
        f'only found {len(starts)}/{n_probs} spatial bridge problems at maze distance '
        f'{target:.1f}+-{tol*100:.0f}% wu'
        + (f' and tortuosity {tort:.2f}+-{tort_tol:.2f}' if tort is not None else '')
        + '; raise --bridge_tol/--bridge_tort_tol or lower --n_comp')
    starts = np.stack(starts).astype(np.float64)
    goals = np.stack(goals).astype(np.float64)
    seps, paths = np.asarray(seps), np.asarray(paths)
    utils.print_color(
        f'[bridge/spatial] {n_probs} '
        f'{"LONGEST-AVAILABLE (not pace-matched)" if over > 1.0 else "PACE-MATCHED"} '
        f'problems for a {gap + 1}-frame horizon. '
        f'Calibration [pace={pace}]: real {min(gap, 159)}-frame windows walk {arc159:.1f} wu of '
        f'ARC but advance only {med_path:.1f} wu along the MAZE (dawdle {arc_speed/prog_speed:.2f}x), '
        f'so progress speed = {prog_speed:.4f} wu/frame (arc speed {arc_speed:.4f}); using '
        f'{speed:.4f} -> target maze path {target:.1f} wu (+-{tol*100:.0f}%). '
        f'Achieved maze path p10/p50/p90 = {np.quantile(paths, 0.1):.1f}/{np.median(paths):.1f}/'
        f'{np.quantile(paths, 0.9):.1f} wu; straight-line separation p10/p50/p90 = '
        f'{np.quantile(seps, 0.1):.1f}/{np.median(seps):.1f}/{np.quantile(seps, 0.9):.1f} wu. '
        f'tortuosity sep/BFS p50 = {np.median(seps / np.maximum(paths, 1e-9)):.2f} '
        f'(data {data_tort:.2f}; '
        + ('NOT CONSTRAINED' if tort is None else
           ('matched' if abs(np.median(seps / np.maximum(paths, 1e-9)) - data_tort) <= tort_tol
            else f'REQUESTED {tort:.2f} BUT DRIFTED -- too few pairs at this maze distance'))
        + '). '
        f'xDataPace = {(np.median(paths) / gap) / prog_speed:.2f}x '
        f'(1.00x = the horizon is exactly as long as the data needs; >1 under-provisioned, '
        f'<1 the plan must dawdle). '
        f'NO ground-truth route exists at this horizon -- reachability is statistical '
        f'(BFS-connected), not by construction; deviation/coverage are unavailable.'
        + (f' PACE MISMATCH {over:.2f}x (clamped to the maze diameter).' if over > 1.0 else ''),
        c='y')
    return (starts, goals, np.stack(st_sc).astype(np.float32),
            np.stack(gl_sc).astype(np.float32), None)


def make_map_projector(grid):
    """(B,T,D) scans -> the nearest realizable scan in ``grid.free_scans``.

    Pure perception: the reference scans are rendered from the KNOWN maze map,
    exactly as the goal scan already is. The model is never told where it is.
    """
    from diffuser.pl_eval.lidar_eval.lidar_grid_common import LL
    ref = np.asarray(grid.free_scans, dtype=np.float32)          # (M, n_beams)

    def project(x):
        a = np.asarray(x, dtype=np.float32)
        flat = a.reshape(-1, a.shape[-1])
        if flat.shape[-1] != ref.shape[-1]:                      # o2d: not applicable
            return a
        out = np.empty_like(flat)
        for s in range(0, len(flat), 256):                       # bound peak memory
            e = min(s + 256, len(flat))
            d = LL.scan_distance(flat[s:e], ref, chunk=1024)      # (b, M)
            out[s:e] = ref[np.argmin(d, axis=1)]
        return out.reshape(a.shape)
    return project


def build_policy(logdir, epoch, device, cond_w, ddim_steps, ddim_eta,
                 n_comp, n_resamples, norm, blend_type='exp', exp_beta=2,
                 projector=None):
    model, ep = load_planner_ema(logdir, epoch, device)
    model.condition_guidance_w = cond_w
    ## same eval sampler as the bench (jsons: use_ddim=True, eta=1, steps=50)
    model.use_ddim = ddim_steps > 0
    if ddim_steps > 0:
        model.ddim_eta = ddim_eta
        model.ddim_num_inference_steps = ddim_steps
    pol_config = dict(ev_n_comp=n_comp, ev_top_n=n_resamples,
                      ev_pick_type='first', ev_goal_pick_tail=8,
                      ev_cp_infer_t_type='interleave')
    tj_blder_config = dict(blend_type=blend_type, exp_beta=exp_beta)
    pol = Stgl_Sml_Policy_V1(model, norm, pol_config, tj_blder_config)
    if projector is not None:
        pol.tj_blder.projector = projector
    if blend_type != 'exp':
        utils.print_color(
            f'[dump] BLEND MODE = {blend_type}'
            + (' (+ map projection)' if (blend_type == 'switch_map' and projector) else '')
            + '. Default is exp/beta-2, a per-beam convex AVERAGE of two segments\' '
              'scans over each 56-frame overlap -- invalid for a range-vector '
              'representation (the average of two poses\' scans is no pose\'s scan; '
              'measured 1.91x less map-realizable in overlap phase, and the decode '
              'derails at the first seam). See traj_blender.blend_2_np_trajs_23d.', c='y')
    utils.print_color(
        f'[dump] {os.path.basename(logdir.rstrip(os.sep))} @ epoch {ep}: '
        f'obs_dim={model.observation_dim} hzn={model.horizon} '
        f'len_ovlp={model.len_ovlp_cd} n_comp={n_comp} -> '
        f'tot_hzn={n_comp * model.horizon - (n_comp - 1) * model.len_ovlp_cd}', c='c')
    return pol, model, int(ep)


## maze_shortest_path_cells / make_waypoints live in lidar_grid_common
## (imported above) so the rollout planner can build waypoints without
## importing this eval tool; re-exported here for the tests and callers.


def sample_model(pol, st_cond, gl_cond, n_resamples, tag, goal_band_k=1, waypoints=None):
    """st_cond/gl_cond: (P, D) conditioning rows (scans or xy). -> (P,N,H,D)

    goal_band_k > 1: tile the goal row into the (1+K, 1, D) st_gl band format
    (policy maps rows 1..K onto frames H-K..H-1 — a stationary-goal band; the
    K is auto-read from the loaded ckpt's diff_config in main())."""
    K = int(max(goal_band_k, 1))
    plans = []
    for p in range(len(st_cond)):
        rows = [st_cond[p][None, :]] + [gl_cond[p][None, :]] * K
        g_cond = {'st_gl': np.stack(rows).astype(np.float32)}   # (1+K, 1, D)
        if waypoints is not None:
            g_cond['waypoints'] = waypoints[:, p:p + 1, :]       # (n_comp-1, 1, D)
        with torch.no_grad():
            m_out = pol.gen_cond_stgl(g_cond=g_cond, b_s=n_resamples)
        cand = np.asarray(m_out.trajs_list_topn_bl, dtype=np.float32)  # (N,H,D)
        plans.append(cand)
        utils.print_color(f'[dump/{tag}] problem {p + 1}/{len(st_cond)}: '
                          f'{cand.shape[0]} plans x {cand.shape[1]} frames', c='c')
    return np.stack(plans)                                       # (P,N,H,D)


def g2_check(plans, starts, goals, env, xy_step, n_dec, out_json, lidar_cfg, gt_xy=None):
    """G2 offline decode (BALL.md §7.3) on n_dec plans/problem:
    G2a validity (multi-cell jumps in the decoded route, seam-aligned or not)
    G2b goal anchoring (decoded endpoint lands in the true goal cell +-1).

    Also accepts D=2 (o2d) plans: decode is then the identity, which makes the
    o2d dump the harness CONTROL — same composition machinery, same metrics,
    but a planner whose representation carries spatial continuity. Expected
    o2d: G2a high (continuous bodies reaching the inpainted goal) — if o2d
    fails G2a too, the sampler/composition harness is at fault, not the model."""
    from diffuser.pl_eval.lidar_eval.lidar_grid_common import LG, LL
    grid = LG.load_or_build_grid(env, lidar_cfg, xy_step=xy_step, n_theta=1)
    scanner, loc = grid.scanner, LL.GridLocalizer(grid, sigma=1.0)
    P, N, H, D = plans.shape

    ## ---- is this problem set able to measure goal-conditioning AT ALL? ----
    ## ogb_pointM_Gi_ev_prob_*.hdf5 is 5 tasks x 20 problems: problems 0-19 share
    ## ONE goal region (52.0, 35.7) +-1 wu, 20-39 another, etc. Every G2 run in
    ## this project used --n_probs 20 = block 0 = a SINGLE start-goal pair. On such
    ## a set G2b still detects failure (a goal-reaching model would score ~100%),
    ## but the permutation null is vacuous -- shuffling goals returns the same cell.
    gcells = np.stack([np.array(scanner.world_to_cell(goals[p:p + 1, 0],
                                                      goals[p:p + 1, 1])).ravel()
                       for p in range(P)])
    n_goal_cells = len(np.unique(gcells, axis=0))
    ## The null pairs each problem's ENDPOINT with OTHER problems' goals. Use the
    ## exact expectation over ALL off-diagonal pairs, not the neighbour q=(p+1)%P:
    ## ev_probs is ordered as 5 blocks of 20 CONSECUTIVE same-goal problems, so the
    ## neighbour shares the true goal 19 times out of 20 and the "null" came out at
    ## 95% -- an artifact of the problem ORDERING, not of the goal diversity.
    ## (2026-07-29: this is why --n_probs 100 tripped the degeneracy guard even
    ## though it spans all 5 task blocks.)
    off = np.array([[np.abs(gcells[p] - gcells[q]).max() <= 1
                     for q in range(P) if q != p] for p in range(P)], dtype=bool)
    shuffled_same = float(off.mean())
    null_ok = shuffled_same < 0.2
    utils.print_color(
        f'[G2] problem set: {P} problems, {n_goal_cells} distinct goal cell(s); over all '
        f'off-diagonal pairs {shuffled_same * 100:.0f}% of OTHER problems\' goals land '
        f'within +-1 cell of the true one (that is the null\'s built-in floor).',
        c=('c' if null_ok else 'r'))
    if not null_ok:
        utils.print_color(
            f'[G2] *** the permutation null below is DEGENERATE ({shuffled_same * 100:.0f}% '
            f'>= 20%) and means little: with only {n_goal_cells} distinct goal cell(s) among '
            f'{P} problems, scoring an endpoint against another problem\'s goal often scores '
            f'it against THE SAME goal. Use --bridge N for per-problem goals. ***', c='r')
    n_dec = min(n_dec, N)
    stride = 2
    idx = np.unique(np.concatenate([np.arange(0, H, stride), [H - 1]]))
    rows, unit = [], scanner.maze_unit
    for p in range(P):
        for n in range(n_dec):
            if D == 2:                       # o2d dump: plans are already xy
                m = np.asarray(plans[p, n][idx], dtype=np.float64)
            else:
                r = loc.localize_sequence(plans[p, n][idx].astype(np.float32),
                                          motion_sigma_cells=1.0, theta_blur=0)
                m = np.asarray(r['map_xy'], dtype=np.float64)
            jd = np.linalg.norm(np.diff(m, axis=0), axis=1)
            jumps = np.where(jd > unit)[0]
            st_err = float(np.linalg.norm(m[0] - starts[p]))
            en_err = float(np.linalg.norm(m[-1] - goals[p]))
            ci = np.array(scanner.world_to_cell(m[-1:, 0], m[-1:, 1])).ravel()
            cg = np.array(scanner.world_to_cell(goals[p:p + 1, 0], goals[p:p + 1, 1])).ravel()
            ## PERMUTATION NULL: score this endpoint against EVERY OTHER problem's
            ## goal and average. This is the empirical chance floor for G2b on
            ## exactly these problems -- no more arguing about whether 28% beats
            ## "~10%" -- and averaging over all q != p removes the problem-ORDERING
            ## artifact that a single q=(p+1)%P neighbour introduces.
            nulls = []
            for q in range(P):
                if q == p:
                    continue
                cq = np.array(scanner.world_to_cell(goals[q:q + 1, 0],
                                                    goals[q:q + 1, 1])).ravel()
                nulls.append(bool(np.abs(ci - cq).max() <= 1))
            row = dict(prob=p, n=n, n_jumps=int(len(jumps)),
                       jump_frames=(idx[jumps] if len(jumps) else np.array([], int)).tolist(),
                       start_err_wu=st_err, end_err_wu=en_err,
                       end_in_goal_cell=bool(np.abs(ci - cg).max() <= 1),
                       end_in_shuffled_goal_cell=float(np.mean(nulls)) if nulls else 0.0)
            if gt_xy is not None:
                ## did it reproduce the TRUE corridor, or merely land near the goal?
                dist = np.linalg.norm(m[:, None, :] - gt_xy[p][None, :, :], axis=2)
                row['gt_dev_wu'] = float(np.median(dist.min(axis=1)))
                row['gt_cover_wu'] = float(np.median(dist.min(axis=0)))
            rows.append(row)
    nj = np.array([r['n_jumps'] for r in rows])
    g2a = float((nj == 0).mean())
    g2b = float(np.mean([r['end_in_goal_cell'] for r in rows]))
    g2b_null = float(np.mean([r['end_in_shuffled_goal_cell'] for r in rows]))
    st_med = float(np.median([r['start_err_wu'] for r in rows]))
    en_med = float(np.median([r['end_err_wu'] for r in rows]))
    utils.print_color(
        f'[G2] {len(rows)} decoded plans: G2a jump-free {g2a * 100:.0f}% '
        f'(median {np.median(nj):.0f} multi-cell jumps/plan) | '
        f'G2b endpoint in goal cell(+-1) {g2b * 100:.0f}% | '
        f'start err med {st_med:.2f} wu | end err med {en_med:.2f} wu',
        c=('g' if g2a > 0.95 and g2b > 0.8 else 'r'))
    lift = g2b - g2b_null
    if null_ok:
        verdict = ('Above chance.' if lift > 0.15 else
                   'AT CHANCE: the endpoints carry no goal information.')
    else:
        verdict = 'MEANINGLESS on this problem set (see the warning above) -- ignore.'
    utils.print_color(
        f'[G2] G2b permutation null (same endpoints scored against another '
        f'problem\'s goal) = {g2b_null * 100:.0f}% -> LIFT {lift * 100:+.0f} pts. {verdict}',
        c=('g' if (null_ok and lift > 0.15) else 'r'))
    if gt_xy is not None:
        dev = np.median([r['gt_dev_wu'] for r in rows])
        cov = np.median([r['gt_cover_wu'] for r in rows])
        utils.print_color(
            f'[G2] vs GROUND-TRUTH route: median deviation {dev:.2f} wu '
            f'(decoded point -> nearest GT point), coverage {cov:.2f} wu '
            f'(GT point -> nearest decoded point).', c='c')
    utils.print_color(
        '[G2] BALL.md gate: pass = G2a>=0.95 & G2b>=0.8. Reference levels, do not '
        're-derive: G2b chance floor ~10% (random endpoint, +-1 cell) -- but prefer '
        'the permutation null printed above, it is measured on THESE problems; '
        'seam-teleport hypothesis REJECTED 3x (jumps sit at the +-12/104 = 24% '
        'uniform null); start_err ~0.4 wu is a round-trip artifact of the inpainted '
        'frame 0 and is NOT evidence the eval path is configured correctly -- see '
        'resolve_lidar_config.', c='y')
    with open(out_json, 'w') as f:
        json.dump(dict(g2a_jump_free=g2a, g2b_end_in_goal_cell=g2b,
                       g2b_permutation_null=g2b_null, g2b_lift=lift,
                       n_distinct_goal_cells=int(n_goal_cells),
                       permutation_null_valid=bool(null_ok),
                       start_err_med=st_med, end_err_med=en_med,
                       lidar_config={k: float(v) if isinstance(v, (int, float)) else v
                                     for k, v in lidar_cfg.items()},
                       rows=rows), f, indent=1)
    utils.print_color(f'[G2] wrote {out_json}', c='c')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='pointmaze-giant-stitch-v0')
    ap.add_argument('--planner_logdir', default='', help='LiDAR planner run dir (required unless --g2_npz)')
    ap.add_argument('--npz', default='',
                    help='full LiDAR npz from gen_lidar_dataset.py (normalizer source; required unless --g2_npz)')
    ap.add_argument('--g2_npz', default='',
                    help='run G2 on an EXISTING plans npz (CPU only, no model load) and exit. '
                         'Works for lidar (D=30, grid decode) and o2d (D=2, identity decode) dumps — '
                         'the o2d run is the harness control experiment.')
    ap.add_argument('--probs', default='data/ogb_maze/ev_probs/ogb_pointM_Gi_ev_prob_numEp20_eSdSt0.hdf5')
    ap.add_argument('--n_probs', type=int, default=20)
    ap.add_argument('--n_resamples', type=int, default=40)
    ap.add_argument('--n_comp', type=int, default=8, help='giant composes 8')
    ap.add_argument('--epoch', default='latest')
    ap.add_argument('--cond_w', type=float, default=2.0)
    ap.add_argument('--ddim_steps', type=int, default=50, help='0 = full T sampling')
    ap.add_argument('--ddim_eta', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='logs/lidar_eval/ball_plans_lidar.npz')
    ap.add_argument('--o2d_logdir', default='', help='optional upstream (x,y) planner run dir')
    ap.add_argument('--o2d_out', default='logs/lidar_eval/ball_plans_o2d.npz')
    ap.add_argument('--g2', action='store_true', help='run the G2 offline-decode gate on the lidar dump')
    ap.add_argument('--g2_n_dec', type=int, default=8, help='plans decoded per problem for G2')
    ap.add_argument('--xy_step', type=float, default=0.5)
    ## sensing geometry: normally auto-resolved from the ckpt; these only override
    ## (and are the ONLY way to set it in --g2_npz mode on a dump with no meta)
    ap.add_argument('--max_range', type=float, default=0.0,
                    help='override the ckpt lidar_config max_range (0 = read from ckpt/meta)')
    ap.add_argument('--n_beams', type=int, default=0,
                    help='override the ckpt lidar_config n_beams (0 = read from ckpt/meta)')
    ## ---- in-distribution BRIDGE mode (the model's own training task) ----
    ap.add_argument('--bridge', type=int, default=0,
                    help='N in-distribution bridge problems drawn from --npz instead of '
                         '--probs: (scan[t], scan[t+gap]) from a real episode, gap = the '
                         'composed horizon. Use with --n_comp 1 for the un-composed test.')
    ap.add_argument('--bridge_gap', type=int, default=0,
                    help='0 = auto (tot_hzn-1 for the given --n_comp)')
    ap.add_argument('--bridge_mode', default='gt', choices=['gt', 'spatial'],
                    help="'gt' = start/goal from ONE episode gap frames apart (needs an "
                         "episode longer than gap; all episodes here are 200 frames, so "
                         "gap<=199). 'spatial' = pace-matched via maze BFS distance, no "
                         "ground-truth route -- the only option for n_comp>=2.")
    ap.add_argument('--bridge_tol', type=float, default=0.2,
                    help='spatial mode: allowed fractional slack on the target maze distance')
    ap.add_argument('--bridge_pace', default='progress', choices=['progress', 'arc'],
                    help="spatial mode: how the maze-distance target is calibrated. "
                         "'progress' (default) = median BFS-path advance per frame of real "
                         "windows -- the same unit as the target. 'arc' = median arc length "
                         "per frame, the pre-2026-07-30 default, which over-shot the target "
                         "by the data's 2.53x dawdle factor and made every spatial ladder "
                         "arm 2.4-2.65x under-provisioned. Use 'arc' only to reproduce that.")
    ap.add_argument('--bridge_tort', type=float, default=None,
                    help='spatial mode: match the data tortuosity (straight-line separation / '
                         'BFS path; real 159-frame windows = 0.75). None = unconstrained, '
                         'which drifts to 0.47-0.61 (straighter than training).')
    ap.add_argument('--bridge_tort_tol', type=float, default=0.15,
                    help='absolute slack on --bridge_tort')
    ap.add_argument('--waypoints', action='store_true',
                    help='aim EVERY segment: give intermediate segment i a goal-stgl at a '
                         'maze-shortest-path waypoint instead of the current stgl_cond={} '
                         '(and drop its forward overlap, matching the trained stovlp_gl '
                         'mode). Diagnosis 2026-07-30: jump density is 0.32/100 frames in '
                         'the start-conditioned segment vs 0.89 in the unconditioned '
                         'interior, and the anchored+aimed n_comp=1 regime scores G2b 90%% '
                         'vs 42%% at n_comp=8. Inference-only, no retrain.')
    ap.add_argument('--blend_type', default='exp',
                    choices=['exp', 'cosine', 'linear', 'smoothstep', 'switch', 'switch_map'],
                    help="how the 56-frame segment overlaps are merged. 'exp' (default, the "
                         "bench setting) per-beam AVERAGES two segments' scans, which is "
                         "invalid for LiDAR -- the mean of two poses' scans is no pose's "
                         "scan. 'switch' hard-steps at the overlap midpoint so every frame "
                         "stays realizable; 'switch_map' also snaps to grid.free_scans.")
    ap.add_argument('--blend_beta', type=float, default=2.0,
                    help='sharpness for the weighted blend modes (bench used 2)')
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    epoch = args.epoch if args.epoch == 'latest' else int(float(args.epoch))

    probs = load_stgl_lh_ev_probs_hdf5(args.probs)
    starts = np.asarray(probs['start_state'], np.float64)[:args.n_probs, :2]
    goals = np.asarray(probs['goal_pos'], np.float64)[:args.n_probs, :2]

    ## ---- offline G2-only mode: re-score an existing dump, no GPU/model ----
    if args.g2_npz:
        dnpz = np.load(args.g2_npz, allow_pickle=True)
        plans = np.asarray(dnpz['plans'], np.float32)
        gt_xy = None
        if 'starts' in dnpz.files:
            ## bridge dumps carry their own problems -- self-describing, so a rescore
            ## can never be silently scored against the wrong start/goal set
            starts = np.asarray(dnpz['starts'], np.float64)
            goals = np.asarray(dnpz['goals'], np.float64)
            gt_xy = np.asarray(dnpz['gt_xy'], np.float64) if 'gt_xy' in dnpz.files else None
            utils.print_color(f'[audit] problems read from the dump itself '
                              f'({len(starts)} bridge problems), --probs ignored', c='g')
        assert plans.shape[0] == len(starts), \
            f'{plans.shape[0]} problems in npz vs {len(starts)} in --probs'
        meta = str(dnpz['meta']) if 'meta' in dnpz.files else None
        lidar_cfg = resolve_lidar_config(meta=meta, n_beams=args.n_beams,
                                         max_range=args.max_range)
        if plans.shape[-1] != 2 and 'lidar_config' not in str(meta) and not args.max_range:
            utils.print_color(
                '[audit] this dump predates lidar_config-in-meta: the decode geometry '
                'is NOT verifiable from the file. Pass --max_range explicitly.', c='r')
        g2_check(plans, starts, goals, args.env, args.xy_step, args.g2_n_dec,
                 os.path.splitext(args.g2_npz)[0] + '_g2.json', lidar_cfg, gt_xy=gt_xy)
        return

    assert args.planner_logdir and args.npz, '--planner_logdir and --npz required unless --g2_npz'
    d = np.load(args.npz)
    lidar = d['lidar'].astype(np.float32)
    actions = d['actions'].astype(np.float32) if 'actions' in d.files else np.zeros((len(lidar), 2), np.float32)
    xy = d['xy'].astype(np.float32)

    ## ---- LiDAR planner: condition on (start scan, goal scan), world frame yaw=0 ----
    ## geometry comes from the CHECKPOINT (see resolve_lidar_config for why)
    lidar_cfg = resolve_lidar_config(logdir=args.planner_logdir, n_beams=args.n_beams,
                                     max_range=args.max_range)
    assert_npz_matches(lidar, lidar_cfg, args.npz)

    norm = make_lidar_normalizer(lidar, actions)
    ## the map projector for --blend_type switch_map: built from the SAME resolved
    ## geometry as the decode grid, so it can never disagree with it
    projector = None
    if args.blend_type == 'switch_map':
        from diffuser.pl_eval.lidar_eval.lidar_grid_common import LG
        projector = make_map_projector(
            LG.load_or_build_grid(args.env, lidar_cfg, xy_step=args.xy_step, n_theta=1))
    pol, model, ep = build_policy(args.planner_logdir, epoch, args.device, args.cond_w,
                                  args.ddim_steps, args.ddim_eta, args.n_comp,
                                  args.n_resamples, norm, blend_type=args.blend_type,
                                  exp_beta=args.blend_beta, projector=projector)

    gt_xy = None
    if args.bridge:
        ## the composed horizon this --n_comp actually produces
        tot_hzn = args.n_comp * model.horizon - (args.n_comp - 1) * model.len_ovlp_cd
        gap = args.bridge_gap or tot_hzn - 1
        if args.bridge_mode == 'spatial':
            ## Arm B of the composition ladder: pace-matched, no ground truth.
            ## Required for n_comp>=2 -- every episode is 200 frames, so a
            ## GT-backed gap of 263+ does not exist in this dataset.
            starts, goals, st_sc, gl_sc, gt_xy = make_spatial_bridge_problems(
                d, lidar, xy, make_scanner_for_env(args.env, lidar_cfg),
                args.bridge, gap, args.seed, tol=args.bridge_tol,
                pace=args.bridge_pace, tort=args.bridge_tort,
                tort_tol=args.bridge_tort_tol)
        else:
            starts, goals, st_sc, gl_sc, gt_xy = make_bridge_problems(
                d, lidar, xy, args.bridge, gap, args.seed)
            if args.bridge_gap and args.n_comp > 1:
                utils.print_color(
                    f'[bridge] CONFOUND, stated: --bridge_gap {gap} with --n_comp '
                    f'{args.n_comp} gives the model {tot_hzn} frames for a {gap + 1}-frame '
                    f'journey, so it must DAWDLE (~{gap / tot_hzn:.0%} of training pace). '
                    f'This is Arm A of the ladder -- the controlled arm (identical problems '
                    f'across depths, GT route available); --bridge_mode spatial is the '
                    f'pace-matched cross-check.', c='y')
    else:
        scanner = make_scanner_for_env(args.env, lidar_cfg)
        st_sc = np.stack([scanner.scan(x, y, 0.0) for x, y in starts]).astype(np.float32)
        gl_sc = np.stack([scanner.scan(x, y, 0.0) for x, y in goals]).astype(np.float32)

    wps = wps_xy = None
    if args.waypoints:
        assert args.n_comp >= 2, '--waypoints needs --n_comp >= 2 (nothing to aim at n_comp=1)'
        wps, wps_xy = make_waypoints(make_scanner_for_env(args.env, lidar_cfg),
                                     starts, goals, args.n_comp)

    plans = sample_model(pol, st_sc, gl_sc, args.n_resamples, 'lidar',
                         goal_band_k=int(getattr(model, 'goal_band_k', 1)),
                         waypoints=wps)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    extra = dict(starts=starts, goals=goals) if args.bridge else {}
    if wps_xy is not None:
        extra['waypoints_xy'] = wps_xy          # (n_comp-1, P, 2), for post-hoc scoring
    if args.bridge and gt_xy is not None:      ## absent in spatial mode (no GT route)
        extra['gt_xy'] = gt_xy
    np.savez_compressed(args.out, plans=plans, problem_ids=np.arange(len(plans)),
                        meta=json.dumps(dict(env=args.env, logdir=args.planner_logdir,
                                             epoch=ep, cond_w=args.cond_w,
                                             ddim=args.ddim_steps, n_comp=args.n_comp,
                                             seed=args.seed, npz=args.npz,
                                             bridge=args.bridge,
                                             bridge_gap=(int(gap) if args.bridge else 0),
                                             bridge_mode=(args.bridge_mode if args.bridge else ''),
                                             blend_type=args.blend_type,
                                             ## self-describing: --g2_npz rescores of this
                                             ## dump reuse the exact training geometry
                                             lidar_config=_json_safe(lidar_cfg))),
                        **extra)
    utils.print_color(f'[dump/lidar] wrote {args.out} plans={plans.shape}', c='g')

    if args.g2:
        g2_check(plans, starts, goals, args.env, args.xy_step, args.g2_n_dec,
                 os.path.splitext(args.out)[0] + '_g2.json', lidar_cfg, gt_xy=gt_xy)

    ## ---- optional o2d twin: condition directly on (start xy, goal xy) ----
    if args.o2d_logdir:
        ## LimitsNormalizer over the same dataset's xy reproduces the o2d training
        ## normalizer up to dataset identity (same OGBench observations).
        norm2 = make_lidar_normalizer(xy, actions)
        ## the o2d twin keeps the default 'exp' blend ON PURPOSE: averaging two
        ## POSITIONS is a position, so weighted blending is valid for D=2 and is
        ## what the bench ran. Switching it here would break the control.
        pol2, model2, ep2 = build_policy(args.o2d_logdir, epoch, args.device, args.cond_w,
                                         args.ddim_steps, args.ddim_eta, args.n_comp,
                                         args.n_resamples, norm2)
        if args.bridge:
            ## the bridge gap was derived from the LiDAR model's geometry; the o2d
            ## twin is only a valid control if its horizon matches frame-for-frame
            assert (model2.horizon, model2.len_ovlp_cd) == (model.horizon, model.len_ovlp_cd), \
                (f'o2d geometry (hzn={model2.horizon}, ovlp={model2.len_ovlp_cd}) differs from '
                 f'lidar (hzn={model.horizon}, ovlp={model.len_ovlp_cd}) -- bridge gap would '
                 f'not be the o2d training horizon, so it is not a matched control')
        plans2 = sample_model(pol2, starts.astype(np.float32),
                              goals.astype(np.float32), args.n_resamples, 'o2d',
                              goal_band_k=int(getattr(model2, 'goal_band_k', 1)))
        np.savez_compressed(args.o2d_out, plans=plans2,
                            problem_ids=np.arange(len(plans2)),
                            meta=json.dumps(dict(env=args.env, logdir=args.o2d_logdir,
                                                 epoch=ep2, bridge=args.bridge,
                                                 bridge_gap=(int(gap) if args.bridge else 0))),
                            **extra)
        utils.print_color(f'[dump/o2d] wrote {args.o2d_out} plans={plans2.shape}', c='g')
        if args.g2:
            ## D=2 -> the decode is the identity, so this is the matched control:
            ## same problems, same sampler, same metric, a representation that
            ## carries spatial continuity by construction.
            g2_check(plans2, starts, goals, args.env, args.xy_step, args.g2_n_dec,
                     os.path.splitext(args.o2d_out)[0] + '_g2.json', lidar_cfg, gt_xy=gt_xy)


if __name__ == '__main__':
    main()
