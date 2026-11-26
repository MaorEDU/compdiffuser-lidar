#!/usr/bin/env python3
"""Tests for the bridge-problem PACE calibration (ball_dump_plans).

Runs standalone on CPU, no GPU, no torch needed for the synthetic cases:
    python diffuser/pl_eval/lidar_eval/test_bridge_pace.py

Why this file exists: on 2026-07-30 we found that
``make_spatial_bridge_problems`` places a goal at a target **maze-BFS distance**
but calibrated that target with ``measure_speed``, which returns **arc length**
per frame. In a *stitch* dataset the demonstrations wander (measured dawdle
arc/BFS = 2.53), so every "pace-matched" spatial problem ever run sat 2.4-2.65x
farther than the model can reach in the frames provided -- and the resulting
"novel start/goal pairs collapse 98% -> 18%" headline was substantially "the
goal was unreachable". Case 1 below is that exact unit error: a trajectory that
oscillates between two adjacent cells has a large arc and *almost no* BFS
progress, so an arc-calibrated target overshoots by the oscillation count.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env  # noqa: E402

ENV = 'pointmaze-giant-stitch-v0'
CFG = dict(n_beams=30, max_range=12.0)
NPZ = 'data/ogb_maze/lidar/pointmaze-giant-stitch-v0_lidar.npz'

_PASS, _FAIL = [], []


def check(name, fn):
    try:
        fn()
        _PASS.append(name); print(f'  PASS {name}')
    except Exception as e:                                      # noqa: BLE001
        _FAIL.append((name, e)); print(f'  FAIL {name}: {e}')


def _scanner():
    return make_scanner_for_env(ENV, CFG)


def _adjacent_free_cells(sc):
    """Two horizontally adjacent free cells, as world-coordinate centres."""
    mp = np.asarray(sc.maze_map)
    for i, j in np.argwhere(mp != 1):
        if j + 1 < mp.shape[1] and mp[i, j + 1] != 1:
            u = sc.maze_unit
            return (np.array([j * u - sc.offset_x, i * u - sc.offset_y]),
                    np.array([(j + 1) * u - sc.offset_x, i * u - sc.offset_y]))
    raise AssertionError('no adjacent free cell pair in the maze')


def _oscillating_episode(a, b, gap, n_laps):
    """gap+1 frames bouncing a->b->a->... ending at b. Arc >> BFS progress."""
    pts = []
    for k in range(n_laps):
        end = b if k % 2 == 0 else a
        beg = a if k % 2 == 0 else b
        n = (gap + 1) // n_laps
        pts.append(beg + (end - beg) * np.linspace(0, 1, n, endpoint=False)[:, None])
    xy = np.concatenate(pts, 0)
    ## pad/trim to exactly gap+1 frames, finishing on b
    while len(xy) < gap + 1:
        xy = np.concatenate([xy, b[None]], 0)
    xy = xy[:gap + 1]
    xy[-1] = b
    return xy


def t1_progress_speed_is_not_arc_speed_when_the_agent_dawdles():
    """THE BUG: arc/frame and BFS-advance/frame differ by the dawdle factor."""
    from diffuser.pl_eval.lidar_eval.ball_dump_plans import measure_progress_speed
    sc = _scanner()
    a, b = _adjacent_free_cells(sc)
    gap, n_laps = 39, 5
    ep = _oscillating_episode(a, b, gap, n_laps)
    ## 4 identical episodes so the sampler has something to draw
    xy = np.concatenate([ep] * 4, 0)
    term = np.zeros(len(xy), bool)
    for k in range(4):
        term[(k + 1) * (gap + 1) - 1] = True
    prog, arc, med_path, med_arc, tort = measure_progress_speed(
        dict(terminals=term), xy, sc, gap, n_sample=4, seed=0)

    assert abs(med_path - sc.maze_unit) < 1e-6, \
        f'BFS advance should be exactly one cell ({sc.maze_unit}), got {med_path}'
    assert med_arc > 3 * med_path, \
        f'oscillating episode should have arc >> BFS ({med_arc} vs {med_path})'
    dawdle = arc / prog
    assert dawdle > 3, f'dawdle factor should be large here, got {dawdle:.2f}'
    ## and this is precisely the over-shoot an arc-calibrated BFS target incurs
    assert abs((arc * gap) / (prog * gap) - dawdle) < 1e-9


def t2_progress_and_arc_agree_on_a_straight_run():
    """Sanity: with no dawdling the two units coincide, so the bug is silent
    on straight data -- which is why it survived (o2d/short windows look fine)."""
    from diffuser.pl_eval.lidar_eval.ball_dump_plans import measure_progress_speed
    sc = _scanner()
    mp = np.asarray(sc.maze_map)
    u = sc.maze_unit
    ## find a run of >=4 free cells in one row
    row = None
    for i in range(mp.shape[0]):
        js = [j for j in range(mp.shape[1]) if mp[i, j] != 1]
        for s in range(len(js) - 3):
            if js[s + 3] == js[s] + 3:
                row = (i, js[s]); break
        if row: break
    assert row, 'no straight 4-cell corridor found'
    i, j0 = row
    gap = 30
    ep = np.stack([np.linspace(j0 * u - sc.offset_x, (j0 + 3) * u - sc.offset_x, gap + 1),
                   np.full(gap + 1, i * u - sc.offset_y)], 1)
    xy = np.concatenate([ep] * 4, 0)
    term = np.zeros(len(xy), bool)
    for k in range(4):
        term[(k + 1) * (gap + 1) - 1] = True
    prog, arc, med_path, med_arc, tort = measure_progress_speed(
        dict(terminals=term), xy, sc, gap, n_sample=4, seed=0)
    assert abs(med_arc - med_path) < 1e-6, f'straight run: {med_arc} vs {med_path}'
    assert abs(arc / prog - 1.0) < 1e-6, f'dawdle should be 1.0, got {arc/prog:.3f}'
    assert abs(tort - 1.0) < 1e-6, f'straight run tortuosity should be 1.0, got {tort}'


def t3_real_data_dawdles_by_about_2_5x():
    """Regression-pins the two constants §1 depends on. Skips without the npz."""
    if not os.path.exists(NPZ):
        print('    (skipped: no npz)'); return
    from diffuser.pl_eval.lidar_eval.ball_dump_plans import measure_progress_speed
    d = np.load(NPZ)
    prog, arc, med_path, med_arc, tort = measure_progress_speed(
        d, d['xy'].astype(np.float64), _scanner(), 159, seed=0)
    assert abs(arc - 0.1919) < 0.002, f'arc speed drifted: {arc:.4f}'
    assert abs(prog - 0.0755) < 0.002, f'progress speed drifted: {prog:.4f}'
    assert 2.3 < arc / prog < 2.8, f'dawdle drifted: {arc/prog:.2f}'
    assert abs(tort - 0.75) < 0.05, f'data tortuosity drifted: {tort:.2f}'


def t4_spatial_problems_are_actually_pace_matched():
    """The end-to-end invariant: achieved BFS path / frames == progress speed.

    This is the assertion whose absence let 2.65x through. Skips without the npz.
    """
    if not os.path.exists(NPZ):
        print('    (skipped: no npz)'); return
    from diffuser.pl_eval.lidar_eval.ball_dump_plans import (
        make_spatial_bridge_problems, measure_progress_speed, maze_bfs_wu)
    d = np.load(NPZ)
    lidar, xy = d['lidar'].astype(np.float32), d['xy'].astype(np.float64)
    sc = _scanner()
    prog = measure_progress_speed(d, xy, sc, 159, seed=0)[0]
    for nc in (1, 2):
        gap = 160 + 104 * (nc - 1) - 1
        s, g, ss, gs, gt = make_spatial_bridge_problems(
            d, lidar, xy, sc, 20, gap, 0, pace='progress', tort=0.75)
        assert s.shape == (20, 2) and ss.shape == (20, 30) and gt is None
        ci, cj = sc.world_to_cell(s[:, 0], s[:, 1])
        gi, gj = sc.world_to_cell(g[:, 0], g[:, 1])
        paths = np.array([maze_bfs_wu(sc, (int(ci[k]), int(cj[k])))[int(gi[k]), int(gj[k])]
                          for k in range(20)])
        pace = (np.median(paths) / gap) / prog
        assert 0.75 <= pace <= 1.25, f'n_comp={nc}: xDataPace {pace:.2f}, not pace-matched'


def t5_arc_mode_reproduces_the_old_overshoot():
    """`--bridge_pace arc` must still be able to rebuild the 2026-07-29 ladder."""
    if not os.path.exists(NPZ):
        print('    (skipped: no npz)'); return
    from diffuser.pl_eval.lidar_eval.ball_dump_plans import (
        make_spatial_bridge_problems, measure_progress_speed, maze_bfs_wu)
    d = np.load(NPZ)
    lidar, xy = d['lidar'].astype(np.float32), d['xy'].astype(np.float64)
    sc = _scanner()
    prog = measure_progress_speed(d, xy, sc, 159, seed=0)[0]
    gap = 159
    s, g, ss, gs, gt = make_spatial_bridge_problems(
        d, lidar, xy, sc, 20, gap, 0, pace='arc')
    ci, cj = sc.world_to_cell(s[:, 0], s[:, 1])
    gi, gj = sc.world_to_cell(g[:, 0], g[:, 1])
    paths = np.array([maze_bfs_wu(sc, (int(ci[k]), int(cj[k])))[int(gi[k]), int(gj[k])]
                      for k in range(20)])
    pace = (np.median(paths) / gap) / prog
    assert pace > 2.0, f'arc mode should over-shoot ~2.5x, got {pace:.2f}'


if __name__ == '__main__':
    for nm, fn in sorted(globals().items()):
        if nm.startswith('t') and callable(fn) and nm[1].isdigit():
            check(nm, fn)
    print(f'\n{len(_PASS)}/{len(_PASS) + len(_FAIL)} bridge-pace tests passed')
    sys.exit(1 if _FAIL else 0)
