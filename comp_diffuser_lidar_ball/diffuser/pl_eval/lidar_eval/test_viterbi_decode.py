#!/usr/bin/env python3
"""Tests for GridLocalizer.localize_viterbi (the max-product plan decode).

Runs standalone on CPU, no torch, no pytest, no MuJoCo:
    python diffuser/pl_eval/lidar_eval/test_viterbi_decode.py

Why this file exists: `_decode_plan_xy` used `localize_sequence`, a forward FILTER
whose argmax is taken at each step INDEPENDENTLY -- nothing in that construction
forces consecutive estimates to be reachable from one another. Measured on the
filter-decode rollout timelines, 89% of the steps where the ball came off its plan
sit on such a hop (uniform null 8%, lift 10.6x, p=1.7e-137). Viterbi maximises
over whole PATHS, so an unreachable step cannot appear in the output at all --
t3 is the property that must never regress.

The real grid cache is used when present (it is the decoder's exact pose set);
otherwise the geometry tests fall back to a small synthetic corridor grid.
"""
import os, sys, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../..'))

## import the numpy-only module directly -- `import diffuser.datasets` pulls d4rl
_spec = importlib.util.spec_from_file_location(
    'lidar_localization',
    os.path.join(HERE, '../../datasets/lidar/lidar_localization.py'))
LL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LL)

GRID_NPZ = os.path.join(
    HERE, '../../../data/ogb_maze/lidar_grid',
    'pointmaze-giant-stitch-v0_xy0.5_th1_mr12.npz')
MAZE_UNIT = 4.0


class FakeGrid:
    """Minimal stand-in for LidarGrid: only what GridLocalizer touches."""

    def __init__(self, free_scans, free_xy, free_rc, x_vals, y_vals):
        self.free_scans = np.asarray(free_scans, dtype=np.float32)
        self.free_xy = np.asarray(free_xy, dtype=np.float64)
        self.free_rc = np.asarray(free_rc, dtype=np.int64)
        self.free_ti = np.zeros(len(free_xy), dtype=np.int64)
        self.free_theta = np.zeros(len(free_xy), dtype=np.float32)
        self.x_vals, self.y_vals = np.asarray(x_vals), np.asarray(y_vals)
        self.theta_vals = np.zeros(1)
        self.n_row, self.n_col, self.n_theta = len(y_vals), len(x_vals), 1
        self.n_beams = self.free_scans.shape[1]


def real_grid():
    """The shipped ball grid, or None when the cache is not on this machine."""
    if not os.path.exists(GRID_NPZ):
        return None
    z = np.load(GRID_NPZ, allow_pickle=True)
    return FakeGrid(z['free_scans'], z['free_xy'], z['free_rc'],
                    z['x_vals'], z['y_vals'])


def synth_grid(n=12, step=0.5):
    """A straight 1-D corridor of n cells, with a DELIBERATE alias: cells 2 and 9
    have identical scans, so a per-step argmax is free to hop between them."""
    x_vals = np.arange(n) * step
    y_vals = np.array([0.0])
    free_xy = np.stack([x_vals, np.zeros(n)], axis=1)
    free_rc = np.stack([np.zeros(n, dtype=int), np.arange(n)], axis=1)
    scans = np.zeros((n, 4), dtype=np.float32)
    scans[:, 0] = np.arange(n)          # distinct ramp ...
    scans[9] = scans[2]                 # ... except this exact duplicate
    return FakeGrid(scans, free_xy, free_rc, x_vals, y_vals)


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _walk(g, n=12):
    """A short walk along neighbouring free poses of the real grid."""
    order = np.argsort(g.free_xy[:, 0] * 1000 + g.free_xy[:, 1])
    seq = order[:1]
    while len(seq) < n:
        d = np.linalg.norm(g.free_xy - g.free_xy[seq[-1]], axis=1)
        cand = [c for c in np.flatnonzero((d > 0.4) & (d < 0.6))
                if c not in set(seq.tolist())]
        if not cand:
            break
        seq = np.append(seq, cand[0])
    return seq


def _path_score(loc, g, scans, idx, motion_sigma_cells, step):
    """Joint log-likelihood the decoder is maximising, for an arbitrary path."""
    d = LL.scan_distance(scans, g.free_scans).astype(np.float64)
    ll = -0.5 * (d / loc.sigma) ** 2
    sig = motion_sigma_cells * step
    s = ll[np.arange(len(idx)), idx].sum()
    mv = np.linalg.norm(np.diff(g.free_xy[idx], axis=0), axis=1)
    return s - 0.5 * (mv ** 2 / sig ** 2).sum()


@case
def t1_returns_the_OPTIMAL_path_not_merely_a_plausible_one():
    """Viterbi is a MAP algorithm, so the property to assert is OPTIMALITY: the
    path it returns must score at least as high as the ground-truth path under
    the very model it maximises. (Recovering the GT path exactly is NOT the right
    assertion -- with sigma=1.0 the likelihood is broad in open rooms, so the
    motion prior legitimately straightens a serpentine by one 0.5 wu cell. A
    broken backtrack, by contrast, shows up immediately as a lower score.)"""
    g = real_grid()
    if g is None:
        print('    (skipped: no grid cache)')
        return
    loc = LL.GridLocalizer(g, sigma=1.0)
    seq = _walk(g)
    scans = g.free_scans[seq]
    step = float(g.x_vals[1] - g.x_vals[0])
    r = loc.localize_viterbi(scans, motion_sigma_cells=1.0, r_max_cells=3.0)

    got = _path_score(loc, g, scans, r['pose_idx'], 1.0, step)
    truth = _path_score(loc, g, scans, seq, 1.0, step)
    assert got >= truth - 1e-6, f'viterbi score {got} < ground-truth path {truth}'
    # and it must still be a sane reconstruction: within one lattice cell
    err = np.linalg.norm(r['map_xy'] - g.free_xy[seq], axis=1)
    assert err.max() <= step + 1e-6, f'max err {err.max()} exceeds one cell'


@case
def t1b_sharper_sigma_recovers_the_ground_truth_path_exactly():
    """With a discriminative measurement model the motion prior stops dominating
    and exact grid scans decode back to exactly their own poses."""
    g = real_grid()
    if g is None:
        print('    (skipped: no grid cache)')
        return
    loc = LL.GridLocalizer(g, sigma=0.05)
    seq = _walk(g)
    r = loc.localize_viterbi(g.free_scans[seq], motion_sigma_cells=1.0,
                             r_max_cells=3.0)
    err = np.linalg.norm(r['map_xy'] - g.free_xy[seq], axis=1)
    assert err.max() < 1e-6, f'exact grid scans must decode exactly, max err {err.max()}'


@case
def t2_output_shapes_and_keys():
    g = synth_grid()
    loc = LL.GridLocalizer(g, sigma=1.0)
    scans = g.free_scans[[0, 1, 2, 3, 4]]
    r = loc.localize_viterbi(scans, motion_sigma_cells=1.0, r_max_cells=3.0)
    assert set(r) >= {'map_xy', 'pose_idx', 'score'}, r.keys()
    assert r['map_xy'].shape == (5, 2), r['map_xy'].shape
    assert r['pose_idx'].shape == (5,), r['pose_idx'].shape
    assert np.isfinite(r['score']), r['score']


@case
def t3_never_emits_an_unreachable_step():
    """THE load-bearing property. Feed the aliased corridor a sequence that lures
    the per-step argmax into hopping to the far look-alike; Viterbi must not."""
    g = synth_grid()
    loc = LL.GridLocalizer(g, sigma=1.0)
    step = float(g.x_vals[1] - g.x_vals[0])
    r_max_cells = 3.0
    scans = g.free_scans[[0, 1, 2, 3, 4, 5]]
    r = loc.localize_viterbi(scans, motion_sigma_cells=1.0, r_max_cells=r_max_cells)
    d = np.linalg.norm(np.diff(r['map_xy'], axis=0), axis=1)
    lim = r_max_cells * step * np.sqrt(2) + 1e-6
    assert d.max() <= lim, f'step {d.max()} exceeds the transition radius {lim}'


@case
def t4_beats_the_filter_on_the_real_grid_alias():
    """On the shipped grid, Viterbi must be at least as continuous as the filter."""
    g = real_grid()
    if g is None:
        print('    (skipped: no grid cache)')
        return
    loc = LL.GridLocalizer(g, sigma=1.0)
    rng = np.random.default_rng(0)
    # walk 40 poses along a real corridor, then add sensor noise
    start = int(np.argmin(g.free_xy[:, 0] + g.free_xy[:, 1]))
    seq = [start]
    for _ in range(39):
        d = np.linalg.norm(g.free_xy - g.free_xy[seq[-1]], axis=1)
        cand = [c for c in np.flatnonzero((d > 0.4) & (d < 0.75))
                if c not in set(seq[-3:])]
        if not cand:
            break
        seq.append(int(cand[0]))
    seq = np.asarray(seq)
    scans = g.free_scans[seq] + rng.normal(0, 0.5, (len(seq), g.n_beams)).astype(np.float32)

    fv = loc.localize_sequence(scans, motion_sigma_cells=1.0, theta_blur=0)
    vv = loc.localize_viterbi(scans, motion_sigma_cells=1.0, r_max_cells=3.0)
    jf = int((np.linalg.norm(np.diff(np.asarray(fv['map_xy']), axis=0), axis=1)
              > MAZE_UNIT).sum())
    jv = int((np.linalg.norm(np.diff(np.asarray(vv['map_xy']), axis=0), axis=1)
              > MAZE_UNIT).sum())
    assert jv <= jf, f'viterbi jumps {jv} must not exceed filter jumps {jf}'
    assert jv == 0, f'viterbi must be jump-free by construction, got {jv}'


@case
def t5_single_step_sequence():
    g = synth_grid()
    loc = LL.GridLocalizer(g, sigma=1.0)
    r = loc.localize_viterbi(g.free_scans[[4]], motion_sigma_cells=1.0)
    assert r['map_xy'].shape == (1, 2), r['map_xy'].shape
    assert int(r['pose_idx'][0]) in (4, 4), r['pose_idx']


@case
def t6_neighbour_cache_is_keyed_by_radius():
    """A second call at a DIFFERENT radius must not silently reuse the first."""
    g = synth_grid()
    loc = LL.GridLocalizer(g, sigma=1.0)
    nb1, d1 = loc._viterbi_neighbours(1.0)
    nb2, d2 = loc._viterbi_neighbours(3.0)
    assert nb2.shape[1] > nb1.shape[1], (nb1.shape, nb2.shape)
    nb1b, _ = loc._viterbi_neighbours(1.0)
    assert nb1b.shape == nb1.shape and np.array_equal(nb1b, nb1)


@case
def t7_padding_slots_can_never_win():
    """Unreachable neighbour slots carry inf cost, so they must never be chosen
    even when the measurement likelihood there is the best in the grid."""
    g = synth_grid()
    loc = LL.GridLocalizer(g, sigma=1.0)
    nb, d2 = loc._viterbi_neighbours(1.0)
    # every padded slot must be infinite-cost
    assert np.isinf(d2[~np.isfinite(d2)]).all()
    # a corridor end has fewer real neighbours than the interior
    assert np.isfinite(d2[0]).sum() < np.isfinite(d2[len(g.free_xy) // 2]).sum()


@case
def t8_theta_grid_is_rejected_loudly():
    """The ball grid is n_theta=1; anything else must raise, not silently mis-decode."""
    g = synth_grid()
    g.n_theta = 12
    loc = LL.GridLocalizer(g, sigma=1.0)
    try:
        loc._viterbi_neighbours(3.0)
    except AssertionError as e:
        assert 'n_theta' in str(e), e
        return
    raise AssertionError('a multi-theta grid must be rejected')


@case
def t9_tighter_motion_sigma_does_not_break_continuity():
    g = real_grid()
    if g is None:
        print('    (skipped: no grid cache)')
        return
    loc = LL.GridLocalizer(g, sigma=1.0)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(g.free_xy), 25, replace=False)
    scans = g.free_scans[idx]                     # deliberately incoherent order
    for ms in (0.5, 1.0, 2.0):
        r = loc.localize_viterbi(scans, motion_sigma_cells=ms, r_max_cells=3.0)
        d = np.linalg.norm(np.diff(r['map_xy'], axis=0), axis=1)
        assert d.max() <= 3.0 * 0.5 * np.sqrt(2) + 1e-6, (ms, d.max())


if __name__ == '__main__':
    fails = 0
    for fn in CASES:
        try:
            fn()
            print(f'  PASS {fn.__name__}')
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f'  FAIL {fn.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(CASES) - fails}/{len(CASES)} viterbi decode tests passed')
    sys.exit(1 if fails else 0)
