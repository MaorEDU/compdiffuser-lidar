"""Unit tests for the two mechanisms the experiment rests on, with no GPU,
no checkpoint and no env: the replan cadence and the observation-history window.

Both are cheap to get subtly wrong and expensive to debug from a rollout, which is
exactly the class of bug the previous fork kept paying for.

Run: ``pytest diffuser/pl_eval/lidar_eval/test_ballv2_core.py``
"""

import numpy as np
import pytest

from diffuser.datasets.lidar.hist_window import left_pad_history
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_planner_v1 import periodic_replan_due


# --------------------------------------------------------------------------- #
# replan cadence
# --------------------------------------------------------------------------- #
def _auto_budget(n_steps, repl_every):
    """What plan_ogb_stgl_sml.py sizes max_n_repl to when the cadence is set."""
    return -(-n_steps // repl_every) + 2


def _fire_steps(repl_every, n_steps=1000, plan_len=888, max_n_repl=None):
    """Simulate the rollout loop's replan trigger over one episode.

    Mirrors the loop exactly: n_act_per_waypnt == 1 so wp_idx == i_et, plan 0 is
    generated at i_et == 0, and every fire resets prev_dfu_wp_idx."""
    if max_n_repl is None:
        max_n_repl = _auto_budget(n_steps, repl_every) if repl_every > 0 else 15
    prev = 0
    cnt = 0
    fires = []
    for i_et in range(1, n_steps):
        if periodic_replan_due(wp_idx=i_et, prev_dfu_wp_idx=prev,
                               last_plan_len=plan_len, repl_every=repl_every,
                               cnt_repl=cnt, max_n_repl=max_n_repl):
            fires.append(i_et)
            cnt += 1
            prev = i_et
    return fires


@pytest.mark.parametrize('every', [50, 100, 200])
def test_periodic_cadence_is_exact(every):
    """With the auto-sized budget the cadence must hold for the WHOLE episode.

    This is the regression test for a real bug: a fixed max_n_repl=15 truncated the
    REPL_EVERY=50 arm at step 750, i.e. the fastest cadence in the sweep would have
    silently run a different experiment for its last quarter."""
    fires = _fire_steps(every)
    assert fires == list(range(every, 1000, every)), fires


@pytest.mark.parametrize('every', [50, 100, 200])
def test_auto_budget_covers_the_whole_episode(every):
    assert _auto_budget(1000, every) >= len(range(every, 1000, every))


def test_zero_means_only_on_plan_exhaustion():
    ## repl_every=0 -> the only trigger left is "the plan ran out"
    assert _fire_steps(0, n_steps=1000, plan_len=888) == [888]
    assert _fire_steps(0, n_steps=500, plan_len=888) == []


def test_budget_caps_the_number_of_replans():
    fires = _fire_steps(50, n_steps=1000, max_n_repl=3)
    assert fires == [50, 100, 150]


@pytest.mark.parametrize('every', [25, 50, 100, 200])
def test_plan_count_stays_within_the_loops_sanity_bound(every):
    """The rollout loop asserts a per-episode plan-count bound. It used to be a hard
    20, which a fast cadence legitimately exceeds -- at REPL_EVERY=25 an episode holds
    41 plans -- so the bound is now derived from the arm's own replan budget. This
    test is what keeps the two definitions in agreement."""
    budget = _auto_budget(1000, every)
    n_plans = 1 + len(_fire_steps(every, max_n_repl=budget))   # plan 0 + the replans
    bound = max(20, budget + 1)
    assert n_plans <= bound, (n_plans, bound)


def test_plan_exhaustion_still_fires_under_a_long_period():
    ## a plan shorter than the period must still trigger when it is consumed
    assert _fire_steps(200, n_steps=1000, plan_len=120)[:3] == [120, 240, 360]


# --------------------------------------------------------------------------- #
# observation-history window
# --------------------------------------------------------------------------- #
def _frames(n, dim=4, base=0):
    return np.arange(base, base + n * dim, dtype=np.float32).reshape(n, dim)


def test_full_window_takes_the_most_recent_frames():
    f = _frames(50)
    out = left_pad_history(f, 32)
    assert out.shape == (32, 4)
    assert np.array_equal(out, f[-32:])


def test_short_window_repeats_the_oldest_frame_at_the_front():
    f = _frames(5)
    out = left_pad_history(f, 32)
    assert out.shape == (32, 4)
    ## the real frames land at the END, oldest-first order preserved
    assert np.array_equal(out[-5:], f)
    ## the front is the oldest frame repeated -- never zeros, never the newest
    assert np.array_equal(out[:27], np.repeat(f[:1], 27, axis=0))


def test_empty_window_is_the_fill_frame_repeated():
    fill = np.arange(4, dtype=np.float32)
    out = left_pad_history(np.empty((0, 4), dtype=np.float32), 32, fill=fill)
    assert out.shape == (32, 4)
    assert np.array_equal(out, np.repeat(fill[None], 32, axis=0))


def test_empty_window_without_a_fill_is_loud():
    with pytest.raises(AssertionError):
        left_pad_history(np.empty((0, 4), dtype=np.float32), 32)


def test_dataset_and_rollout_agree_on_the_same_frames():
    """The dataset slices a buffer; the rollout appends observed scans to a deque.
    Given the same underlying frames, both must produce the identical window --
    that equality is the whole reason `left_pad_history` exists."""
    from collections import deque
    episode = _frames(200, dim=4)
    hist_len, start = 32, 90

    ## dataset side: the hist_len frames before `start`
    lo = max(0, start - hist_len)
    dataset_win = left_pad_history(episode[lo:start], hist_len, fill=episode[start])

    ## rollout side: a maxlen deque fed one frame per control step
    dq = deque(maxlen=hist_len)
    dq.extend(left_pad_history(np.empty((0, 4), dtype=np.float32),
                               hist_len, fill=episode[0]))
    for t in range(start):
        dq.append(episode[t])
    rollout_win = np.asarray(dq, dtype=np.float32)

    assert np.array_equal(dataset_win, rollout_win)


def test_rollout_window_at_episode_start_is_the_start_frame_repeated():
    """The first control step of an episode: the deque must be a full window of the
    start observation, not a short window and not zeros -- that is the distribution
    the model's left-padded training windows put it in."""
    from collections import deque
    hist_len = 32
    start_scan = np.arange(4, dtype=np.float32)
    dq = deque(maxlen=hist_len)
    dq.extend(left_pad_history(np.empty((0, 4), dtype=np.float32),
                               hist_len, fill=start_scan))
    win = np.asarray(dq, dtype=np.float32)
    assert win.shape == (hist_len, 4)
    assert np.array_equal(win, np.repeat(start_scan[None], hist_len, axis=0))


# --------------------------------------------------------------------------- #
# ada_dist: the trigger must be a WORLD distance, not obs-dims vs plan-dims
# --------------------------------------------------------------------------- #
class _FakeLidarPlanner:
    """The two methods `_get_ada_dist` needs, with no env/model/checkpoint."""
    _pos_source = 'belief'

    def __init__(self, plan_xy, belief_xy):
        self._plan_xy = plan_xy
        self._belief_xy = belief_xy
        self._ada_no_plan = 0
        self._ada_dist_last = float('nan')

    def _pos(self):
        return self._belief_xy

    # the method under test, bound from the real class
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as _R)
    _get_ada_dist = _R._get_ada_dist
    del _R


def _plan_line(n=888, step=0.07):
    """A decoded plan walking +x at `step` wu per frame."""
    return np.stack([np.arange(n) * step, np.zeros(n)], axis=1)


def test_ada_dist_is_a_world_distance_from_the_decoded_plan():
    plan = _plan_line()
    p = _FakeLidarPlanner(plan, np.array([plan[100][0], 2.0]))
    ## exactly 2.0 wu off the waypoint the schedule says we should be at
    assert p._get_ada_dist(obs_cur=None, goal_cur=None, wp_idx=100) == pytest.approx(2.0)


def test_ada_dist_is_zero_when_on_schedule():
    plan = _plan_line()
    p = _FakeLidarPlanner(plan, plan[250].copy())
    assert p._get_ada_dist(obs_cur=None, goal_cur=None, wp_idx=250) == pytest.approx(0.0)


def test_ada_dist_does_not_fire_every_step_for_an_agent_tracking_the_plan():
    """The regression that cost this project the arm-C baseline: the old formula
    compared world (x, y) to beams 0 and 1, exceeded thres=1 at essentially every
    step, and burned the whole replan budget in the first `max_n_repl` steps."""
    plan = _plan_line()
    thres = 1.0
    fires = 0
    for i_et in range(1, 500):
        ## agent follows the plan with 0.3 wu of jitter -- never a real deviation
        pos = plan[i_et] + np.array([0.0, 0.3 * np.sin(i_et)])
        p = _FakeLidarPlanner(plan, pos)
        fires += int(p._get_ada_dist(None, None, i_et) > thres)
    assert fires == 0, f'trigger fired {fires}/499 steps on an on-schedule agent'


def test_ada_dist_fires_once_the_agent_falls_behind():
    plan = _plan_line()
    ## stuck at the origin while the schedule marches on
    p = _FakeLidarPlanner(plan, np.array([0.0, 0.0]))
    assert p._get_ada_dist(None, None, 5) < 1.0        # still close early
    assert p._get_ada_dist(None, None, 400) > 1.0      # far behind later


def test_ada_dist_without_a_decoded_plan_is_silent_and_counted():
    p = _FakeLidarPlanner(None, np.array([0.0, 0.0]))
    assert p._get_ada_dist(None, None, 10) == 0.0
    assert p._ada_no_plan == 1


def test_ada_dist_clamps_past_the_end_of_the_plan():
    plan = _plan_line(n=10)
    p = _FakeLidarPlanner(plan, plan[-1].copy())
    assert p._get_ada_dist(None, None, 10_000) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# plan decode: argmin vs the forward filter
#
# `_decode_plan_xy` had NO coverage before 2026-08-24, which is how a decoder that
# resolved only 24% of endpoints to the right cell shipped as the executor's aim.
# These bind the real method with a stub localizer, so no grid/checkpoint/GPU.
# --------------------------------------------------------------------------- #
class _FakeGrid:
    """A tiny pose lattice whose scans ARE the (x, y) they came from."""

    def __init__(self, xs=(0.0, 1.0, 2.0, 3.0), n_beams=4):
        self.free_xy = np.array([[x, 0.0] for x in xs], dtype=np.float64)
        ## scan of pose k = [x, 0, 0, ...] -> nearest-scan == nearest-x, exactly
        self.free_scans = np.zeros((len(xs), n_beams), dtype=np.float32)
        self.free_scans[:, 0] = [x for x in xs]


class _FakeLoc:
    def __init__(self, grid, seq_xy=None):
        self.grid = grid
        self._seq_xy = seq_xy
        self.calls = []

    def localize_sequence(self, scans, motion_sigma_cells=None, theta_blur=None,
                          **kw):
        self.calls.append(dict(scans=np.array(scans, copy=True),
                               motion_sigma_cells=motion_sigma_cells))
        n = len(scans)
        xy = (self._seq_xy[:n] if self._seq_xy is not None
              else np.stack([scans[:, 0], np.zeros(n)], axis=1))
        return dict(map_xy=np.asarray(xy, dtype=np.float64))


class _FakeScanner:
    maze_unit = 4.0

    ## `_pull_plan` drops the decode's wall points before pulling, so it needs the
    ## world<->cell pair. An all-free map keeps the decode tests measuring the decode.
    walls = frozenset()

    def world_to_cell(self, x, y):
        return (np.floor(np.asarray(y) / self.maze_unit).astype(int),
                np.floor(np.asarray(x) / self.maze_unit).astype(int))

    def _is_wall(self, i, j):
        i = np.asarray(i); j = np.asarray(j)
        if not self.walls:
            return np.zeros(i.shape, dtype=bool)
        out = np.zeros(i.shape, dtype=bool)
        for a, b in self.walls:
            out |= (i == a) & (j == b)
        return out


class _FakeDecoder:
    """`_decode_plan_xy` with only the attributes it actually touches."""

    def __init__(self, mode='argmin', stride=2, seq_xy=None, n_beams=4):
        self.n_beams = n_beams
        self.grid = _FakeGrid(n_beams=n_beams)
        self.loc = _FakeLoc(self.grid, seq_xy)
        self.scanner = _FakeScanner()
        self._decode_stride = stride
        self._decode_mode = mode
        self._decode_motion_sigma = 1.0
        self._decode_median = 1            # de-jitter off unless a test sets it
        self._decode_stats = []
        ## string-pull OFF by default, matching the shipped executor: every decode
        ## test below asserts the UNPULLED decode, so a pull here would silently
        ## rewrite what they measure.
        self._pull_win = 0
        self._pull_stride = 4
        self._pull_resample_wu = 0.4
        self._los_margin = 0.6
        self._pull_stats = []

    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as _R)
    _decode_plan_xy = _R._decode_plan_xy
    _decode_viterbi = _R._decode_viterbi
    _pull_plan = _R._pull_plan
    ## re-wrap: reading a staticmethod off the class yields a plain function,
    ## which this class body would otherwise turn back into an instance method
    _median_dejitter = staticmethod(_R._median_dejitter)
    del _R


def _ramp_plan(n=888, n_beams=4, step=0.003):
    """A scan plan whose beam-0 value ramps, i.e. a pose walking +x."""
    s = np.zeros((n, n_beams), dtype=np.float32)
    s[:, 0] = np.arange(n) * step
    return s


@pytest.mark.parametrize('mode', ['argmin', 'filter'])
@pytest.mark.parametrize('H', [888, 328, 3, 1])
def test_decode_preserves_length_and_index_alignment(mode, H):
    """`_advance_cursor` and `_get_ada_dist` index the decode by PLAN index."""
    d = _FakeDecoder(mode=mode)
    out = d._decode_plan_xy(_ramp_plan(n=H))
    assert out.shape == (H, 2), (mode, H, out.shape)
    assert np.isfinite(out).all()


def test_argmin_decode_snaps_to_the_nearest_grid_pose():
    """H < 3 skips the 3-tap smoothing, so this sees the raw snap."""
    d = _FakeDecoder(mode='argmin', stride=1)
    s = np.zeros((2, 4), dtype=np.float32)
    s[:, 0] = [0.1, 1.9]                       # -> lattice x of {0,1,2,3}
    out = d._decode_plan_xy(s)
    assert out[:, 0] == pytest.approx([0.0, 2.0])


def test_decode_output_is_box_smoothed_in_the_interior():
    """PID targets must not jump, so the interior is still the 3-tap mean; only the
    two anchor frames are pinned (see test_argmin_decode_pins_its_endpoints)."""
    d = _FakeDecoder(mode='argmin', stride=1)
    s = np.zeros((5, 4), dtype=np.float32)
    s[:, 0] = [0.1, 1.9, 2.6, 1.9, 0.1]        # snaps to 0, 2, 3, 2, 0
    out = d._decode_plan_xy(s)
    snapped = np.array([0.0, 2.0, 3.0, 2.0, 0.0])
    pad = np.pad(snapped, 1, mode='edge')
    expect = np.convolve(pad, np.ones(3) / 3.0, 'valid')
    assert out[1:-1, 0] == pytest.approx(expect[1:-1])      # interior smoothed
    assert out[[0, -1], 0] == pytest.approx(snapped[[0, -1]])  # anchors pinned


def test_argmin_decode_never_calls_the_sequence_filter():
    """The whole point of the default: no motion model, no belief to lock in."""
    d = _FakeDecoder(mode='argmin')
    d._decode_plan_xy(_ramp_plan())
    assert d.loc.calls == []


def test_filter_mode_still_uses_the_forward_filter():
    """`LIDAR_DECODE=filter` must reproduce the pre-2026-08-24 path exactly."""
    d = _FakeDecoder(mode='filter')
    d._decode_plan_xy(_ramp_plan())
    assert len(d.loc.calls) == 1
    assert d.loc.calls[0]['motion_sigma_cells'] == 1.0
    ## it is handed the STRIDED frames, last frame always included
    assert len(d.loc.calls[0]['scans']) == len(
        np.unique(np.concatenate([np.arange(0, 888, 2), [887]])))


def test_decode_mode_is_recorded_in_the_stats():
    """BALLV2 section 5 forbids comparing G2 across decoders -- so log which ran."""
    for mode in ('argmin', 'filter'):
        d = _FakeDecoder(mode=mode)
        d._decode_plan_xy(_ramp_plan())
        assert d._decode_stats[-1]['decode'] == mode


def test_the_two_decoders_disagree_when_the_filter_drifts():
    """A filter that has lost the plan must NOT look identical to argmin.

    Guards the test suite itself: if the stub made both decoders agree by
    construction, every test above would pass on a broken implementation.
    """
    n = 100
    drift = np.stack([np.full(n, 99.0), np.zeros(n)], axis=1)   # filter says "far away"
    a = _FakeDecoder(mode='argmin', stride=1)._decode_plan_xy(_ramp_plan(n=n))
    f = _FakeDecoder(mode='filter', stride=1, seq_xy=drift)._decode_plan_xy(
        _ramp_plan(n=n))
    assert np.linalg.norm(a - f, axis=1).min() > 4.0


def test_unknown_decode_mode_is_loud():
    """An unrecognised decoder must raise, never silently fall back."""
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as R)
    import inspect
    src = inspect.getsource(R.setup_load)
    assert "assert self._decode_mode in ('argmin', 'filter', 'viterbi')" in src


# --------------------------------------------------------------------------- #
# argmin de-jitter: the memoryless decode teleports on aliased frames, and
# `_carrot_idx` advances by ARC LENGTH, so raw argmin makes the ball crawl.
# --------------------------------------------------------------------------- #
def test_median_dejitter_removes_an_isolated_spike():
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as R)
    xy = np.zeros((21, 2))
    xy[:, 0] = np.arange(21) * 0.1
    xy[10, 0] = 50.0                                  # one aliased frame
    out = R._median_dejitter(xy, 5)
    assert abs(out[10, 0] - 1.0) < 0.3, out[10]


def test_median_dejitter_is_a_noop_below_width_3():
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as R)
    xy = np.random.default_rng(0).normal(size=(30, 2))
    assert np.array_equal(R._median_dejitter(xy, 1), xy)
    assert np.array_equal(R._median_dejitter(xy, 2), xy)


def test_median_dejitter_preserves_length_and_endpoints():
    """plan[-1] is the goal anchor; plan[0] is what head_err_vs_true reads."""
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as R)
    xy = np.stack([np.arange(40) * 0.25, np.zeros(40)], axis=1)
    out = R._median_dejitter(xy, 15)
    assert out.shape == xy.shape
    assert out[0] == pytest.approx(xy[0], abs=0.6)
    assert out[-1] == pytest.approx(xy[-1], abs=0.6)


def test_median_dejitter_shortens_a_jittery_path():
    """The operational point: arc length must come down, or the carrot stalls."""
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as R)
    rng = np.random.default_rng(0)
    clean = np.stack([np.arange(200) * 0.2, np.zeros(200)], axis=1)
    noisy = clean.copy()
    spike = rng.random(200) < 0.08
    noisy[spike, 1] += rng.normal(0, 8.0, spike.sum())
    arc = lambda a: np.linalg.norm(np.diff(a, axis=0), axis=1).sum()
    assert arc(R._median_dejitter(noisy, 15)) < 0.5 * arc(noisy)


def test_filter_decode_is_not_dejittered():
    """`LIDAR_DECODE=filter` must stay bit-identical to the pre-2026-08-24 path."""
    d = _FakeDecoder(mode='filter')
    d._decode_median = 15
    out = d._decode_plan_xy(_ramp_plan())
    d2 = _FakeDecoder(mode='filter')
    d2._decode_median = 1
    assert np.array_equal(out, d2._decode_plan_xy(_ramp_plan()))


def test_argmin_decode_pins_its_endpoints():
    """frame 0 and frame -1 must survive the 3-tap mean -- they are the two frames
    that carry real information (start anchor, goal anchor)."""
    d = _FakeDecoder(mode='argmin', stride=1)
    s = np.zeros((5, 4), dtype=np.float32)
    s[:, 0] = [0.1, 1.1, 2.1, 2.9, 0.1]      # snaps to 0, 1, 2, 3, 0
    out = d._decode_plan_xy(s)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[-1, 0] == pytest.approx(0.0)


def test_filter_decode_does_not_pin_endpoints():
    """The filter path stays the bit-identical pre-2026-08-24 reproduction."""
    d = _FakeDecoder(mode='filter', stride=1)
    s = np.zeros((5, 4), dtype=np.float32)
    s[:, 0] = [0.0, 1.0, 2.0, 3.0, 0.0]
    out = d._decode_plan_xy(s)
    ## edge-padded 3-tap mean of the last three: (3 + 0 + 0)/3
    assert out[-1, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# viterbi decode: the globally-consistent option added 2026-08-30.
#
# The property under test is NOT "the output is smooth" -- a hard step cap makes
# that true by construction, which is exactly why G2a's jump half stops carrying
# information for this decoder. What must be tested is that the constraint is a
# CONNECTIVITY constraint (a wall is impassable, not merely expensive) and that it
# actually changes which of two equally-good-looking poses is chosen.
# ---------------------------------------------------------------------------

class _WallScanner:
    """Two parallel free corridors (cell rows 0 and 2) split by a wall (row 1)."""

    maze_unit = 4.0

    def world_to_cell(self, x, y):
        j = np.floor_divide(np.asarray(x, dtype=np.float64) + 2.0, 4.0).astype(int)
        i = np.floor_divide(np.asarray(y, dtype=np.float64) + 2.0, 4.0).astype(int)
        return i, j

    def _is_wall(self, i, j):
        return np.asarray(i) == 1


class _WallGrid:
    """Poses with hand-set scans, so 'aliased' can be made exact rather than hoped for."""

    def __init__(self, xy, scans):
        self.free_xy = np.asarray(xy, dtype=np.float64)
        self.free_scans = np.asarray(scans, dtype=np.float32)
        self.scanner = _WallScanner()


def _wall_graph(grid, max_step=7.0):
    from diffuser.datasets.lidar.lidar_viterbi import FreeSpaceGraph
    return FreeSpaceGraph(grid, max_step=max_step)


def test_free_space_graph_refuses_to_cross_a_wall():
    """The whole reason this decoder exists: two poses can be near in Euclidean
    terms and unreachable in fact. `filter`'s Gaussian prior cannot say that."""
    grid = _WallGrid([[0.0, 1.0], [2.0, 1.0], [0.0, 7.0]],
                     [[0.0], [1.0], [2.0]])
    g = _wall_graph(grid, max_step=7.0)
    nb0 = set(g.nbr[0][g.valid[0]].tolist())
    assert 1 in nb0, 'same corridor, 2 wu apart -- must be reachable'
    assert 2 not in nb0, 'across a wall row -- must NOT be reachable'


def test_free_space_graph_always_admits_staying_put():
    """Every pose neighbours itself, so the DP can never dead-end and fall back to a
    restart. Restarts are the only teleport this decoder can emit, so this is the
    invariant that makes `jump_frac` structural rather than accidental."""
    grid = _WallGrid([[0.0, 1.0], [2.0, 1.0], [0.0, 7.0]],
                     [[0.0], [1.0], [2.0]])
    g = _wall_graph(grid, max_step=7.0)
    for p in range(g.M):
        assert p in set(g.nbr[p][g.valid[p]].tolist()), p


def test_viterbi_prefers_the_connected_pose_over_an_aliased_unreachable_one():
    """Poses 1 and 2 have IDENTICAL scans -- the aliasing that makes argmin teleport.
    Pose 2 is across a wall from the start, so only pose 1 explains the sequence."""
    from diffuser.datasets.lidar.lidar_viterbi import viterbi_decode_xy
    grid = _WallGrid(
        xy=[[0.0, 1.0], [2.0, 1.0], [2.0, 7.0]],
        ## pose 2 is listed so that a raw argmin over ties would take it (lower index
        ## wins ties in np.argmin, so make it strictly nearer to break toward it)
        scans=[[0.0], [5.0], [5.0]])
    g = _wall_graph(grid, max_step=7.0)
    scans = np.array([[0.0], [5.0]], dtype=np.float32)     # start pose, then the alias
    out = viterbi_decode_xy(scans, grid, g, beam=0, start_xy=[0.0, 1.0])
    assert out[0] == pytest.approx([0.0, 1.0])
    assert out[1] == pytest.approx([2.0, 1.0]), (
        'took the unreachable aliased pose; the connectivity constraint did nothing')


def test_viterbi_never_exceeds_max_step():
    """The structural guarantee. If this fails, `jump_frac` under viterbi is
    meaningless and any gate pass built on it is void."""
    from diffuser.datasets.lidar.lidar_viterbi import viterbi_decode_xy
    rng = np.random.default_rng(0)
    xy = np.stack([np.arange(0, 20, 0.5), np.full(40, 1.0)], axis=1)
    grid = _WallGrid(xy, rng.normal(size=(len(xy), 3)))
    g = _wall_graph(grid, max_step=2.0)
    scans = rng.normal(size=(25, 3)).astype(np.float32)    # noise -> maximal ambiguity
    out = viterbi_decode_xy(scans, grid, g, beam=0)
    step = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert step.max() <= 2.0 + 1e-9, step.max()


@pytest.mark.parametrize('H', [888, 328, 3, 1])
def test_viterbi_decode_preserves_length_and_index_alignment(H):
    d = _FakeDecoder(mode='viterbi', n_beams=4)
    d._vit_max_step, d._vit_beam = 4.0, 0
    d._vit_w_step, d._vit_sigma_e, d._vit_graph = 0.0, 1.0, None
    d.grid.scanner = _WallScanner()
    d.loc.grid.free_xy = np.stack(
        [d.grid.free_xy[:, 0], np.ones(len(d.grid.free_xy))], axis=1)
    out = d._decode_plan_xy(_ramp_plan(n=H))
    assert out.shape == (H, 2), (H, out.shape)
    assert np.isfinite(out).all()


def test_viterbi_decode_does_not_median_dejitter():
    """The median exists to erase argmin's isolated teleports. This decoder cannot
    emit one, so applying it would only round off the corners the constraint won."""
    d = _FakeDecoder(mode='viterbi', stride=1, n_beams=4)
    d._vit_max_step, d._vit_beam = 4.0, 0
    d._vit_w_step, d._vit_sigma_e, d._vit_graph = 0.0, 1.0, None
    d.grid.scanner = _WallScanner()
    d.loc.grid.free_xy = np.stack(
        [d.grid.free_xy[:, 0], np.ones(len(d.grid.free_xy))], axis=1)
    d._decode_median = 15
    out_med = d._decode_plan_xy(_ramp_plan(n=40))
    d2 = _FakeDecoder(mode='viterbi', stride=1, n_beams=4)
    d2._vit_max_step, d2._vit_beam = 4.0, 0
    d2._vit_w_step, d2._vit_sigma_e, d2._vit_graph = 0.0, 1.0, None
    d2.grid.scanner = _WallScanner()
    d2.loc.grid.free_xy = np.stack(
        [d2.grid.free_xy[:, 0], np.ones(len(d2.grid.free_xy))], axis=1)
    d2._decode_median = 1
    assert np.array_equal(out_med, d2._decode_plan_xy(_ramp_plan(n=40)))


def test_viterbi_decode_never_calls_the_sequence_filter():
    d = _FakeDecoder(mode='viterbi', n_beams=4)
    d._vit_max_step, d._vit_beam = 4.0, 0
    d._vit_w_step, d._vit_sigma_e, d._vit_graph = 0.0, 1.0, None
    d.grid.scanner = _WallScanner()
    d.loc.grid.free_xy = np.stack(
        [d.grid.free_xy[:, 0], np.ones(len(d.grid.free_xy))], axis=1)
    d._decode_plan_xy(_ramp_plan())
    assert d.loc.calls == []


# =========================================================================
# stall recovery: two rates, two risks (RETRAIN §8.5)
# =========================================================================

class _FakeCursor:
    """`_advance_cursor` / `_los_seek` lifted off the planner, with a scripted
    visibility oracle in place of the raycaster."""
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as _R)
    _advance_cursor = _R._advance_cursor
    del _R

    def __init__(self, plan, visible, stall_steps=5, blind_walk_steps=50,
                 los_shortcut_win=0, cursor_leash_wu=0.0):
        self._plan_xy = np.asarray(plan, dtype=np.float64)
        self._visible = visible          # set of plan indices with clear LOS
        self._cursor = 0
        self._stall = 0
        self._pursuit_win = 60
        self._stall_steps = stall_steps
        self._blind_walk_steps = blind_walk_steps
        self._stall_escapes = 0
        self._blind_walks = 0
        self._los_shortcut_win = los_shortcut_win
        self._los_shortcut_stride = 1
        self._los_shortcuts = 0
        self._los_shortcut_gain = 0
        ## leash OFF by default: the blind-walk tests above assert the shipped
        ## behaviour and must stay bit-identical.
        self._cursor_leash_wu = cursor_leash_wu
        self._pursuit_back = 15
        self._cursor_reacquires = 0
        self._los_margin = 0.6
        self.scanner = _FakeScanner()

    def _los_seek(self, pos, lo, hi, stride=1):
        vis = [i for i in range(lo + 1, hi, max(int(stride), 1)) if i in self._visible]
        return max(vis) if vis else None

    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as _R2)
    _cursor_reacquire = _R2._cursor_reacquire
    del _R2


def _trapped_plan(n=200):
    """Every plan point sits far from the ball, so nearest-snap never advances."""
    return np.stack([np.full(n, 500.0), np.arange(n, dtype=np.float64)], axis=1)


def test_los_recovery_fires_at_stall_steps_not_at_blind_walk_steps():
    """The SAFE recovery is the one that was throttled to 1-per-50. It must now fire
    on the fast timer, and must not wait for the blind timer."""
    c = _FakeCursor(_trapped_plan(), visible={7}, stall_steps=5, blind_walk_steps=50)
    for _ in range(5):
        c._advance_cursor(np.zeros(2))
    assert c._cursor == 7, c._cursor          # jumped to the visible point
    assert (c._stall_escapes, c._blind_walks) == (1, 0)


def test_blind_walk_stays_on_the_slow_timer_when_nothing_is_visible():
    """Running the unsafe `cursor+1` walk at `stall_steps` would advance ~200 indices
    an episode and outrun the ball past `pursuit_win`. It stays gated at 50."""
    c = _FakeCursor(_trapped_plan(), visible=set(), stall_steps=5, blind_walk_steps=50)
    for _ in range(49):
        c._advance_cursor(np.zeros(2))
    assert c._cursor == 0 and c._blind_walks == 0     # still holding at 49 steps
    c._advance_cursor(np.zeros(2))
    assert c._cursor == 1 and c._blind_walks == 1     # one index, at step 50
    assert c._stall_escapes == 0                      # never miscounted as an escape


def test_los_seek_is_retried_every_step_while_stalled():
    """Between `stall_steps` and `blind_walk_steps` the cursor is held but the cheap
    vectorised LOS probe is re-run, so recovery lands the step it becomes possible."""
    c = _FakeCursor(_trapped_plan(), visible=set(), stall_steps=5, blind_walk_steps=50)
    for _ in range(20):
        c._advance_cursor(np.zeros(2))
    assert c._cursor == 0                    # nothing visible yet, nothing forced
    c._visible = {11}                        # a wall clears
    c._advance_cursor(np.zeros(2))
    assert c._cursor == 11 and c._stall_escapes == 1


def test_escape_counters_are_separate_so_frozen_frac_cannot_be_misread():
    """`frozen_frac ~= stall_escapes * stall_steps / total` fitted the 12 bench runs at
    r=0.99 only because both recoveries shared one counter and one timer. Keep them
    distinguishable or that diagnostic silently lies again."""
    c = _FakeCursor(_trapped_plan(), visible=set(), stall_steps=5, blind_walk_steps=10)
    for _ in range(10):
        c._advance_cursor(np.zeros(2))
    c._visible = {30}
    for _ in range(5):
        c._advance_cursor(np.zeros(2))
    assert c._blind_walks == 1 and c._stall_escapes == 1


# =========================================================================
# LOS shortcut: the decoded plan is ~315 wu of arc against a 200 wu physical
# budget (0.2 wu/step x 1000), so 77 of 80 dumped plans cannot be driven at
# all. String-pulling the plan against the ball's own line of sight is what
# makes it drivable. Measured 2026-09-03; see `_advance_cursor`.
# =========================================================================

def _straight_plan(n=200):
    """A plan the ball is sitting ON, so nearest-snap advances one index at a time."""
    return np.stack([np.zeros(n), np.arange(n, dtype=np.float64) * 0.1], axis=1)


def test_shortcut_defaults_off_and_is_bit_identical():
    """The knob must default to 0 and reproduce the shipped cursor track exactly.
    Every number this project has produced was taken at win=0."""
    off = _FakeCursor(_straight_plan(), visible=set(range(200)), los_shortcut_win=0)
    ref = _FakeCursor(_straight_plan(), visible=set(range(200)))
    assert off._los_shortcut_win == 0
    for k in range(30):
        pos = np.array([0.0, k * 0.1])
        assert off._advance_cursor(pos) == ref._advance_cursor(pos)
    assert off._los_shortcuts == 0


def test_shortcut_skips_visible_arc_the_ball_never_needed_to_drive():
    """With a wide window the cursor jumps to the FURTHEST visible index, instead of
    crawling every wiggle. This is the whole point: arc 312 -> 80 wu."""
    c = _FakeCursor(_straight_plan(), visible={3, 40, 150}, los_shortcut_win=888)
    c._advance_cursor(np.zeros(2))
    assert c._cursor == 150, c._cursor
    assert c._los_shortcuts == 1 and c._los_shortcut_gain == 150


def test_shortcut_never_moves_the_cursor_backwards():
    """`_los_seek` searches only ahead of the cursor, so a visible point BEHIND the
    ball can never be selected -- the forward-only guarantee must survive the wide
    window, or the ball can be sent back down the plan it already ran."""
    c = _FakeCursor(_straight_plan(), visible={5, 90}, los_shortcut_win=888)
    c._advance_cursor(np.zeros(2))
    assert c._cursor == 90
    c._visible = {2, 10}                      # only points BEHIND are visible now
    assert c._advance_cursor(np.zeros(2)) == 90
    assert c._los_shortcuts == 1              # did not fire a second time


def test_shortcut_only_ever_picks_a_point_with_clear_line_of_sight():
    """Safety by construction: nothing visible => no shortcut, cursor untouched, and
    the stall machinery is left to handle it. A wide window must NOT degrade into the
    blind `cursor+1` walk."""
    c = _FakeCursor(_trapped_plan(), visible=set(), los_shortcut_win=888)
    for _ in range(20):
        c._advance_cursor(np.zeros(2))
    assert c._cursor == 0
    assert (c._los_shortcuts, c._blind_walks, c._stall_escapes) == (0, 0, 0)


def test_shortcut_stride_subsamples_candidates_without_losing_the_far_point():
    """The wide probe raycasts every control step, so candidates are strided. The
    lattice starts at ``lo + 1`` (NOT at the cursor), so from cursor 0 a stride of 4
    samples 1, 5, ..., 101 -- pinning that offset is the point of this test."""
    c = _FakeCursor(_straight_plan(), visible={101}, los_shortcut_win=888)
    c._los_shortcut_stride = 4
    c._advance_cursor(np.zeros(2))
    assert c._cursor == 101                   # 101 == 1 + 25*4, on the lattice
    c2 = _FakeCursor(_straight_plan(), visible={100}, los_shortcut_win=888)
    c2._los_shortcut_stride = 4
    c2._advance_cursor(np.zeros(2))
    assert c2._cursor == 0                    # 100 is not sampled; no false jump


# =========================================================================
# Plan string-pull (`_pull_plan`) -- the CORRECTED form of the shortcut.
# The runtime cursor version was refuted by benchmark runs: it pulled from
# the BALL's position, so the monotone cursor teleported past the route
# (cursor_max 215 -> 751 of 888, min_goal_dist 32.1 -> 30.2, corr = -0.16).
# This one pulls PLAN-POINT TO PLAN-POINT, so the result stays a connected
# route from start to goal -- which is what the 80-plan census measured.
# =========================================================================

class _FakePull:
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as _R)
    _pull_plan = _R._pull_plan
    del _R

    def __init__(self, win=888, stride=1, resample=0.4):
        self.scanner = _FakeScanner()
        self._los_margin = 0.6
        self._pull_win = win
        self._pull_stride = stride
        self._pull_resample_wu = resample
        self._pull_stats = []


def _detour_plan():
    """A straight run with a big pointless detour bolted into the middle -- the
    decoded plan's defect in miniature (6.2x the direct distance, all in free space)."""
    a = np.stack([np.linspace(0, 10, 40), np.zeros(40)], axis=1)
    b = np.stack([np.full(40, 10.0), np.linspace(0, 8, 40)], axis=1)
    c = np.stack([np.full(40, 10.0), np.linspace(8, 0, 40)], axis=1)
    d = np.stack([np.linspace(10, 20, 40), np.zeros(40)], axis=1)
    return np.concatenate([a, b, c, d], axis=0)


def _arc(xy):
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def test_pull_defaults_off_and_returns_the_plan_untouched():
    """win=0 must be bit-identical to the shipped executor."""
    p = _FakePull(win=0)
    xy = _detour_plan()
    out = p._pull_plan(xy)
    assert out is xy


def test_pull_removes_the_detour_arc(monkeypatch):
    """Everything visible => the pull collapses to the straight line, and the arc
    falls from 36 wu to 20. This is the census result (312 -> 80 wu) in miniature."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    monkeypatch.setattr(M, 'LOS_CLEAR', lambda sc, p0, pts, margin=0.6: np.ones(len(pts), bool))
    xy = _detour_plan()
    out = _FakePull()._pull_plan(xy)
    assert _arc(xy) > 35.0
    assert _arc(out) == pytest.approx(20.0, abs=0.5), _arc(out)


def test_pull_preserves_both_anchors_exactly():
    """The last point is the goal anchor the whole episode aims at (and what G2b
    scores); the first is what `head_err_vs_true` reads. Resampling must not move
    either, or the pull would trade the endpoint accuracy the decode earned."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    xy = _detour_plan()
    for vis in (True, False):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(M, 'LOS_CLEAR',
                       lambda sc, p0, pts, margin=0.6, v=vis: np.full(len(pts), v, bool))
            out = _FakePull()._pull_plan(xy)
            assert np.allclose(out[0], xy[0]), (vis, out[0], xy[0])
            assert np.allclose(out[-1], xy[-1]), (vis, out[-1], xy[-1])


def test_pull_is_a_noop_when_nothing_is_visible():
    """Nothing visible => the walk falls back to i+1 and the route is preserved.
    A pull must never SHORTEN a plan it could not verify."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, 'LOS_CLEAR', lambda sc, p0, pts, margin=0.6: np.zeros(len(pts), bool))
        xy = _detour_plan()
        out = _FakePull()._pull_plan(xy)
        assert _arc(out) == pytest.approx(_arc(xy), rel=0.02)


def test_pull_output_is_densely_resampled_for_the_cursor():
    """The pursuit cursor advances by NEAREST plan point, so a bare polyline with
    10 wu straights would strand it between corners. Consecutive spacing must stay
    at ~pull_resample_wu."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, 'LOS_CLEAR', lambda sc, p0, pts, margin=0.6: np.ones(len(pts), bool))
        out = _FakePull(resample=0.4)._pull_plan(_detour_plan())
        step = np.linalg.norm(np.diff(out, axis=0), axis=1)
        assert step.max() <= 0.45, step.max()
        assert len(out) >= 40


def test_pull_survives_the_repeated_points_the_median_dejitter_emits():
    """REGRESSION. The decoded plan is ~43% repeated consecutive points (median-15
    emits constant runs), which reach the pulled polyline as duplicate cumulative-arc
    values. `np.interp` needs a strictly increasing xp; with duplicates it scattered
    the resampled path off the route -- free_frac 0.988 -> 0.726, 27% of the plan
    inside walls, while arc and endpoint both still looked healthy."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, 'LOS_CLEAR', lambda sc, p0, pts, margin=0.6: np.zeros(len(pts), bool))
        ## every point duplicated: nothing visible, so the pull walks i+1 and the
        ## polyline it resamples is half zero-length segments
        xy = np.repeat(np.stack([np.linspace(0, 10, 40), np.zeros(40)], axis=1), 2, axis=0)
        out = _FakePull()._pull_plan(xy)
        assert np.isfinite(out).all()
        step = np.linalg.norm(np.diff(out, axis=0), axis=1)
        assert step.max() <= 0.45, step.max()
        ## must stay on the y=0 line it was given, not wander off it
        assert np.abs(out[:, 1]).max() < 1e-6, np.abs(out[:, 1]).max()
        assert out[:, 0].min() >= -1e-6 and out[:, 0].max() <= 10.0 + 1e-6


# =========================================================================
# The three fork guards (comp_diffuser_ball), adopted after
# the benchmark runs reproduced the exact cursor-runaway they were built to fix.
# =========================================================================

class _FakeCarrot:
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as _R)
    _carrot_idx = _R._carrot_idx
    del _R

    def __init__(self, plan, max_carrot_wu=0.0):
        self._plan_xy = np.asarray(plan, dtype=np.float64)
        self._lookahead_wu = 2.0
        self._los_margin = 0.6
        self._max_carrot_wu = max_carrot_wu
        self._los_pullbacks = 0
        self._los_blind = 0
        self.scanner = _FakeScanner()


def _break_plan():
    """A plan with a decode BREAK in it: index 5 teleports 40 wu away. 2 wu of ARC
    past the break is 40 wu of real DISTANCE -- the defect the cap exists for."""
    near = np.stack([np.linspace(0, 1.0, 6), np.zeros(6)], axis=1)
    far = np.stack([np.linspace(40.0, 42.0, 6), np.zeros(6)], axis=1)
    return np.concatenate([near, far], axis=0)


def test_max_carrot_cap_refuses_a_target_across_a_decode_break():
    """Uncapped, arc-lookahead walks straight across the break and hands the
    saturated PID a target 40 wu away. Capped, it must not."""
    plan = _break_plan()
    pos = np.zeros(2)
    loose = _FakeCarrot(plan)._carrot_idx(0, pos)
    tight = _FakeCarrot(plan, max_carrot_wu=4.0)._carrot_idx(0, pos)
    assert np.linalg.norm(plan[loose] - pos) > 30.0, plan[loose]
    assert np.linalg.norm(plan[tight] - pos) <= 4.0, plan[tight]


def test_max_carrot_holds_at_the_cursor_instead_of_aiming_at_the_far_target():
    """When NOTHING satisfies the cap, the shipped code fell through to the furthest
    lookahead point -- pressing a saturated PID into a wall for the rest of the
    episode. With a cap set it must hold at the cursor."""
    plan = _break_plan()
    c = _FakeCarrot(plan, max_carrot_wu=0.5)
    out = c._carrot_idx(5, plan[5] + np.array([20.0, 0.0]))   # ball far from everything
    assert out == 5, out
    assert c._los_blind == 1


def test_max_carrot_off_is_bit_identical():
    plan = _break_plan()
    pos = np.zeros(2)
    assert _FakeCarrot(plan, 0.0)._carrot_idx(0, pos) == _FakeCarrot(plan)._carrot_idx(0, pos)


def test_cursor_leash_reacquires_backward_instead_of_marching_away():
    """The blind march is correct while the cursor is near the ball and catastrophic
    once it is not: `lo` only increases and the LOS seek only looks forward, so
    nothing brings the carrot back. Past the leash the cursor must retreat."""
    plan = _trapped_plan()          # every point 500 wu away in x
    plan[:3] = np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0]])   # a few near the ball
    c = _FakeCursor(plan, visible=set(), stall_steps=5, blind_walk_steps=50,
                    cursor_leash_wu=4.0)
    c._cursor = 2
    for _ in range(50):
        c._advance_cursor(np.zeros(2))
    assert c._cursor <= 2, c._cursor              # retreated / held, never marched out
    assert c._cursor_reacquires >= 1
    assert c._blind_walks == 0


def test_cursor_leash_off_still_marches_forward():
    """Leash 0 must reproduce the shipped blind walk exactly."""
    c = _FakeCursor(_trapped_plan(), visible=set(), stall_steps=5, blind_walk_steps=50)
    for _ in range(50):
        c._advance_cursor(np.zeros(2))
    assert c._cursor == 1 and c._blind_walks == 1 and c._cursor_reacquires == 0


def test_cursor_leash_escapes_forward_when_nothing_behind_is_in_reach():
    """A leashed run must never be MORE stuck than the control arm: with nothing
    within the leash behind either, it escapes forward as before."""
    c = _FakeCursor(_trapped_plan(), visible=set(), stall_steps=5,
                    blind_walk_steps=50, cursor_leash_wu=4.0)
    for _ in range(50):
        c._advance_cursor(np.zeros(2))
    assert c._cursor == 1 and c._blind_walks == 1 and c._cursor_reacquires == 0


@pytest.mark.parametrize('dist_wu,expect', [(9.0, 2), (40.0, 4), (63.0, 6), (300.0, 8)])
def test_ada_ncomp_clips_composition_to_the_distance_left(dist_wu, expect):
    """Fixed n_comp=8 emits ~300 wu of plan arc for a 9 wu hop as readily as for a
    63 wu route. The clip is `ceil(cells/3.4)+1`, bounded to [2, n_max]."""
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as C)

    class F:
        _ada_ncomp_for_now = C._ada_ncomp_for_now
        _ada_ncomp, _ada_ncomp_cells = 1, 3.4
        scanner = _FakeScanner()
        def __init__(self, d):
            self._goal_xy_est = np.array([d, 0.0])
        def _pos(self):
            return np.zeros(2)
    assert F(dist_wu)._ada_ncomp_for_now(8) == expect


def test_ada_ncomp_keeps_the_configured_depth_when_the_goal_is_undecoded():
    """No goal estimate => None => the caller keeps the configured n_comp. A failed
    goal decode must never silently truncate the plan to the floor."""
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as C)

    class F:
        _ada_ncomp_for_now = C._ada_ncomp_for_now
        _ada_ncomp, _ada_ncomp_cells = 1, 3.4
        scanner = _FakeScanner()
        _goal_xy_est = None
        def _pos(self):
            return np.zeros(2)
    assert F()._ada_ncomp_for_now(8) is None


# =========================================================================
# FIX 9 -- goal capture (2026-09-04).
#
# The measurement it exists for: ~64 of 300 episodes come within 3 cells of the goal
# and 3-6 convert, and that `<12wu` count is 61-67 in EVERY arm ever run. Goal decode
# from the goal scan alone is 0.26 wu p50; the belief is 0.25 wu p50 on 86% of
# episodes. What loses it is `cursor_max` p50 = 215 of 888 -- the ball is beside the
# goal while the cursor aims at plan[217] somewhere else, and then it leaves.
#
# These assert the MECHANISM (when it engages, when it holds, when it releases, and
# above all WHAT IT AIMS AT), never the resulting trajectory.
# =========================================================================

class _FakeCapture:
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as _R)
    _capture_target = _R._capture_target
    del _R

    def __init__(self, goal_est, capture_wu=12.0, release_wu=None, stuck=0):
        self._goal_xy_est = None if goal_est is None else np.asarray(goal_est, float)
        self._capture_wu = capture_wu
        self._capture_release_wu = (1.5 * capture_wu) if release_wu is None else release_wu
        self._captured = False
        self._capture_engagements = 0
        self._capture_episodes = 0
        self._capture_steps = 0
        self._capture_step_cnt = 0
        self._capture_first_step = -1
        self._capture_dist_at_engage = float('nan')
        self._capture_near_min = float('inf')
        self._capture_in_range_steps = 0
        self._capture_blocked_steps = 0
        self._capture_margin = 0.6
        self._capture_stuck_steps = 0
        self._capture_releases_stuck = 0
        self._capture_best_d = float('inf')
        self._capture_no_gain = 0
        self._capture_stuck_steps = stuck
        self._ep_log = dict(true_xy=[None] * 7)   # 7 steps taken this episode
        self._los_margin = 0.6
        self.scanner = _FakeScanner()


def _los(visible):
    """Stand in for the raycaster: `visible` decides, not the geometry."""
    return lambda sc, p0, pts, margin=0.6: np.full(len(np.atleast_2d(pts)),
                                                   bool(visible))


def test_capture_off_is_bit_identical():
    """capture_wu=0 must return None on every call, so `_pid_lidar_act` falls through
    to exactly the carrot target the shipped executor computed."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    c = _FakeCapture([0.0, 0.0], capture_wu=0.0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        assert c._capture_target(np.array([0.5, 0.0])) is None
    assert c._captured is False and c._capture_engagements == 0


def test_capture_engages_inside_the_radius_with_clear_line_of_sight():
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        out = c._capture_target(goal + np.array([8.0, 0.0]))
    assert out is not None and np.allclose(out, goal)
    assert c._captured and c._capture_engagements == 1
    assert c._capture_first_step == 7   # within-episode index, not the run counter
    assert abs(c._capture_dist_at_engage - 8.0) < 1e-9


def test_capture_refuses_to_engage_outside_the_radius():
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        assert c._capture_target(goal + np.array([12.5, 0.0])) is None
    assert not c._captured


def test_capture_refuses_to_engage_through_a_wall():
    """In range but occluded: the PID is pure proportional, so aiming at an occluded
    goal is pressing into a wall until the episode ends. That is the failure the LOS
    pullback exists for and capture must not reintroduce it."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(False))
        assert c._capture_target(goal + np.array([3.0, 0.0])) is None
    assert not c._captured


def test_capture_holds_through_a_transient_occlusion():
    """Hysteresis is on DISTANCE ONLY. A corner briefly occludes the goal while the
    ball closes on it, and releasing on that would flicker the target between the
    goal and a plan index every few steps."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        c._capture_target(goal + np.array([8.0, 0.0]))
    with pytest.MonkeyPatch().context() as mp:      # now blind
        mp.setattr(M, 'LOS_CLEAR', _los(False))
        out = c._capture_target(goal + np.array([6.0, 0.0]))
    assert out is not None and np.allclose(out, goal)
    assert c._captured


def test_capture_holds_between_the_engage_and_release_radii():
    """Engaged at 8 wu, drifting to 15 wu: inside the 18 wu release radius it stays
    captured, so a single bad step cannot drop it back onto the wandering plan."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0)          # release = 18.0
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        c._capture_target(goal + np.array([8.0, 0.0]))
        assert c._capture_target(goal + np.array([15.0, 0.0])) is not None
    assert c._captured


def test_capture_releases_past_the_release_radius_and_can_re_engage():
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0)          # release = 18.0
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        c._capture_target(goal + np.array([8.0, 0.0]))
        assert c._capture_target(goal + np.array([19.0, 0.0])) is None
        assert not c._captured
        ## and it must be able to engage again on the next approach
        assert c._capture_target(goal + np.array([5.0, 0.0])) is not None
    assert c._capture_engagements == 2


def test_capture_records_why_it_did_not_engage():
    """An arm that never engages has two completely different explanations and the
    engagement counter cannot separate them. `capture_near_min` >> capture_wu means
    capture never had a chance and the blocker is upstream; a small `near_min` with a
    large `blocked` share means the radius or the LOS margin is what to look at.
    Without this, a null reads exactly like a bug -- which is the instrument gap this
    project has paid for twice."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])

    ## never in range: near_min records the closest approach, nothing is "blocked"
    far = _FakeCapture(goal, capture_wu=12.0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        far._capture_target(goal + np.array([40.0, 0.0]))
        far._capture_target(goal + np.array([25.0, 0.0]))
    assert far._capture_in_range_steps == 0 and far._capture_blocked_steps == 0
    assert abs(far._capture_near_min - 25.0) < 1e-9
    assert far._capture_first_step < 0

    ## in range every step and refused every step: that is the guard, not the geometry
    blind = _FakeCapture(goal, capture_wu=12.0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(False))
        for _ in range(4):
            blind._capture_target(goal + np.array([3.0, 0.0]))
    assert blind._capture_in_range_steps == 4 and blind._capture_blocked_steps == 4
    assert abs(blind._capture_near_min - 3.0) < 1e-9
    assert blind._capture_first_step < 0


def test_capture_counts_episodes_once_even_when_it_re_engages():
    """The pre-registered gate is stated in EPISODES (15-30%), but capture can release
    and re-engage several times within one episode. Conflating the two would inflate
    the gate and let a mis-set radius read as a pass."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0)          # release = 18.0
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        c._capture_target(goal + np.array([8.0, 0.0]))     # engage
        c._capture_target(goal + np.array([19.0, 0.0]))    # release
        c._capture_target(goal + np.array([5.0, 0.0]))     # engage again
    assert c._capture_engagements == 2
    assert c._capture_episodes == 1
    ## and the first-engage record is the FIRST one, not the latest
    assert abs(c._capture_dist_at_engage - 8.0) < 1e-9


def test_capture_is_a_noop_when_the_goal_scan_did_not_decode():
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    c = _FakeCapture(None, capture_wu=12.0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        assert c._capture_target(np.zeros(2)) is None


def test_capture_aims_at_the_DECODED_goal_and_never_at_a_privileged_one():
    """THE test. Capture is only in-constraint because it aims at `_goal_xy_est` --
    the agent's own argmin decode of the goal SCAN it was handed. Aiming at
    `_true_goal_xy`, `gl_pos` or `env.get_xy()` would be reading the answer, and the
    two are within a cell of each other most of the time, so a rollout would NOT
    show the difference. Assert the identity here or it can regress silently."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    decoded = np.array([30.0, 10.0])
    true_goal = np.array([31.7, 11.3])               # a cell away, as it really is
    c = _FakeCapture(decoded, capture_wu=12.0)
    c._true_goal_xy = true_goal                      # present, and must be ignored
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        out = c._capture_target(decoded + np.array([4.0, 0.0]))
    assert np.allclose(out, decoded)
    assert not np.allclose(out, true_goal)
    ## and the source must be the decode object itself, not a copy of the truth
    assert out is c._goal_xy_est or np.shares_memory(out, c._goal_xy_est)


def test_capture_counts_engaged_steps_for_the_mechanism_gate():
    """`capture_frac` and `ball_capture_episodes` are the PRE-REGISTERED gate: the
    arm should engage on 15-30% of episodes. A run whose counters stay 0 did not run
    the arm, whatever it scored."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        for _ in range(5):
            c._capture_target(goal + np.array([4.0, 0.0]))
    assert c._capture_steps == 5 and c._capture_step_cnt == 5
    assert c._capture_engagements == 1               # engaged once, held for 5 steps


# =========================================================================
# FIX 10 -- the belief latch guard.
# =========================================================================

def test_eps_mix_defaults_to_off_and_is_bit_identical():
    """0.0 must reproduce the pure-product update exactly: every number this project
    has produced was measured under it."""
    from diffuser.datasets.lidar.lidar_arrival import LiveLocalizer
    assert LiveLocalizer.__init__.__defaults__[-1] == 0.0


def test_eps_mix_rejects_an_out_of_range_value():
    from diffuser.datasets.lidar.lidar_arrival import LiveLocalizer

    class _L:
        _dense_free = np.ones((2, 2, 1))
    for bad in (-0.1, 1.0, 2.0):
        with pytest.raises(AssertionError):
            LiveLocalizer(_L(), eps_mix=bad)


def test_eps_mix_lets_a_latched_belief_recover():
    """`loc_err_p50` is bimodal -- 0.25 wu on 86% of episodes and ~45 wu on 14% --
    because the update is a pure product with no floor: once the true pose's mass
    underflows, no later scan can bring it back however well it matches. With a
    uniform floor it must recover; without one it must not. That contrast IS the fix."""
    def step(belief, like, eps):
        if eps > 0:
            u = np.ones_like(belief) / belief.size
            belief = (1 - eps) * belief + eps * u
        b = belief * like
        return b / b.sum()

    ## a belief fully latched on pose 0; every later scan says pose 3
    latched = np.array([1.0, 0.0, 0.0, 0.0])
    like = np.array([1e-9, 1e-9, 1e-9, 1.0])

    stuck = latched.copy()
    for _ in range(30):
        stuck = step(stuck, like, 0.0)
    assert stuck.argmax() == 0, 'no-floor filter must stay latched (that is the defect)'

    freed = latched.copy()
    for _ in range(30):
        freed = step(freed, like, 1e-3)
    assert freed.argmax() == 3, freed


# =========================================================================
# FIX 9b -- goal re-entry. The tier that makes capture reachable at all.
#
# A CPU census of the 5 eval goals says the goal is visible from only ~19% of free
# positions at 8-12 wu (6% on block 4), so a LOS-gated capture almost never engages:
# a benchmark run sat 8-9.7 wu from a goal it decoded to 0.2 wu and engaged 0 times in 4
# episodes. Euclidean proximity is not reachability in a maze. What gets through the
# corner is the plan -- whose approach to the goal is never executed (`cursor_max` p50
# 215 of 888). Re-entry makes that approach executable.
#
# The tests that matter are the two GUARDS, because this is one wrong clause away from
# being FIX 5 (jump the cursor, watch `carrot_dist_p50` go 1.95 -> 6.4, call phantom
# progress a win).
# =========================================================================

class _FakeReentry:
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as _R)
    _goal_reentry = _R._goal_reentry
    del _R

    def __init__(self, plan, goal_est, reentry_wu=12.0):
        self._plan_xy = np.asarray(plan, dtype=np.float64)
        self._goal_xy_est = None if goal_est is None else np.asarray(goal_est, float)
        self._reentry_wu = reentry_wu
        self._reentry_hops = 0
        self._reentry_ep_hops = 0


def _goal_region_plan():
    """A plan that wanders away and only comes back to the goal in its TAIL -- the
    real shape: `cursor_max` p50 215 of 888 means the tail is never executed."""
    out = np.stack([np.linspace(0.0, 60.0, 60), np.zeros(60)], axis=1)   # runs away
    back = np.stack([np.linspace(60.0, 2.0, 40), np.zeros(40)], axis=1)  # returns
    return np.concatenate([out, back], axis=0)                            # goal at ~2,0


def test_reentry_off_is_bit_identical():
    c = _FakeReentry(_goal_region_plan(), [2.0, 0.0], reentry_wu=0.0)
    assert c._goal_reentry(np.array([3.0, 0.0]), 5) is None
    assert c._reentry_hops == 0


def test_reentry_snaps_the_cursor_into_the_plans_goal_approach():
    """The whole point: the ball is beside the goal with the cursor stuck early in a
    plan that leads away. Re-entry must move it into the tail that comes back."""
    plan = _goal_region_plan()
    c = _FakeReentry(plan, [2.0, 0.0], reentry_wu=12.0)
    out = c._goal_reentry(np.array([4.0, 0.0]), 5)
    assert out is not None and out > 5
    assert np.linalg.norm(plan[out] - np.array([2.0, 0.0])) <= 12.0
    assert c._reentry_hops == 1


def test_reentry_does_not_fire_when_the_ball_is_far_from_its_decoded_goal():
    """It is a TERMINAL manoeuvre. Firing it mid-route would be FIX 5 with extra steps."""
    c = _FakeReentry(_goal_region_plan(), [2.0, 0.0], reentry_wu=12.0)
    assert c._goal_reentry(np.array([40.0, 0.0]), 5) is None
    assert c._reentry_hops == 0


def test_reentry_refuses_to_move_the_cursor_further_from_the_ball():
    """THE guard that separates this from FIX 5. FIX 5 jumped to the furthest index
    visible from the ball and `carrot_dist_p50` went 1.95 -> 6.4 wu, which is the
    cursor running away and progress becoming a phantom. Re-entry may only ever move
    the cursor CLOSER to the ball, so `carrot_dist` is bounded by construction."""
    ## cursor already sits exactly on the ball; every goal-region index is further
    plan = np.array([[0.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    c = _FakeReentry(plan, [5.0, 0.0], reentry_wu=12.0)
    assert c._goal_reentry(np.array([0.0, 0.0]), 0) is None
    assert c._reentry_hops == 0


def test_reentry_is_monotone_and_never_looks_backward():
    """The cursor is monotone everywhere except the leash. A backward re-entry would
    send the ball around a loop it has already run."""
    plan = _goal_region_plan()
    c = _FakeReentry(plan, [2.0, 0.0], reentry_wu=12.0)
    cur = 95                                   # already deep in the returning tail
    out = c._goal_reentry(plan[cur] + np.array([0.5, 0.0]), cur)
    assert out is None or out > cur


def test_reentry_is_a_noop_without_a_decoded_goal_or_a_plan():
    plan = _goal_region_plan()
    assert _FakeReentry(plan, None)._goal_reentry(np.zeros(2), 5) is None
    c = _FakeReentry(plan, [2.0, 0.0])
    c._plan_xy = None
    assert c._goal_reentry(np.zeros(2), 5) is None


def test_reentry_is_a_noop_when_the_plan_never_reaches_the_goal_region():
    """A severed plan whose tail lands somewhere else entirely must leave the cursor
    alone rather than snapping to the least-bad wrong index."""
    plan = np.stack([np.linspace(0.0, 60.0, 100), np.zeros(100)], axis=1)
    c = _FakeReentry(plan, [-40.0, 0.0], reentry_wu=12.0)
    assert c._goal_reentry(np.array([-38.0, 0.0]), 5) is None


def test_capture_margin_defaults_to_the_pursuit_margin_and_is_separate():
    """Capture gets its OWN LOS margin because `los_margin` also governs the carrot
    pullback and the stall recovery -- lowering it globally would change the whole
    controller and confound the arm. -1 must inherit, so the default is bit-identical."""
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as C)
    import inspect
    src = inspect.getsource(C.setup_load)
    assert "ball_capture_margin" in src and "self._capture_margin = self._los_margin" in src


def test_capture_uses_its_own_margin_not_the_pursuit_one():
    """The raycast that gates engagement must be called with `_capture_margin`. If it
    silently used `_los_margin` the arm would be a no-op and still look wired."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0)
    c._los_margin, c._capture_margin = 0.6, 0.25
    seen = {}

    def spy(sc, p0, pts, margin=0.6):
        seen['margin'] = margin
        return np.ones(len(np.atleast_2d(pts)), bool)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', spy)
        c._capture_target(goal + np.array([4.0, 0.0]))
    assert seen['margin'] == 0.25, seen


def test_capture_releases_when_it_stops_making_progress():
    """A looser margin can let a capture clip a corner and aim the saturated PID at a
    wall; release is on DISTANCE only, so without this valve nothing would free it.
    It also covers a badly decoded goal."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0, stuck=3)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        assert c._capture_target(goal + np.array([5.0, 0.0])) is not None   # engage
        for _ in range(2):                                                  # no gain
            assert c._capture_target(goal + np.array([5.0, 0.0])) is not None
        assert c._capture_target(goal + np.array([5.0, 0.0])) is None       # released
    assert not c._captured and c._capture_releases_stuck == 1


def test_capture_progress_resets_the_stuck_counter():
    """Closing on the goal must never trip the valve -- that is the whole success case."""
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0, stuck=3)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        for d in (9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0):
            assert c._capture_target(goal + np.array([d, 0.0])) is not None
    assert c._captured and c._capture_releases_stuck == 0


def test_capture_stuck_valve_off_by_default_never_releases():
    import diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 as M
    goal = np.array([30.0, 10.0])
    c = _FakeCapture(goal, capture_wu=12.0, stuck=0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(M, 'LOS_CLEAR', _los(True))
        for _ in range(50):
            assert c._capture_target(goal + np.array([5.0, 0.0])) is not None
    assert c._captured and c._capture_releases_stuck == 0


# --------------------------------------------------------------------------- #
# FIX 11 -- confidence-triggered replan
# --------------------------------------------------------------------------- #
class _AliasGrid:
    """Four poses. x=10 is >1 cell (4 wu) from x=0 and x=1 and scans ALMOST like
    x=0 -- a far lookalike. x=11 scans like nothing else."""

    def __init__(self, n_beams=4):
        self.free_xy = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
        self.free_scans = np.zeros((4, n_beams), dtype=np.float32)
        self.free_scans[:, 0] = [0.0, 1.0, 0.05, 5.0]


class _ConfDecoder(_FakeDecoder):
    def __init__(self, thr=0.3, run=3, min_gap=20):
        super().__init__(mode='argmin', stride=1)
        self.grid = _AliasGrid()
        self.loc = _FakeLoc(self.grid)
        self._conf_repl_thr = thr
        self._conf_repl_run = run
        self._conf_repl_min_gap = min_gap
        self._conf_ep_flags = []
        self._plan_flag_idx = None

    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as _R)
    _conf_flag = _R._conf_flag
    del _R


def _alias_plan(n=60, ambiguous=range(40, 60)):
    """Frames scan like the unique x=11 pose, except `ambiguous` ones, which sit in
    the near-tie between x=0 and its far lookalike x=10."""
    s = np.zeros((n, 4), dtype=np.float32)
    s[:, 0] = 5.0
    s[list(ambiguous), 0] = 0.02
    return s


def test_conf_flag_marks_the_first_ambiguous_run():
    d = _ConfDecoder()
    d._decode_plan_xy(_alias_plan())
    assert d._plan_flag_idx == 40
    assert d._conf_ep_flags == [40]


def test_conf_flag_needs_a_full_run():
    d = _ConfDecoder(run=3)
    d._decode_plan_xy(_alias_plan(ambiguous=[30, 31]))      # 2 frames < run of 3
    assert d._plan_flag_idx is None and d._conf_ep_flags == [-1]


def test_conf_flag_ignores_an_ambiguous_plan_head():
    """A flag inside the first min_gap frames would refire every step: ada latch."""
    d = _ConfDecoder(min_gap=20)
    d._decode_plan_xy(_alias_plan(ambiguous=range(0, 10)))
    assert d._plan_flag_idx is None


def test_conf_flag_never_changes_the_decoded_plan():
    """The flag is read-only: the plan the ball pursues is bit-identical on and off."""
    plan = _alias_plan()
    on, off = _ConfDecoder(thr=0.3), _ConfDecoder(thr=0.0)
    assert np.array_equal(on._decode_plan_xy(plan), off._decode_plan_xy(plan))
    assert off._plan_flag_idx is None and off._conf_ep_flags == []


def test_decode_without_the_knob_attribute_is_unchanged():
    """Offline shims (g2a_floor, pull_plan_audit) never set the knob; they must not
    break or start flagging."""
    d = _FakeDecoder(mode='argmin')
    out = d._decode_plan_xy(_ramp_plan(n=100))
    assert out.shape == (100, 2) and not hasattr(d, '_plan_flag_idx')


class _FakeTrigger:
    def __init__(self, thr=0.3, flag=40, cursor=0, captured=False, min_gap=20, extra=10):
        self._conf_repl_thr = thr
        self._plan_flag_idx = flag
        self._cursor = cursor
        self._captured = captured
        self._conf_repl_min_gap = min_gap
        self._conf_repl_extra = extra
        self._conf_repls = 0
        self._conf_ep_repls = 0
        self._conf_ep_fire_steps = []
        self._conf_ep_flag_lead = []

    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 as _R)
    _extra_replan_due = _R._extra_replan_due
    _extra_replan_budget = _R._extra_replan_budget
    del _R


def test_conf_trigger_fires_when_the_cursor_reaches_the_flag_and_consumes_it():
    t = _FakeTrigger(flag=40, cursor=41)
    assert t._extra_replan_due(wp_idx=130, prev_dfu_wp_idx=100)
    assert t._conf_ep_repls == 1 and t._conf_ep_fire_steps == [130]
    assert t._conf_ep_flag_lead == [1]
    assert not t._extra_replan_due(wp_idx=131, prev_dfu_wp_idx=100)   # consumed


@pytest.mark.parametrize('kw, wp', [
    (dict(cursor=39), 130),              # cursor short of the flag
    (dict(cursor=60), 110),              # 10 steps since the plan < min_gap 20
    (dict(cursor=60, captured=True), 130),
    (dict(cursor=60, flag=None), 130),   # plan had no flagged run
    (dict(cursor=60, thr=0.0), 130),     # knob off
])
def test_conf_trigger_holds(kw, wp):
    t = _FakeTrigger(**kw)
    assert not t._extra_replan_due(wp_idx=wp, prev_dfu_wp_idx=100)
    assert t._conf_repls == 0


def test_conf_trigger_budget_is_zero_when_off():
    assert _FakeTrigger(thr=0.0, extra=10)._extra_replan_budget() == 0
    assert _FakeTrigger(thr=0.3, extra=10)._extra_replan_budget() == 10


def test_base_planner_event_hooks_are_inert():
    """The periodic loop calls these on every non-scheduled step; the base must
    never fire and never add budget, or every existing arm changes."""
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_planner_v1 import (
        OgB_Stgl_Sml_MazeEnvPlanner_V1 as Base)
    assert Base._extra_replan_due(None, wp_idx=500, prev_dfu_wp_idx=0) is False
    assert Base._extra_replan_budget(None) == 0
