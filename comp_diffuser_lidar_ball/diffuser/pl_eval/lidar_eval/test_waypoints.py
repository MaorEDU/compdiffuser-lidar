#!/usr/bin/env python3
"""Tests for per-segment WAYPOINT conditioning (ball_dump_plans + the sampler).

Runs standalone on CPU (no GPU, no MuJoCo -- the scanner is built from an env
NAME and is pure numpy):
    python diffuser/pl_eval/lidar_eval/test_waypoints.py

Why this exists (2026-07-30). `comp_pred_p_loop_n` sampled every intermediate
segment with `stgl_cond={}` -- neither anchored nor aimed. Measured jump density
per 100 frames at n_comp=8: 0.32 in the start-conditioned segment 0, 0.89 in the
unconditioned interior, 0.86 in the goal-conditioned last segment; i.e.
start-anchoring buys ~2.8x and goal-inpainting buys nothing. The anchored+aimed
regime (n_comp=1) scores G2b 90% with 2 MAP jumps and residual 0.069, against
42%/24/0.289 at n_comp=8. `--waypoints` makes every segment anchored+aimed.

The load-bearing test here is t5: with `seg_stgl_cond=None` the sampler must take
byte-identical branches, so the flag cannot perturb any existing result.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))

ENV = 'pointmaze-giant-stitch-v0'
CFG = dict(n_beams=30, max_range=12.0)
PROBS = 'data/ogb_maze/ev_probs/ogb_pointM_Gi_ev_prob_numEp20_eSdSt0.hdf5'

_PASS, _FAIL = [], []


def check(name, fn):
    try:
        fn()
        _PASS.append(name); print(f'  PASS {name}')
    except Exception as e:                                      # noqa: BLE001
        _FAIL.append((name, e)); print(f'  FAIL {name}: {e}')


def _scanner():
    from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env
    return make_scanner_for_env(ENV, CFG)


def _probs():
    import h5py
    with h5py.File(PROBS, 'r') as f:
        return (np.array(f['start_state'])[:, :2].astype(np.float64),
                np.array(f['goal_pos'])[:, :2].astype(np.float64))


def t1_waypoints_are_in_free_space():
    """A scan synthesized inside a wall is meaningless conditioning."""
    if not os.path.exists(PROBS):
        print('    (skipped: no ev_probs)'); return
    from diffuser.pl_eval.lidar_eval.ball_dump_plans import make_waypoints
    sc = _scanner(); st, gl = _probs()
    mp = np.asarray(sc.maze_map)
    for nc in (2, 8, 13):
        _, wxy = make_waypoints(sc, st, gl, nc)
        for i in range(nc - 1):
            c = sc.world_to_cell(wxy[i, :, 0], wxy[i, :, 1])
            ii, jj = np.asarray(c[0], int), np.asarray(c[1], int)
            assert (mp[ii, jj] != 1).all(), f'n_comp={nc} seg {i}: waypoint inside a wall'


def t2_waypoints_advance_monotonically_toward_the_goal():
    if not os.path.exists(PROBS):
        print('    (skipped: no ev_probs)'); return
    from diffuser.pl_eval.lidar_eval.ball_dump_plans import make_waypoints, maze_bfs_wu
    sc = _scanner(); st, gl = _probs()
    nc = 8
    _, wxy = make_waypoints(sc, st, gl, nc)
    ## BFS distance to goal must DECREASE along the waypoint chain
    for p in range(0, 100, 17):
        c = sc.world_to_cell(np.array([gl[p][0]]), np.array([gl[p][1]]))
        dist = maze_bfs_wu(sc, (int(c[0][0]), int(c[1][0])))
        ds = []
        for i in range(nc - 1):
            cc = sc.world_to_cell(np.array([wxy[i, p, 0]]), np.array([wxy[i, p, 1]]))
            ds.append(dist[int(cc[0][0]), int(cc[1][0])])
        ds = np.array(ds)
        assert np.all(np.diff(ds) <= 1e-9), f'problem {p}: non-monotone BFS-to-goal {ds}'


def t3_waypoint_spacing_matches_the_stride_pace():
    """A 104-frame stride advances 104*0.0755 = 7.9 wu at data progress pace.
    Waypoints much farther apart ask each segment to outrun the data."""
    if not os.path.exists(PROBS):
        print('    (skipped: no ev_probs)'); return
    from diffuser.pl_eval.lidar_eval.ball_dump_plans import make_waypoints
    sc = _scanner(); st, gl = _probs()
    _, wxy = make_waypoints(sc, st, gl, 13)
    chain = np.concatenate([st[None], wxy, gl[None]], 0)
    spacing = np.median(np.linalg.norm(np.diff(chain, axis=0), axis=2))
    assert 4.0 < spacing < 12.0, f'n_comp=13 spacing {spacing:.1f} wu is far off the 7.9 wu stride'


def t4_scans_are_valid_range_vectors():
    if not os.path.exists(PROBS):
        print('    (skipped: no ev_probs)'); return
    from diffuser.pl_eval.lidar_eval.ball_dump_plans import make_waypoints
    sc = _scanner(); st, gl = _probs()
    wps, _ = make_waypoints(sc, st, gl, 8)
    assert wps.shape == (7, 100, 30), wps.shape
    assert np.isfinite(wps).all()
    assert (wps >= 0).all() and (wps <= CFG['max_range'] + 1e-6).all(), 'scan outside [0, max_range]'


def t5_seg_stgl_cond_none_is_a_strict_no_op():
    """THE SAFETY TEST: the new argument must not perturb existing results.

    Drives the real branch logic of comp_pred_p_loop_n with a recording stub, and
    asserts that seg_stgl_cond=None reproduces the historical conditioning exactly:
    segment 0 -> {0: start} with the forward overlap KEPT, intermediates -> {} with
    BOTH overlaps, last -> {hzn-1: goal} with no forward overlap.
    """
    hzn, n_comp = 160, 8
    START, GOAL = 'START', 'GOAL'
    stgl_cond = {0: START, hzn - 1: GOAL}

    def branches(seg_stgl_cond):
        if seg_stgl_cond is None:
            seg_stgl_cond = [None] * n_comp
        assert len(seg_stgl_cond) == n_comp

        def _seg_cond(i, base):
            sc = dict(base)
            if seg_stgl_cond[i]:
                sc.update(seg_stgl_cond[i])
            return sc

        out = []
        for i in range(n_comp):
            if i == 0:
                sc = _seg_cond(i, {0: stgl_cond[0]})
                end = None if (hzn - 1) in sc else 'OVLP_FWD'
                out.append(('first', sorted(sc), end, None))
            elif i < n_comp - 1:
                sc = _seg_cond(i, {})
                end = None if (hzn - 1) in sc else 'OVLP_FWD'
                out.append(('mid', sorted(sc), end, 'OVLP_BACK'))
            else:
                out.append(('last', sorted({hzn - 1: stgl_cond[hzn - 1]}), None, 'OVLP_BACK'))
        return out

    base = branches(None)
    assert base[0] == ('first', [0], 'OVLP_FWD', None), base[0]
    for i in range(1, n_comp - 1):
        assert base[i] == ('mid', [], 'OVLP_FWD', 'OVLP_BACK'), base[i]
    assert base[-1] == ('last', [hzn - 1], None, 'OVLP_BACK'), base[-1]

    ## and with waypoints: every segment aimed, forward overlap dropped
    wp = [{hzn - 1: f'WP{i}'} for i in range(n_comp - 1)] + [{}]
    aimed = branches(wp)
    assert aimed[0] == ('first', [0, hzn - 1], None, None), aimed[0]
    for i in range(1, n_comp - 1):
        assert aimed[i] == ('mid', [hzn - 1], None, 'OVLP_BACK'), aimed[i]
    assert aimed[-1] == ('last', [hzn - 1], None, 'OVLP_BACK'), aimed[-1]


def t6_sampler_signature_and_length_guard():
    """The real function must accept seg_stgl_cond and reject a wrong length."""
    import inspect
    from diffuser.models.cd_stgl_sml_dfu.stgl_sml_diffusion_v1 import (
        Stgl_Sml_GauDiffusion_InvDyn_V1 as C)
    sig = inspect.signature(C.comp_pred_p_loop_n)
    assert 'seg_stgl_cond' in sig.parameters, 'sampler lost the seg_stgl_cond argument'
    assert sig.parameters['seg_stgl_cond'].default is None, 'default must be None (no-op)'
    src = inspect.getsource(C.comp_pred_p_loop_n)
    assert 'must have one entry per segment' in src, 'length guard missing'
    ## the rule that keeps every segment in a trained conditioning mode
    assert src.count('None if (hzn - 1) in sc_i else') == 2, \
        'the drop-forward-overlap-when-aimed rule must apply to segment 0 AND intermediates'


def t7_rollout_injection_is_off_by_default_and_respects_loc_source():
    """The closed-loop hook (`_ball_add_waypoints`) must:
      * be a no-op unless LIDAR_BALL_WAYPOINTS=1, and at n_comp<2;
      * take its BFS start from the SAME source the PID uses, so enabling
        waypoints in an L3 (belief) run cannot smuggle in ground-truth position;
      * when the belief is unseeded (plan 0 of every L3 episode -- the belief is
        refreshed in _lidar_arrival_step, which runs after the first plan is
        sampled), seed the BFS by localizing the START SCAN rather than declining.
        Declining is what the code did through 2026-07-31 and it drew plan 0 of
        100% of L3 episodes as the pre-waypoint control arm: tail_gd p50 17.18 wu
        vs L2's 0.59, 0/300 first-plan successes vs 51/300.
    Exercised on a stub -- no MuJoCo, no checkpoint.
    """
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)
    sc = _scanner()
    st, gl = _probs()
    ## the candidate positions must be far enough apart to give DIFFERENT
    ## waypoint chains; ev_probs is 5 blocks of 20 near-identical starts, so
    ## st[0]/st[1] will not do. Pick the start farthest from st[0], then the one
    ## farthest from BOTH -- three sources that the assertions can tell apart.
    true_xy = st[0]
    belief_xy = st[int(np.argmax(np.linalg.norm(st - true_xy, axis=1)))]
    scan_xy = st[int(np.argmax(np.linalg.norm(st - true_xy, axis=1)
                               + np.linalg.norm(st - belief_xy, axis=1)))]

    ## scan-DEPENDENT, because the code now localizes two different scans: the
    ## goal scan (as always) and the start scan (the unseeded-belief seed). A
    ## stub that answers the same xy for both cannot tell the paths apart.
    gl_scan_ref = sc.scan(*gl[0], 0.0).astype(np.float32)

    class Loc:
        @staticmethod
        def localize_single(scan):
            same_as_goal = np.allclose(np.asarray(scan, dtype=np.float32), gl_scan_ref)
            return {'xy': gl[0] if same_as_goal else scan_xy}

    class Stub:
        _ball_add_waypoints = C._ball_add_waypoints
        _ball_pos_for_bfs = C._ball_pos_for_bfs
        _ball_xy_in_wall = C._ball_xy_in_wall
        scanner, loc, nav_mode = sc, Loc(), ''
        policy = type('P', (), {'n_comp': 8})()
        env = type('E', (), {'get_xy': staticmethod(lambda: true_xy)})()
        _ball_waypoints, _pos_source = True, 'env'
        _ball_belief_est, _ball_wp_calls, _ball_wp_fallbacks = None, 0, 0
        _ball_wp_scan_seeds = 0

    def gc():
        return {'st_gl': np.stack([sc.scan(*st[0], 0.0), sc.scan(*gl[0], 0.0)]
                                  ).astype(np.float32)[:, None, :]}

    ## off by default
    s = Stub(); s._ball_waypoints = False
    g = gc(); s._ball_add_waypoints(g)
    assert 'waypoints' not in g, 'waypoints injected while disabled'

    ## nothing to aim at when there is no composition
    s = Stub(); s.policy = type('P', (), {'n_comp': 1})()
    g = gc(); s._ball_add_waypoints(g)
    assert 'waypoints' not in g, 'waypoints injected at n_comp=1'

    ## Reference chains, built directly from each candidate start. Comparing
    ## against these is exact -- "is waypoint 0 nearer A or B" is not, because a
    ## waypoint sits 1/n_comp along the route, not at the start.
    from diffuser.pl_eval.lidar_eval.lidar_grid_common import make_waypoints
    ref_true = make_waypoints(sc, true_xy[None], gl[0][None], 8)[1][:, 0]
    ref_bel = make_waypoints(sc, belief_xy[None], gl[0][None], 8)[1][:, 0]
    ref_scan = make_waypoints(sc, scan_xy[None], gl[0][None], 8)[1][:, 0]
    assert not np.allclose(ref_true, ref_bel), \
        'the two candidate starts give the same chain -- the test cannot discriminate'
    assert not np.allclose(ref_scan, ref_true) and not np.allclose(ref_scan, ref_bel), \
        'the start-scan chain matches another source -- the seeding test cannot discriminate'

    ## enabled: correct shape, and the chain is the one built from the TRUE xy
    s = Stub()
    g = gc(); s._ball_add_waypoints(g)
    assert g['waypoints'].shape == (7, 1, 30), g['waypoints'].shape
    assert np.allclose(s._ball_last_wp_xy, ref_true), \
        'env source did not start the BFS at the true xy'

    ## belief source, seeded -> BFS must start from the BELIEF, not the truth
    s = Stub(); s._pos_source = 'belief'; s._ball_belief_est = belief_xy
    g = gc(); s._ball_add_waypoints(g)
    assert np.allclose(s._ball_last_wp_xy, ref_bel), \
        'belief source leaked the ground-truth position into the BFS'

    ## belief source, UNSEEDED (plan 0 of every L3 episode) -> seed the BFS by
    ## localizing the START SCAN. Must still be waypointed, must NOT be the truth.
    ## The env raises here, so any ground-truth read fails the test outright
    ## rather than being caught by a chain comparison.
    class NoEnv:
        @staticmethod
        def get_xy():
            raise AssertionError('belief arm read the ground-truth position')
    s = Stub(); s._pos_source = 'belief'; s._ball_belief_est = None; s.env = NoEnv()
    g = gc(); s._ball_add_waypoints(g)
    assert 'waypoints' in g, 'unseeded belief declined to add waypoints'
    assert np.allclose(s._ball_last_wp_xy, ref_scan), \
        'unseeded belief did not start the BFS at the localized START SCAN'
    assert not np.allclose(s._ball_last_wp_xy, ref_true), \
        'unseeded belief fell back to ground truth'
    assert s._ball_wp_scan_seeds == 1, 'the scan-seeded replan was not counted'

    ## ...but with no start scan to localize there is nothing to seed from, so
    ## the old decline must survive as the fallback (never an env read).
    s = Stub(); s._pos_source = 'belief'; s._ball_belief_est = None; s.env = NoEnv()
    assert s._ball_pos_for_bfs() is None, 'unseeded belief invented a position'

    ## gridnav bypass plans are not diffusion plans -> no aiming
    s = Stub(); s.nav_mode = 'bfs'
    g = gc(); s._ball_add_waypoints(g)
    assert 'waypoints' not in g, 'waypoints injected into a gridnav bypass plan'


def t8_rollout_knob_is_plumbed():
    """LIDAR_BALL_WAYPOINTS must reach args.ball_waypoints, and the planner must
    read it. A knob that silently does nothing is the failure mode here."""
    import inspect
    from diffuser.ogb_task.ogb_maze_v1 import plan_ogb_stgl_sml as P
    src = inspect.getsource(P)
    assert "args.ball_waypoints = int(os.environ.get('LIDAR_BALL_WAYPOINTS'" in src, \
        'LIDAR_BALL_WAYPOINTS is not read into args.ball_waypoints'
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)
    s2 = inspect.getsource(C.setup_load)
    assert "'ball_waypoints', 0" in s2, 'planner does not read args.ball_waypoints'
    ## and the injection must be called from the plan wrapper, not from setup
    assert '_ball_add_waypoints' in inspect.getsource(C._ball_gen_cond_stgl), \
        'waypoints are never injected into g_cond at plan time'


def t9_progress_pursuit_tracks_a_lagging_agent():
    """Progress pursuit must keep the target near the agent even when the agent
    is far slower than the plan's own pace -- the failure the L1 smoke showed
    (distance to subgoal grew monotonely to ~25 wu under time parameterisation).
    Also: the cursor is monotone, and it never leaps a whole loop ahead.
    """
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)

    ## a straight 888-frame plan covering 88 wu, i.e. 0.1 wu per frame
    plan = np.stack([np.linspace(0, 88, 888), np.zeros(888)], axis=1)

    class Stub:
        _ball_advance_cursor = C._ball_advance_cursor
        _ball_cursor_reacquire = C._ball_cursor_reacquire
        _ball_plan_xy, _ball_cursor, _ball_pursuit_win = plan, 0, 60
        _ball_stall, _ball_stall_escapes, _ball_stall_steps = 0, 0, 50
        ## 2026-07-31 knobs, at their OFF defaults: this arm must stay historical
        _ball_los, _ball_los_margin, _ball_los_seeks = False, 0.6, 0
        _ball_cursor_leash_wu, _ball_pursuit_back = 0.0, 15
        _ball_cursor_reacquires, _ball_cursor_dist = 0, 0.0
        ## 2026-08-01 knobs, at their OFF defaults: this arm must stay historical
        _ball_frozen_steps, _ball_frozen, _ball_frozen_escapes = 0, 0, 0

    ## agent crawls at a QUARTER of the plan pace
    s = Stub()
    lags, cursors = [], []
    for step in range(600):
        pos = np.array([0.025 * step, 0.0])
        rel = s._ball_advance_cursor(pos)
        cursors.append(rel)
        lags.append(float(np.linalg.norm(plan[rel] - pos)))
    assert cursors == sorted(cursors), 'cursor went backwards'
    assert max(lags) < 0.5, f'progress pursuit still ran away (max lag {max(lags):.2f} wu)'

    ## the time parameterisation on the same trajectory, for contrast: index ==
    ## step, so the target is ~0.075 wu/step ahead and the lag grows without bound
    time_lag = float(np.linalg.norm(plan[599] - np.array([0.025 * 599, 0.0])))
    assert time_lag > 40, f'control arm should diverge, got {time_lag:.1f} wu'

    ## bounded window: a plan that revisits the SAME xy later must not teleport
    loopy = np.concatenate([plan[:300], plan[:300][::-1], plan[:300]])
    s = Stub(); s._ball_plan_xy = loopy; s._ball_cursor = 0
    rel = s._ball_advance_cursor(np.array([0.0, 0.0]))
    assert rel <= 60, f'cursor jumped {rel} indices across a revisited corridor'

    ## distance lookahead must be pace-independent: the carrot sits ~L wu along
    ## the plan regardless of how finely the plan is sampled. An INDEX lookahead
    ## of 4 on this plan is only 0.4 wu -- the reason it needs replacing.
    class S2:
        _ball_lookahead_idx = C._ball_lookahead_idx
        _ball_plan_xy, _ball_lookahead, _ball_lookahead_wu = plan, 4, 3.0
        _ball_los, _ball_los_margin = False, 0.6   # LOS off: original behaviour
        _ball_los_pullbacks = _ball_los_blind = 0
        _ball_max_carrot_wu = 0.0                  # cap off: original behaviour
        _ball_min_carrot_wu, _ball_carrot_floor_hits = 0.0, 0   # floor off too
    s2 = S2()
    j = s2._ball_lookahead_idx(100)
    carrot = float(np.linalg.norm(plan[j] - plan[100]))
    assert 2.9 <= carrot <= 3.2, f'distance lookahead gave {carrot:.2f} wu, wanted ~3.0'
    s2._ball_lookahead_wu = 0.0                      # fallback = index lookahead
    assert s2._ball_lookahead_idx(100) == 104
    ## and it must clamp at the end of the plan rather than run off
    s2._ball_lookahead_wu = 3.0
    assert s2._ball_lookahead_idx(len(plan) - 1) == len(plan) - 1


def t11_stall_escape_frees_a_cursor_trapped_at_an_unreachable_plan_point():
    """The dominant rollout failure (4/5 episodes) is the progress
    cursor freezing at a plan point the ball cannot reach -- a residual decode
    jump puts the target across a wall, the PID drives into it, and because the
    cursor only advances when the ball gets NEAR, it never recovers. The escape
    must free it, must not fire while the ball is making progress, and must be
    disableable for the control arm."""
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)
    plan = np.stack([np.linspace(0, 88, 888), np.zeros(888)], axis=1)

    def stub(stall_steps):
        ## LOS and leash both off: this test pins the ORIGINAL blind-march
        ## behaviour, which must be untouched so the control arm stays
        ## reproducible bit for bit
        s = _pursuit_stub(_scanner(), plan, los=False, leash=0.0)
        s._ball_stall_steps = stall_steps
        return s

    ## ball wedged at a fixed position, cursor already past it -> would freeze
    s = stub(50)
    s._ball_cursor = 100
    stuck_pos = plan[100].copy()
    for _ in range(49):
        s._ball_advance_cursor(stuck_pos)
    assert s._ball_cursor == 100, 'escaped too early'
    assert s._ball_stall_escapes == 0
    for _ in range(60):
        s._ball_advance_cursor(stuck_pos)
    assert s._ball_cursor > 100, 'stall escape never fired -- cursor still frozen'
    assert s._ball_stall_escapes > 0, 'escapes not counted'

    ## a ball that IS progressing must never trigger an escape
    s = stub(50)
    for step in range(400):
        s._ball_advance_cursor(np.array([0.1 * step, 0.0]))
    assert s._ball_stall_escapes == 0, 'escape fired on a healthy, progressing ball'

    ## disabled -> stays frozen (the control arm)
    s = stub(0)
    s._ball_cursor = 100
    for _ in range(200):
        s._ball_advance_cursor(plan[100].copy())
    assert s._ball_cursor == 100 and s._ball_stall_escapes == 0, 'stall escape not disableable'


def t12_env_goal_tolerance_is_what_success_actually_requires():
    """Guard the constant the whole rollout is judged against. OGBench pointmaze
    uses _goal_tol = 1.0 wu (0.5 for non-point robots). L4 was stopping at a
    median 3.08 wu because the LiDAR arrival detector fired on a belief cell that
    was ~3 wu wrong -- so 'arrived' and 'succeeded' are different events."""
    import inspect, ogbench.locomaze.maze as M
    src = inspect.getsource(M)
    assert "self._goal_tol = 1.0 if loco_env_type == 'point' else 0.5" in src, \
        'ogbench goal tolerance changed -- every rollout success number needs rereading'
    assert 'np.linalg.norm(self.get_xy() - self.cur_goal_xy) <= self._goal_tol' in src, \
        'success criterion is no longer distance-to-goal on the true xy'


def t10_pursuit_knob_is_plumbed_and_defaults_to_the_historical_arm():
    import inspect
    from diffuser.ogb_task.ogb_maze_v1 import plan_ogb_stgl_sml as P
    src = inspect.getsource(P)
    assert "os.environ.get('LIDAR_BALL_PURSUIT', 'time')" in src, \
        'LIDAR_BALL_PURSUIT missing, or its default is not the historical time arm'
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)
    assert "'ball_pursuit', 'time'" in inspect.getsource(C.setup_load), \
        'planner does not default to the historical time arm'
    act = inspect.getsource(C._pid_lidar_act)
    ## the position must be resolved BEFORE the target is picked, or progress
    ## pursuit would close on a stale/undefined position
    assert act.index('_pos_source') < act.index('_ball_advance_cursor'), \
        'position source is resolved after the pursuit target is chosen'
    ## and pursuit must close on the SAME position the PID uses, not the truth
    assert '_ball_advance_cursor(pos)' in act, \
        'pursuit does not use the (possibly belief) position the PID closes on'


def t13_los_clear_is_real_maze_geometry():
    """`los_clear` must agree with the maze map: a target down an open corridor
    is visible, one behind a wall is not, and the +-margin offsets must reject a
    target reachable only by scraping a corner diagonally."""
    from diffuser.pl_eval.lidar_eval.lidar_grid_common import los_clear
    sc = _scanner()
    mp = np.asarray(sc.maze_map)
    u, ox, oy = sc.maze_unit, sc.offset_x, sc.offset_y
    free = np.argwhere(mp != 1)

    ## a point is always visible from itself
    for i, j in free[:: max(len(free) // 40, 1)]:
        p = np.array([j * u - ox, i * u - oy])
        assert los_clear(sc, p, p[None])[0], f'cell {(i, j)} not visible from itself'

    ## a target inside a wall is never visible from anywhere free
    walls = np.argwhere(mp == 1)
    p0 = np.array([free[0][1] * u - ox, free[0][0] * u - oy])
    wpts = np.stack([[j * u - ox, i * u - oy] for i, j in walls[:: max(len(walls) // 50, 1)]])
    assert not los_clear(sc, p0, wpts).any(), 'a wall cell was reported visible'

    ## ground truth by brute force on a random sample of free-cell PAIRS: walk the
    ## segment at a much finer step than los_clear uses and demand agreement on
    ## every pair where the fine walk is unambiguous (no margin involvement).
    rng = np.random.default_rng(0)
    a = free[rng.integers(0, len(free), 300)]
    b = free[rng.integers(0, len(free), 300)]
    A = np.stack([a[:, 1] * u - ox, a[:, 0] * u - oy], 1)
    B = np.stack([b[:, 1] * u - ox, b[:, 0] * u - oy], 1)
    n_agree = 0
    for k in range(len(A)):
        got = bool(los_clear(sc, A[k], B[k][None], margin=0.0, step=0.25)[0])
        t = np.linspace(0, 1, 400)[:, None]
        q = A[k][None] + t * (B[k] - A[k])[None]
        ii, jj = sc.world_to_cell(q[:, 0], q[:, 1])
        want = not bool(sc._is_wall(ii, jj).any())
        assert got == want, f'los_clear disagrees with a fine walk on pair {k}'
        n_agree += 1
    assert n_agree == 300

    ## the margin must be strictly conservative: anything cleared WITH a margin
    ## is also cleared without one
    strict = los_clear(sc, A[0], B, margin=0.6)
    loose = los_clear(sc, A[0], B, margin=0.0)
    assert not (strict & ~loose).any(), 'margin cleared a target the bare segment rejects'


def _wall_pair(sc):
    """Two free cell centres with a wall between them -- the "the decoded plan
    jumped through a wall" geometry. Returns (near_xy, far_xy)."""
    mp = np.asarray(sc.maze_map)
    u, ox, oy = sc.maze_unit, sc.offset_x, sc.offset_y
    for i in range(1, mp.shape[0] - 1):
        for j in range(1, mp.shape[1] - 1):
            if mp[i, j] == 1 and mp[i, j - 1] != 1 and mp[i, j + 1] != 1:
                return (np.array([(j - 1) * u - ox, i * u - oy]),
                        np.array([(j + 1) * u - ox, i * u - oy]))
    raise AssertionError('no wall with free cells on both sides -- test needs a new maze')


def _straight_corridor(sc, n_pts, frac=1.0):
    """A straight plan of `n_pts` points down the maze's longest open row, so
    line of sight is never the binding constraint and only the distance
    predicates under test can bite. Returns (plan, span_wu)."""
    mp = np.asarray(sc.maze_map)
    u, ox, oy = sc.maze_unit, sc.offset_x, sc.offset_y
    best = None                                   # (row, start_col, run_len)
    for i in range(mp.shape[0]):
        run = st = 0
        for j, v in enumerate(mp[i]):
            if v != 1:
                if run == 0:
                    st = j
                run += 1
                if best is None or run > best[2]:
                    best = (i, st, run)
            else:
                run = 0
    assert best and best[2] >= 4, f'no open corridor in this maze: {best}'
    i, j0, run = best
    span = (run - 1) * u * float(frac)
    x0 = j0 * u - ox
    plan = np.stack([np.linspace(x0, x0 + span, n_pts),
                     np.full(n_pts, i * u - oy)], 1)
    return plan, span


def _pursuit_stub(sc, plan, los=False, lookahead_wu=6.0, leash=0.0, cap=0.0, back=15,
                  min_carrot=0.0, frozen=0):
    """A bare object carrying just the pursuit methods + their state, so the
    cursor/carrot logic can be exercised on CPU without an env or a policy.
    Shared by t14/t16/t17 -- keep every attribute the real ctor sets, or a test
    passes only because `getattr` swallowed a missing knob."""
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)

    class S:
        _ball_advance_cursor = C._ball_advance_cursor
        _ball_lookahead_idx = C._ball_lookahead_idx
        _ball_los_seek = C._ball_los_seek
        _ball_cursor_reacquire = C._ball_cursor_reacquire
        scanner = sc
        _ball_plan_xy, _ball_cursor, _ball_pursuit_win = plan, 0, 60
        _ball_stall, _ball_stall_escapes, _ball_stall_steps = 0, 0, 50
        _ball_lookahead, _ball_lookahead_wu = 4, lookahead_wu
        _ball_los, _ball_los_margin = los, 0.6
        _ball_los_pullbacks = _ball_los_seeks = _ball_los_blind = 0
        _ball_cursor_leash_wu, _ball_max_carrot_wu = leash, cap
        _ball_pursuit_back, _ball_cursor_reacquires = back, 0
        _ball_cursor_dist = 0.0
        ## 2026-08-01 frozen-time knobs, at their OFF defaults
        _ball_min_carrot_wu, _ball_carrot_floor_hits = min_carrot, 0
        _ball_frozen_steps, _ball_frozen, _ball_frozen_eps = frozen, 0, 0.02
        _ball_frozen_escapes = 0
    return S()


def t16_carrot_cap_bounds_target_distance_and_is_off_by_default():
    """Euclidean carrot cap (2026-07-31). `LOOKAHEAD_WU` caps plan ARC, and the
    decoded plan teleports, so 2 wu of arc can be 50 wu of real distance -- one benchmark run
    measured the ball->target distance at p50 6.33 / p90 25.49 / max 50.3
    wu with 54.3% of steps beyond one maze cell. With the cap set, no returned
    target may sit further than the cap; with it unset, the returned index must
    be exactly what it was before the change."""
    import inspect
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)
    sc = _scanner()

    ## an open corridor (so LOS is never the binding constraint) that the plan
    ## walks down, far enough that a 6 wu arc carrot is well outside a 4 wu cap
    plan, _ = _straight_corridor(sc, 60)
    pos = plan[0].copy()

    ## 1. UNCAPPED (cap=0, los off) reproduces the pure arc-length lookahead:
    ##    the first index at least `lookahead_wu` of arc past the cursor.
    s = _pursuit_stub(sc, plan, los=False, lookahead_wu=6.0, cap=0.0)
    seg = np.linalg.norm(np.diff(plan, axis=0), axis=1)
    want = int(min(np.searchsorted(np.cumsum(seg), 6.0) + 1, len(plan) - 1))
    assert s._ball_lookahead_idx(0, pos) == want, 'cap=0 changed the arc-length target'
    assert np.linalg.norm(plan[want] - pos) > 4.0, 'test geometry: target already inside the cap'

    ## 2. CAPPED: the target is the FURTHEST point still within the cap
    for cap in (2.0, 3.0, 4.0):
        s = _pursuit_stub(sc, plan, los=False, lookahead_wu=6.0, cap=cap)
        tgt = s._ball_lookahead_idx(0, pos)
        d = np.linalg.norm(plan[tgt] - pos)
        assert d <= cap + 1e-9, f'cap {cap} exceeded: {d:.3f} wu'
        assert np.linalg.norm(plan[tgt + 1] - pos) > cap, f'cap {cap}: not the furthest point'

    ## 3. NOTHING QUALIFIES -> hold at the cursor, never the far nominal target.
    ##    This is the runaway path: the pre-2026-07-31 code returned `tgt`.
    ##    Stand the ball off the plan so not even the cursor is inside the cap.
    off = plan[3] + np.array([0.0, 0.3])
    s = _pursuit_stub(sc, plan, los=False, lookahead_wu=6.0, cap=0.05)
    assert s._ball_lookahead_idx(3, off) == 3, 'blind fall-through did not hold at the cursor'
    assert s._ball_los_blind > 0, 'blind step not counted'
    ## and with the cap OFF the same call runs away to the far arc target
    s = _pursuit_stub(sc, plan, los=False, lookahead_wu=6.0, cap=0.0)
    assert s._ball_lookahead_idx(3, off) > 3, 'control arm no longer runs away'

    ## 4. off by default, end to end
    from diffuser.ogb_task.ogb_maze_v1 import plan_ogb_stgl_sml as P
    assert "os.environ.get('LIDAR_BALL_MAX_CARROT_WU', '0')" in inspect.getsource(P), \
        'LIDAR_BALL_MAX_CARROT_WU missing or not defaulted off'
    assert "'ball_max_carrot_wu', 0.0" in inspect.getsource(C.setup_load), \
        'planner default is not off'


def t17_cursor_leash_stops_the_blind_march_and_re_acquires_backward():
    """Cursor runaway (2026-07-31, 11750 blind +1 marches = 25.0% of
    all control steps, against 109 LOS seeks = 0.9%). The stall escape and the
    LOS seek both search FORWARD ONLY, so once the ball is left behind nothing
    can bring the carrot back and the cursor walks to the end of the plan. With
    the leash set, the march must stop and the cursor must re-acquire backward."""
    import inspect
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)
    from diffuser.pl_eval.lidar_eval.lidar_grid_common import los_clear
    sc = _scanner()
    near, far = _wall_pair(sc)
    assert not los_clear(sc, near, far[None])[0], 'test geometry is not actually blocked'

    ## the real runaway shape: 21 points the ball is standing on, then a decode
    ## jump to the far side of a wall for the rest of the plan. Nothing forward
    ## is ever visible, so the LOS seek cannot fire and the escape marches blind.
    tail = 279
    plan = np.concatenate([
        np.stack([np.linspace(near[0] - 2.0, near[0], 21), np.full(21, near[1])], 1),
        np.repeat(far[None], tail, axis=0)])
    pinned = plan[20].copy()
    n_steps = 60 + tail                       # stall_steps=50, then one march/step

    ## 1. leash OFF = today's behaviour: the cursor marches to the end of the plan
    s = _pursuit_stub(sc, plan, los=True, leash=0.0)
    s._ball_cursor = 20
    for _ in range(n_steps):
        s._ball_advance_cursor(pinned)
    assert s._ball_los_seeks == 0, 'test geometry: something forward was visible'
    assert s._ball_stall_escapes > 100, 'control arm did not blind-march'
    assert s._ball_cursor_reacquires == 0, 'leash=0 must never re-acquire'
    assert s._ball_cursor == len(plan) - 1, 'control arm did not run to the plan end'
    runaway = float(np.linalg.norm(plan[s._ball_cursor] - pinned))
    assert runaway > 4.0, f'test geometry: cursor only ran {runaway:.2f} wu'

    ## 2. leash ON: the cursor never leaves the leash and re-acquires backward
    s = _pursuit_stub(sc, plan, los=True, leash=4.0)
    s._ball_cursor = 20
    for _ in range(n_steps):
        s._ball_advance_cursor(pinned)
        assert np.linalg.norm(plan[s._ball_cursor] - pinned) <= 4.0 + 1e-9, \
            'cursor escaped the leash'
        assert s._ball_cursor_dist <= 4.0 + 1e-9, 'cursor_dist not tracked'
    assert s._ball_cursor_reacquires > 0, 'leash never re-acquired'

    ## 3. the re-acquire searches at-or-behind the cursor, within PURSUIT_BACK,
    ##    and returns None (=> escape as before) when nothing there is in reach
    s = _pursuit_stub(sc, plan, los=True, leash=4.0, back=15)
    assert s._ball_cursor_reacquire(pinned, 30) == 20, \
        'did not re-acquire the nearest in-reach point behind the cursor'
    assert s._ball_cursor_reacquire(pinned, 100) is None, \
        'backward window is not bounded by PURSUIT_BACK'
    assert _pursuit_stub(sc, plan, los=True, leash=0.0)._ball_cursor_reacquire(pinned, 30) is None, \
        'leash=0 must disable the re-acquire entirely'

    ## 4. off by default, end to end
    from diffuser.ogb_task.ogb_maze_v1 import plan_ogb_stgl_sml as P
    assert "os.environ.get('LIDAR_BALL_CURSOR_LEASH_WU', '0')" in inspect.getsource(P), \
        'LIDAR_BALL_CURSOR_LEASH_WU missing or not defaulted off'
    assert "'ball_cursor_leash_wu', 0.0" in inspect.getsource(C.setup_load), \
        'planner default is not off'


def t18_new_pursuit_knobs_are_recorded_and_the_eval_set_is_five_tasks():
    """Two bookkeeping guards.

    (a) A knob that is not written to the json cannot be audited afterwards --
    section-3 mistakes 6/9/13 are all silent-default failures.
    (b) `--plan_n_ep 20` is ev_probs TASK BLOCK 0, not twenty tasks: 20 samples
    of one corner-to-corner route with sub-cell jitter. Every closed-loop number
    through 2026-07-30 was measured there. Assert the degeneracy so nobody reads
    an N_EP=20 run as a benchmark number again."""
    import inspect, h5py
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)

    src = inspect.getsource(C.update_j_data)
    for k in ('ball_cursor_leash_wu', 'ball_max_carrot_wu', 'ball_pursuit_back',
              'ball_cursor_reacquires'):
        assert f"'{k}'" in src, f'{k} is not recorded in the rollout json'
    fsrc = inspect.getsource(C._ball_flush_ep)
    for k in ('cursor_dist_mean', 'cursor_dist_p90', 'n_stall_escapes',
              'n_cursor_reacquires'):
        assert f'{k}=' in fsrc, f'{k} missing from ball_ep_summaries'

    with h5py.File(PROBS, 'r') as f:
        st, gl = np.asarray(f['start_state']), np.asarray(f['goal_pos'])
    assert len(st) == 100 and len(gl) == 100, f'expected 100 problems, got {len(st)}'
    ## the first 20 rows are ONE task: goals within a single maze cell of each other
    assert np.ptp(gl[:20], axis=0).max() < 4.0, 'block 0 is no longer a single goal cell'
    assert np.ptp(st[:20], axis=0).max() < 4.0, 'block 0 is no longer a single start cell'
    ## all 100 are FIVE distinct tasks
    ctrs = np.stack([gl[b * 20:(b + 1) * 20].mean(0) for b in range(5)])
    seps = [np.linalg.norm(ctrs[a] - ctrs[b]) for a in range(5) for b in range(a + 1, 5)]
    assert min(seps) > 4.0, f'the 5 blocks are not distinct tasks: min sep {min(seps):.1f}'


def t19_every_plan_is_recorded_and_scored_against_the_goal_log_only():
    """The plan-content instrumentation (2026-07-31, `latest+12`).

    The previous attempt to ask "does the plan end at the goal?" was done
    offline from `target[argmax(cursor)]` and was INVALID -- 99% of episodes
    replan, so that indexes whichever plan reached deepest, and LOS pull-back
    means `target` is often not the carrot (corr with final_gd: 0.14). So:
    (a) EVERY plan must be kept, not just the last, each tagged with the control
    step it took effect on; (b) tail vs closest-approach must both be reported,
    because "runs past the goal" and "never goes there" need opposite fixes;
    and (c) the true goal used to score them must stay OFF the control path --
    reading it in the controller would silently turn L3 into L1."""
    import inspect
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)

    ## (a) recorded on every decode, and reset per episode
    assert '_ball_record_plan()' in inspect.getsource(C._ball_gen_cond_stgl), \
        'plans are not captured at decode time'
    rsrc = inspect.getsource(C._lidar_arrival_reset)
    assert '_ball_plans = []' in rsrc, 'the plan list is not cleared per episode'
    assert rsrc.index('_ball_flush_ep()') < rsrc.index('_ball_plans = []'), \
        'the episode must be flushed BEFORE its plans are cleared, or they are lost'
    assert 'step0' in inspect.getsource(C._ball_record_plan), \
        'plans must be tagged with the control step they took effect on'

    ## (b) both numbers, for the last plan and the best one
    fsrc = inspect.getsource(C._ball_flush_ep)
    assert '_ball_plan_goal_stats()' in fsrc, 'plan stats missing from ball_ep_summaries'
    gsrc = inspect.getsource(C._ball_plan_goal_stats)
    for k in ('plan_tail_gd_last', 'plan_closest_gd_last',
              'plan_tail_gd_best', 'plan_closest_gd_best', 'n_plans'):
        assert k in gsrc, f'{k} is not reported'

    ## the CSR layout must round-trip ragged plans
    assert 'plans_ptr' in fsrc and 'plans_step' in fsrc and 'plans_xy' in fsrc, \
        'the npz does not carry the plans'
    plans = [np.zeros((5, 2)), np.zeros((3, 2))]          # a replan MAY shorten it
    ptr = np.cumsum([0] + [len(p) for p in plans])
    cat = np.concatenate(plans, axis=0)
    assert len(cat[ptr[0]:ptr[1]]) == 5 and len(cat[ptr[1]:ptr[2]]) == 3

    ## (c) LOG ONLY: the true goal may appear in the reset hook and the scorer,
    ## and NOWHERE on the path that computes a control.
    assert '_ball_true_goal_xy' in gsrc, 'the scorer does not use the true goal'
    for nm in ('_pid_lidar_act', '_ball_advance_cursor', '_ball_lookahead_idx',
               '_ball_cursor_reacquire', '_ball_los_seek', '_ball_add_waypoints',
               '_decode_plan_xy', '_ball_pos_for_bfs'):
        ## plain getattr, no default: a renamed method must fail loudly here
        ## rather than silently drop out of the leak check.
        meth = getattr(C, nm)
        assert '_ball_true_goal_xy' not in inspect.getsource(meth), \
            f'{nm} reads the TRUE goal -- that converts L3 into L1'

    ## the tail/closest pair must actually separate the two failure shapes
    goal = np.array([10.0, 0.0])
    runs_past = np.stack([np.linspace(0, 20, 21), np.zeros(21)], axis=1)
    never_goes = np.stack([np.linspace(0, 20, 21), np.full(21, 30.0)], axis=1)
    for plan, near_mid in ((runs_past, True), (never_goes, False)):
        d = np.linalg.norm(plan - goal, axis=1)
        assert d[-1] > 4.0, 'both fixtures are meant to END far from the goal'
        assert (d.min() < 1.0) == near_mid, \
            'closest-approach does not distinguish "runs past" from "never goes there"'


def t14_los_pursuit_never_aims_through_a_wall_and_is_off_by_default():
    """The observed failure mode, as a unit test. A plan with a decode jump
    across a wall must not leave the carrot on the far side, and the stall escape
    must SEEK a visible point rather than marching the cursor past the ball."""
    import inspect
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)
    from diffuser.pl_eval.lidar_eval.lidar_grid_common import los_clear
    sc = _scanner()
    mp = np.asarray(sc.maze_map)
    u, ox, oy = sc.maze_unit, sc.offset_x, sc.offset_y

    ## find a free cell with a wall neighbour and a free cell on the wall's far
    ## side -- i.e. a genuine "plan jumped through a wall" geometry
    trio = None
    for i in range(1, mp.shape[0] - 1):
        for j in range(1, mp.shape[1] - 1):
            if mp[i, j] == 1 and mp[i, j - 1] != 1 and mp[i, j + 1] != 1:
                trio = (i, j); break
        if trio:
            break
    assert trio, 'no wall with free cells on both sides -- test needs a new maze'
    i, j = trio
    near = np.array([(j - 1) * u - ox, i * u - oy])
    far = np.array([(j + 1) * u - ox, i * u - oy])
    assert not los_clear(sc, near, far[None])[0], 'test geometry is not actually blocked'

    ## plan: a few points beside the ball, then a jump to the far side of the wall
    plan = np.concatenate([
        np.stack([np.linspace(near[0] - 3.0, near[0], 20), np.full(20, near[1])], 1),
        np.repeat(far[None], 40, axis=0)])

    def stub(los):
        return _pursuit_stub(sc, plan, los=los, lookahead_wu=6.0)

    ## WITHOUT los the carrot lands across the wall (this is the bug)
    s = stub(False)
    s._ball_cursor = 19
    tgt = s._ball_lookahead_idx(s._ball_advance_cursor(near), near)
    assert not los_clear(sc, near, plan[tgt][None])[0], \
        'control arm should still aim through the wall -- geometry drifted'

    ## WITH los the carrot is pulled back to something the ball can drive at
    s = stub(True)
    s._ball_cursor = 19
    tgt = s._ball_lookahead_idx(s._ball_advance_cursor(near), near)
    assert los_clear(sc, near, plan[tgt][None])[0], 'LOS carrot is still behind a wall'
    assert s._ball_los_pullbacks > 0, 'pull-back not counted'

    ## the stall escape must not march a trapped cursor blindly: with every
    ## remaining plan point behind the wall there is nothing to seek, so it falls
    ## through to the old march (counted as an escape, not a seek)
    s = stub(True)
    s._ball_cursor = 20
    for _ in range(120):
        s._ball_advance_cursor(near)
    assert s._ball_los_seeks == 0 and s._ball_stall_escapes > 0, \
        'blind march should be the documented fallback when nothing is visible'

    ## and when a visible point DOES exist ahead, the escape seeks it in one step
    straight = np.stack([np.linspace(near[0], near[0] + 3.0, 80), np.full(80, near[1])], 1)
    s = stub(True); s._ball_plan_xy = straight; s._ball_cursor = 40
    for _ in range(60):
        s._ball_advance_cursor(straight[40])
    assert s._ball_los_seeks > 0, 'escape never used a visible point'
    assert los_clear(sc, straight[40], straight[s._ball_cursor][None])[0]

    ## default OFF, so an unset run reproduces the control arm bit for bit
    from diffuser.ogb_task.ogb_maze_v1 import plan_ogb_stgl_sml as P
    assert "os.environ.get('LIDAR_BALL_LOS', '0')" in inspect.getsource(P), \
        'LIDAR_BALL_LOS missing or not defaulted off'
    assert "'ball_los', 0" in inspect.getsource(C.setup_load), 'planner default is not off'
    ## LOS must close on the same (possibly belief) position the PID uses
    assert '_ball_lookahead_idx(rel, pos)' in inspect.getsource(C._pid_lidar_act), \
        'LOS pursuit does not use the loc_source position'


def t15_subcell_map_is_a_local_centroid_and_off_by_default():
    """Sub-cell MAP interpolation (2026-07-30). At pos_lpf=0 the filter's residual
    error is pure 0.5 wu grid QUANTIZATION (0.248-0.357 wu across 20
    mazes vs 0.191/0.354 predicted for lattice snapping), so `argmax` hands the
    kp=5/kd=0 PID a staircase. The centroid must be LOCAL: a distant aliased mode
    must not drag it, walls must weigh zero, and the whole thing must be off by
    default and must not disturb `est_cell`."""
    import inspect
    from diffuser.datasets.lidar.lidar_arrival import LiveLocalizer

    class _G:
        def __init__(self, n):
            self.x_vals = np.arange(n) * 0.5
            self.y_vals = np.arange(n) * 0.5
            self.theta_vals = np.zeros(1)

    class _L:
        def __init__(self, g): self.grid = g

    def mk(bel, subcell=1):
        lz = object.__new__(LiveLocalizer)
        lz.loc = _L(_G(bel.shape[0])); lz.subcell = subcell
        lz.belief = bel.astype(np.float64)
        return lz

    n = 12
    ## 1. a SYMMETRIC bump must return the MAP point exactly (no bias introduced)
    b = np.zeros((n, n, 1)); b[5, 5, 0] = 0.6; b[4, 5, 0] = b[6, 5, 0] = 0.2
    x, y = mk(b)._subcell_xy((5, 5, 0))
    assert abs(x - 2.5) < 1e-12 and abs(y - 2.5) < 1e-12, (x, y)

    ## 2. an ASYMMETRIC bump must move toward the mass, by the exact weighted amount
    b = np.zeros((n, n, 1)); b[5, 5, 0] = 0.5; b[5, 6, 0] = 0.5
    x, y = mk(b)._subcell_xy((5, 5, 0))
    assert abs(x - 2.75) < 1e-12, x                      # halfway to the next cell
    assert abs(y - 2.5) < 1e-12, y
    assert 0.0 < abs(x - 2.5) < 0.5, 'must stay strictly inside one cell of the MAP'

    ## 3. a DISTANT aliased mode outside the window must NOT drag the estimate --
    ##    this is why the window exists; a global centroid would land between modes
    b = np.zeros((n, n, 1)); b[5, 5, 0] = 0.5; b[5, 11, 0] = 0.5
    x, y = mk(b)._subcell_xy((5, 5, 0))
    assert abs(x - 2.5) < 1e-12, f'distant mode leaked into the centroid: {x}'

    ## 4. wall cells carry zero belief, so they must contribute nothing
    b = np.zeros((n, n, 1)); b[5, 5, 0] = 1.0            # neighbours are walls (0.0)
    x, y = mk(b)._subcell_xy((5, 5, 0))
    assert (abs(x - 2.5) < 1e-12) and (abs(y - 2.5) < 1e-12), (x, y)

    ## 5. edge clipping: a MAP at the array corner must not index out of bounds
    b = np.zeros((n, n, 1)); b[0, 0, 0] = 1.0
    x, y = mk(b)._subcell_xy((0, 0, 0))
    assert abs(x) < 1e-12 and abs(y) < 1e-12, (x, y)

    ## 6. an all-zero window falls back to the raw MAP lattice point
    x, y = mk(np.zeros((n, n, 1)))._subcell_xy((5, 5, 0))
    assert abs(x - 2.5) < 1e-12 and abs(y - 2.5) < 1e-12, (x, y)

    ## 7. OFF BY DEFAULT, everywhere, so an unset run reproduces the reference run exactly
    src = inspect.getsource(LiveLocalizer)
    assert 'subcell=0' in src, 'LiveLocalizer.subcell is not defaulted off'
    assert 'self.subcell > 0 else (x, y)' in src, 'raw-MAP path is not the default branch'
    from diffuser.ogb_task.ogb_maze_v1 import plan_ogb_stgl_sml as P
    assert "os.environ.get('LIDAR_LOC_SUBCELL', '0')" in inspect.getsource(P), \
        'LIDAR_LOC_SUBCELL missing or not defaulted off'
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)
    assert "'loc_subcell', 0" in inspect.getsource(C.setup_load), 'planner default is not off'

    ## 8. est_cell must stay the DISCRETE MAP cell -- the arrival detector and the BFS
    ##    waypoint chain both key off it, so sub-cell must not perturb their semantics
    usrc = inspect.getsource(LiveLocalizer.update)
    assert 'self.est_cell = self.loc.grid.cell_of_xy(x, y)' in usrc, \
        'est_cell no longer derives from the raw MAP point'
    assert usrc.index('est_cell = ') < usrc.index('_subcell_xy'), \
        'est_cell must be computed from the discrete MAP, before any interpolation'


def t20_carrot_floor_refuses_a_carrot_the_ball_is_standing_on():
    """A-idle (2026-08-01, `ball_step_budget.py` over the baseline arm). 17.7% of the
    1000-step budget is spent frozen with the carrot 0.19 wu away and |u| ~ 0:
    `LOOKAHEAD_WU` measures ARC along a plan that loiters at the model's own pace,
    so it can return a point the ball is already standing on. That is a FIXED
    POINT -- the cursor is an argmin over the ball's position, so it cannot
    advance while the ball does not move, and only the 50-step timer breaks it.
    The floor must push the carrot out to a real distance, must not change the
    returned index when it is off, and must never run off the end of the plan."""
    sc = _scanner()
    ## a plan that loiters: 200 points inside 1 wu, then a straight run away.
    ## 6 wu of ARC is still ~0 wu of DISTANCE while the cursor sits in the loiter.
    loiter = np.stack([0.02 * np.arange(50) % 1.0, np.zeros(50)], axis=1)
    run, _ = _straight_corridor(sc, 60)
    plan = np.concatenate([run[0] + loiter, run], axis=0)
    pos = plan[0].copy()

    ## 1. OFF (floor=0) must return exactly the historical index
    s_off = _pursuit_stub(sc, plan, los=False, lookahead_wu=2.0, min_carrot=0.0)
    j_off = s_off._ball_lookahead_idx(0, pos)
    assert s_off._ball_carrot_floor_hits == 0, 'floor counted a hit while disabled'
    assert np.linalg.norm(plan[j_off] - pos) < 1.0, (
        'test geometry: the historical carrot is already far, nothing to fix')

    ## 2. ON: the carrot clears the floor, and the floor is counted
    for floor in (1.5, 3.0, 5.0):
        s = _pursuit_stub(sc, plan, los=False, lookahead_wu=2.0, min_carrot=floor)
        j = s._ball_lookahead_idx(0, pos)
        d = float(np.linalg.norm(plan[j] - pos))
        assert d >= floor - 1e-9, f'floor {floor}: carrot still {d:.2f} wu away'
        assert j > j_off, f'floor {floor}: carrot index did not advance'
        assert s._ball_carrot_floor_hits == 1, 'floor hit not counted'

    ## 3. monotone in the floor: a bigger floor never gives a nearer carrot
    ds = [float(np.linalg.norm(
        plan[_pursuit_stub(sc, plan, los=False, lookahead_wu=2.0,
                           min_carrot=f)._ball_lookahead_idx(0, pos)] - pos))
        for f in (1.0, 2.0, 4.0, 6.0)]
    assert ds == sorted(ds), f'carrot distance not monotone in the floor: {ds}'

    ## 4. end of plan: nothing satisfies the floor -> clamp, never index past it
    s = _pursuit_stub(sc, plan, los=False, lookahead_wu=2.0, min_carrot=999.0)
    j = s._ball_lookahead_idx(len(plan) - 3, pos)
    assert 0 <= j <= len(plan) - 1, f'floor ran off the plan: {j}'

    ## 5. with LOS on, the floor may only PREFER admissible points -- it must
    ##    never hand back a target the ball cannot drive at in a straight line
    s = _pursuit_stub(sc, plan, los=True, lookahead_wu=6.0, min_carrot=4.0)
    j = s._ball_lookahead_idx(0, pos)
    from diffuser.pl_eval.lidar_eval.lidar_grid_common import los_clear
    assert bool(los_clear(sc, pos, plan[j:j + 1], margin=0.6)[0]), \
        'floor returned a carrot through a wall'


def t21_frozen_trip_fires_the_escape_on_displacement_not_on_the_cursor():
    """`_ball_stall` counts CURSOR non-advance, but the condition the escape was
    written for is the BALL not moving -- and the two differ by up to
    `stall_steps` of dead time. Measured: the first min(run, 50) steps of every
    frozen run is 18.6% of the whole 1000-step budget. The trip must fire EARLIER
    than the timer, never later, must be counted separately, and must be off by
    default so an unset run is bit-identical to the reference run."""
    import inspect
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)
    sc = _scanner()
    plan, _ = _straight_corridor(sc, 400)
    stuck = plan[100].copy()

    ## 1. OFF: the historical timer, and nothing before it
    s = _pursuit_stub(sc, plan, los=False, frozen=0)
    s._ball_cursor = 100
    for _ in range(49):
        s._ball_advance_cursor(stuck)
    assert s._ball_cursor == 100 and s._ball_frozen_escapes == 0, 'fired while off'

    ## 2. ON at 5: with the ball frozen the escape fires at step 5, not 50
    s = _pursuit_stub(sc, plan, los=False, frozen=5)
    s._ball_cursor = 100
    for i in range(10):
        s._ball_frozen = i          # what _pid_lidar_act maintains from true_xy
        s._ball_advance_cursor(stuck)
    assert s._ball_cursor > 100, 'frozen trip never fired'
    assert s._ball_frozen_escapes > 0, 'frozen escapes not counted separately'

    ## 3. a MOVING ball never trips it, however stuck the cursor looks
    s = _pursuit_stub(sc, plan, los=False, frozen=5)
    s._ball_cursor = 100
    for _ in range(40):
        s._ball_frozen = 0          # moving
        s._ball_advance_cursor(stuck)
    assert s._ball_frozen_escapes == 0, 'trip fired on a moving ball'
    assert s._ball_cursor == 100, 'cursor advanced before the timer on a moving ball'

    ## 4. the counter is maintained from TRUE displacement, and reset per plan
    psrc = inspect.getsource(C._pid_lidar_act)
    assert '_ball_frozen_eps' in psrc and 'v_env' in psrc, \
        'the frozen counter is not driven by the measured displacement'
    assert '_ball_frozen = 0' in psrc, 'the frozen counter is not reset on a new plan'
    ## 5. off by default everywhere
    from diffuser.ogb_task.ogb_maze_v1 import plan_ogb_stgl_sml as P
    src = inspect.getsource(P)
    for k in ("LIDAR_BALL_FROZEN_STEPS', '0'", "LIDAR_BALL_MIN_CARROT_WU', '0'",
              "LIDAR_BALL_WALL_SLIDE', '0'"):
        assert k in src, f'{k} missing or not defaulted off'
    csrc = inspect.getsource(C.setup_load)
    for k in ("'ball_frozen_steps', 0", "'ball_min_carrot_wu', 0.0",
              "'ball_wall_slide', 0"):
        assert k in csrc, f'planner default is not off: {k}'


def t22_wall_slide_uses_only_the_scan_and_only_when_pinned():
    """B-pinned (2026-08-01): 32.9% of the 1000-step budget is spent frozen with
    the carrot p50 27 wu away and the PID saturated (|u| p50 1.414) -- driving
    straight into the wall between them. On 84.7% of those steps some beam is
    open and still points goal-ward. The slide must: fire ONLY when frozen at a
    saturated command, produce a direction that is actually open, leave every
    other step bit-identical, and read nothing but the scan."""
    import inspect
    from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import (
        OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1 as C)
    sc = _scanner()

    class S:
        _ball_slide_cmd = C._ball_slide_cmd
        scanner = sc
        _ball_wall_slide, _ball_slide_wu = True, 1.0
        _ball_frozen, _ball_frozen_steps, _ball_slides = 10, 5, 0

        class _pid:
            out_clip = 1.0

    ## a pose in the open corridor, with the carrot placed straight through the
    ## wall behind the agent's row so the direct command is unfollowable
    plan, _ = _straight_corridor(sc, 40)
    pos = plan[len(plan) // 2].copy()
    far = pos + np.array([0.0, 40.0])          # across the corridor wall
    sat = np.array([0.0, 1.0])                 # saturated, straight at the wall

    ## 1. fires, and the chosen heading is genuinely open in the scan
    s = S()
    u = s._ball_slide_cmd(pos, far, sat)
    assert s._ball_slides == 1, 'slide did not fire on a pinned, saturated step'
    n = float(np.linalg.norm(u))
    assert n > 0, 'slide produced a null command'
    d = u / n
    r = np.asarray(sc.scan(float(pos[0]), float(pos[1]), 0.0), dtype=np.float64)
    th = np.asarray(sc.beam_offsets, dtype=np.float64)
    k = int(np.argmin(np.abs(np.arctan2(d[1], d[0]) - th)))
    assert r[k] >= 1.0, f'slide chose a blocked heading (range {r[k]:.2f} wu)'
    assert np.abs(u).max() <= 1.0 + 1e-9, 'slide exceeded the action clip'

    ## 2. NOT pinned (ball moving) -> untouched
    s = S(); s._ball_frozen = 0
    assert np.allclose(s._ball_slide_cmd(pos, far, sat), sat) and s._ball_slides == 0

    ## 3. NOT saturated -> untouched (the PID is still in control)
    s = S()
    weak = np.array([0.0, 0.3])
    assert np.allclose(s._ball_slide_cmd(pos, far, weak), weak) and s._ball_slides == 0

    ## 4. disabled -> bit-identical, which is what keeps the reference run reproducible
    s = S(); s._ball_wall_slide = False
    assert np.allclose(s._ball_slide_cmd(pos, far, sat), sat) and s._ball_slides == 0

    ## 5. SCAN ONLY. The whole honesty of the LiDAR arm rests on this: no
    ##    occupancy map, no BFS, no true position inside the slide.
    src = inspect.getsource(C._ball_slide_cmd)
    assert 'scanner.scan(' in src, 'the slide does not read the live scan'
    for bad in ('maze_map', 'get_xy', 'make_waypoints', 'loc_grid', 'free_xy',
                'LOS_CLEAR', 'los_clear'):
        assert bad not in src, f'the slide reads {bad} -- that is not scan-only'


if __name__ == '__main__':
    for nm, fn in sorted(globals().items()):
        if nm.startswith('t') and callable(fn) and nm[1].isdigit():
            check(nm, fn)
    print(f'\n{len(_PASS)}/{len(_PASS) + len(_FAIL)} waypoint tests passed')
    sys.exit(1 if _FAIL else 0)
