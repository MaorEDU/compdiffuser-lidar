"""
BALL (pointmaze) LiDAR closed-loop rollout planner -- BALL.md Phases 5-6.

Subclass of the ant LiDAR planner (``ogb_stgl_sml_lidar_planner_v1``) for the
OGBench point-mass agent. What changes vs. the ant:

  * NO heading. The ball's obs is 2-D (x, y); every scan is world-frame
    (yaw == 0) -- ``_obs_xy_yaw`` / ``_nav_proprio_yaw`` are overridden, and the
    gridnav route synthesizer is wrapped so route/probe scans are re-rendered
    at yaw=0 (matching the training scans exactly).
  * NO learned executor needed (Arm A, default): the plan's waypoint SCANS are
    decoded to (x, y) targets with the grid localizer (sequence-consistent
    decode, once per replan) and tracked by a PID law (``pid_controller.PID2D``;
    Kp=5/Kd=0/Ki=0 == the upstream pointmaze P-controller). A learned
    inverse-dynamics executor (Arm B) is used instead iff
    ``LIDAR_INV_MODEL_PATH`` points at a trained xyv_lidar inv-dyn run.
  * Position source is a LADDER KNOB (``LIDAR_LOC_SOURCE``): 'env' feeds the
    PID the true position (rungs L1/L2 -- isolates planner + goal-loc error);
    'belief' feeds it the live histogram-filter estimate (rung L3 -- the full
    True-LiDAR loop; position error becomes part of the measurement).
  * 4-D state [x, y, vx, vy]: velocities are finite differences of consecutive
    observations (v=0 at episode start), matching the dataset convention in
    ``ogb_lidar_utils.finite_diff_velocities``.
  * Deadlock (zigzag) detector per the design notes: net displacement < D over a
    W-step window while the commanded action stays large -> flag the episode,
    dump an artifact pack, and (policy) pause + consult the team. Detection only;
    it never fakes success or ends the run.

Everything else (goal scan/band synthesis, localization re-rank, arrival
detector + verify-and-hop, gridnav BFS navigator, periodic replanning, video +
json emission) is inherited from the ant planner unchanged -- those modules are
already env-agnostic and the maze geometry resolves from the env name.

Eval-ladder cheat sheet (see BALL.md §10):
    L1: LIDAR_NAV_GOAL_MODE unused, LIDAR_LOC_SOURCE=env,   goal scan from true pos (default path)
    L2: LIDAR_ARRIVAL_GOAL_MODE=lidar / rerank 'lidar' modes, LIDAR_LOC_SOURCE=env
    L3: LIDAR_LOC_SOURCE=belief  (+ arrival on, ends run)
    L4: LIDAR_NAV_MODE=bfs (planner-free grid navigator reference)
"""

import os
import numpy as np
from types import SimpleNamespace

import diffuser.utils as utils
from diffuser.utils import load_config
from diffuser.datasets.lidar.pid_controller import PID2D
from diffuser.datasets.lidar.lidar_arrival import LiveLocalizer
from diffuser.pl_eval.lidar_eval.lidar_grid_common import make_waypoints
from diffuser.pl_eval.lidar_eval.lidar_grid_common import los_clear as LOS_CLEAR
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_planner_v1 import (
    OgB_Stgl_Sml_Lidar_MazeEnvPlanner_V1,
)

## leading [x, y, vx, vy] channels in the ball inv-dyn obs ('xyv_lidar')
N_XYV = 4


class OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1(OgB_Stgl_Sml_Lidar_MazeEnvPlanner_V1):
    """Closed-loop rollout for the LiDAR pointmaze (ball) planner."""

    def __init__(self, args_train, args):
        super().__init__(args_train, args=args)
        assert 'pointmaze' in args_train.dataset, args_train.dataset
        ## Arm B (learned inv-dyn executor) iff an inv run is given; else Arm A
        ## (decode + PID). The base __init__ set act_control='pd_ogb' for
        ## pointmaze -- that branch expects a 2-D (x, y) plan and MUST NOT run
        ## against 30-D scan waypoints.
        self._ball_arm_b = bool(getattr(args, 'inv_model_path', '') or '')
        self.act_control = 'pred_inv' if self._ball_arm_b else 'pid_lidar'
        utils.print_color(
            f"[ball] executor arm = {'B (learned inv-dyn)' if self._ball_arm_b else 'A (decode + PID)'}",
            c='g')

    # ================= setup =================
    def setup_load(self, ld_config):
        super().setup_load(ld_config)
        args = self.args

        ## the decode grid/localizer is ALWAYS needed (waypoint scans -> xy),
        ## not only when rerank/nav/arrival turned it on.
        self._ensure_grid_loc()
        assert self.loc_grid.n_theta == 1, (
            f'ball grid must be built with n_theta=1 (got {self.loc_grid.n_theta}); '
            f'set LIDAR_LOC_NTHETA=1')

        ## ---- position source (the L1/L2 vs L3 ladder knob) ----
        self._pos_source = str(getattr(args, 'loc_source', 'env')).lower()
        assert self._pos_source in ('env', 'belief'), self._pos_source
        ## live belief tracker for 'belief' (reuses the gridnav one if running)
        ## subcell>0 reports the belief-weighted centroid of the +/-N cell window around
        ## the MAP instead of the raw lattice point. The filter's residual error at
        ## pos_lpf=0 is pure 0.5 wu QUANTIZATION (0.248-0.357 wu across 20
        ## mazes, vs 0.191/0.354 predicted for lattice snapping), and argmax hands the
        ## kp=5/kd=0 PID a staircase. Default 0 == bit-identical to 45831.
        self._ball_subcell = int(getattr(args, 'loc_subcell', 0))
        self._ball_live = LiveLocalizer(
            self.loc,
            motion_sigma_cells=float(getattr(args, 'loc_motion_sigma', 1.0)),
            theta_blur=0, subcell=self._ball_subcell)
        self._ball_belief_est = None      # (x, y) MAP estimate, refreshed per step
        self._ball_belief_conf = 0.0

        ## ---- PID controller (Arm A) ----
        ## !! THE 0.7 BELIEF DEFAULT BELOW IS CONVICTED AS HARMFUL (2026-07-30). !!
        ## It was meant to damp belief jitter, and instead it is the single cause of
        ## every "L3 localization failure" measured to date. Paired A/B at L3, seed 0,
        ## sigma=1.0, same 20 tasks, ONLY this knob different:
        ##   lpf=0.7: 4/20 success, 8/20 episodes lose the belief entirely
        ##                    (loc_err_mean up to 11.1 wu, loc_err_max up to 59.1 wu)
        ##   lpf=0.0: 6/20 success, 0/20 lost -- ALL 8 recover, and the WORST
        ##                    belief error over all 20 episodes is 0.36 wu mean /
        ##                    1.86 wu max, i.e. sub-cell everywhere.
        ## Mechanism: this is a belief<->control FEEDBACK loop, invisible open-loop.
        ## `pos` here is the RAW MAP estimate; the EMA is applied downstream inside
        ## `pid.step`, so it does not bias `loc_err` -- it changes the ACTIONS, hence
        ## the trajectory, hence the scan sequence the filter conditions on. With
        ## kp=5/kd=0 the lagged estimate also overshoots. Prefer LIDAR_BALL_POS_LPF=0
        ## at L3/L4 until a replacement damper is measured; the guard accepts "0"
        ## because the test is `_lpf in ('', None)`, not truthiness.
        _lpf = getattr(args, 'ball_pos_lpf', '')
        pos_lpf = (0.7 if self._pos_source == 'belief' else 0.0) \
            if _lpf in ('', None) else float(_lpf)
        self._pid = PID2D(
            kp=float(getattr(args, 'ball_kp', 5.0)),
            kd=float(getattr(args, 'ball_kd', 0.0)),
            ki=float(getattr(args, 'ball_ki', 0.0)),
            ki_zone=float(getattr(args, 'ball_ki_zone', 2.0)),
            pos_lpf=pos_lpf)
        self._ball_lookahead = int(getattr(args, 'ball_lookahead', 4))
        self._ball_decode_stride = max(int(getattr(args, 'ball_decode_stride', 2)), 1)
        ## plan decode: 'filter' (historical forward filter + per-step argmax) or
        ## 'viterbi' (max-product over paths). The filter's argmax is taken per
        ## step INDEPENDENTLY, so it can hop between posterior modes; measured on
        ## the filter-decode benchmark timelines, 89% of the steps where the ball came off its
        ## plan sit on such a hop (uniform null 8%, lift 10.6x, p=1.7e-137).
        ## Default 'filter' so an unset run is bit-identical to 46179.
        self._ball_decode_mode = str(
            getattr(args, 'ball_decode_mode', 'filter')).lower()
        assert self._ball_decode_mode in ('filter', 'viterbi'), self._ball_decode_mode
        self._ball_decode_rmax = float(getattr(args, 'ball_decode_rmax', 3.0))

        ## ---- per-segment aim poses (the offline --waypoints fix, in the loop) ----
        ## Offline (2026-07-30) every intermediate composed segment got
        ## stgl_cond={} -- anchored by its backward overlap but AIMED at nothing.
        ## That is the whole composed defect: BFS progress was 17%/83% across the
        ## two halves of the plan and the MAP and filtered decodes disagreed by
        ## 18.7 wu. Handing each segment a goal-stgl at a maze-shortest-path
        ## waypoint moved ev_probs G2b 44% -> 93%, endErr 17.6 -> 0.5 wu, decode
        ## disagreement -> 0.50 wu and the progress profile to a flat 50/50.
        self._ball_waypoints = bool(int(getattr(args, 'ball_waypoints', 0)))
        self._ball_wp_fallbacks = 0
        self._ball_wp_calls = 0
        ## replans whose BFS start came from localizing the START SCAN because the
        ## belief was not seeded yet -- i.e. plan 0 of an L3 episode. See
        ## _ball_pos_for_bfs; this used to be a silent decline to add waypoints.
        self._ball_wp_scan_seeds = 0

        ## ---- pursuit: how the plan index advances (BALL.md 7.2) ----
        ## 'time' = one index per control step (what every rollout has used);
        ## 'progress' = nearest plan point at or ahead of the agent, the pure
        ## pursuit BALL.md actually specifies.
        self._ball_pursuit = str(getattr(args, 'ball_pursuit', 'time')).lower()
        assert self._ball_pursuit in ('time', 'progress'), self._ball_pursuit
        self._ball_pursuit_win = int(getattr(args, 'ball_pursuit_win', 60))
        ## >0 => the pursuit target is this far ALONG the plan from the cursor,
        ## in wu, instead of a fixed number of indices (progress arm only).
        self._ball_lookahead_wu = float(getattr(args, 'ball_lookahead_wu', 0.0))
        self._ball_cursor = 0
        ## stall escape: control steps without cursor advance before we force it
        self._ball_stall_steps = int(getattr(args, 'ball_stall_steps', 50))
        self._ball_stall = 0
        self._ball_stall_escapes = 0
        ## ---- line-of-sight pursuit (2026-07-30 night, benchmark post-mortem) ----
        ## Never aim at a plan point the ball cannot drive at in a straight line.
        ## Two places use it, both off unless ball_los=1 so the control arm stays
        ## bit-identical: (a) the carrot is pulled back to the furthest VISIBLE
        ## point at or before the nominal lookahead; (b) the stall escape SEEKS
        ## the furthest visible point ahead instead of marching +1/step.
        self._ball_los = bool(int(getattr(args, 'ball_los', 0)))
        self._ball_los_margin = float(getattr(args, 'ball_los_margin', 0.6))
        self._ball_los_pullbacks = 0   # carrot moved nearer because of a wall
        self._ball_los_seeks = 0       # stall escapes resolved by a visible point
        self._ball_los_blind = 0       # nothing visible ahead: old behaviour used

        ## ---- cursor runaway (2026-07-31, arm I timeline post-mortem) ----
        ## `los_blind` is not a route defect, it is a DISTANCE symptom: measured
        ## over 47021 control steps the ball->target distance is p50 6.33 wu
        ## (p90 25.5, max 50.3) and LOS is clear 98% of the time under 2 wu but
        ## ~0% beyond 6 wu. The cause is the stall escape above: it marched the
        ## cursor +1/step 11750 times (25% of ALL steps) while the LOS seek that
        ## was meant to rescue it fired 109 times (0.9%), because BOTH search
        ## forward-only from a cursor that has already left the ball behind.
        ## Two knobs, both default 0 = off so an unset run stays bit-identical:
        ##   ball_cursor_leash_wu -- do not blind-march past this euclidean
        ##     distance from the ball; re-acquire BACKWARD instead (the only
        ##     place the monotone cursor is allowed to move backward).
        ##   ball_max_carrot_wu   -- the carrot is capped by straight-line
        ##     distance, not just by plan ARC (2 wu of arc across a decode jump
        ##     is up to 50 wu of real distance).
        self._ball_cursor_leash_wu = float(getattr(args, 'ball_cursor_leash_wu', 0.0))
        self._ball_max_carrot_wu = float(getattr(args, 'ball_max_carrot_wu', 0.0))
        self._ball_pursuit_back = int(getattr(args, 'ball_pursuit_back', 15))
        self._ball_cursor_reacquires = 0   # backward re-acquisitions (leash on)
        self._ball_cursor_dist = 0.0       # ||pos - plan[cursor]|| this step

        ## ---- frozen time (2026-08-01, `ball_step_budget.py` on the baseline arm) ----
        ## At OGBench's own 1000-step budget, **51% of control steps move the ball
        ## ZERO wu**, and a moving ball always moves at the physical maximum
        ## (displacements are quantised at 0 / 0.200 / 0.283 wu -- there is no
        ## "slow", only "stopped"). Deleting the frozen steps takes the SAME plans
        ## from 45.2 to 74.2 @1000, so this is the whole remaining gap. Two
        ## mechanisms, each with its own knob, all default 0 = OFF so an unset run
        ## is bit-identical to the reference run:
        ##
        ##   A-idle (17.7% of the budget, worth +6.2 pts). The carrot collapses
        ##     onto the cursor point the ball is already standing on (|target-pos|
        ##     p50 0.19 wu, |u| p50 0.0), so the command is ~0, the ball cannot
        ##     move, and the cursor -- an argmin over the ball's position -- cannot
        ##     advance either. Only the stall timer breaks it, 50 steps later.
        ##     `ball_min_carrot_wu` refuses a carrot that close: 2 wu of plan ARC
        ##     is not 2 wu of DISTANCE when the plan loiters, and 57% of A-idle
        ##     steps had a visible point ahead and idled anyway.
        ##   B-pinned (32.9%, worth +19.0 pts). The escape marched the monotone
        ##     cursor out of reach (carrot p50 27 wu), the PID saturates straight
        ##     into a wall, and only a replan resets the cursor -- a backward
        ##     re-acquire reaches a point within 3 wu on just 16.6% of these steps
        ##     (31.1% even searching the whole plan), which is why the leash was
        ##     measured NEGATIVE. `ball_wall_slide` instead deflects the command
        ##     along a free heading taken from the LIVE SCAN: 84.7% of pinned
        ##     steps have one that still points goal-ward. Scan only -- no map,
        ##     no BFS, so the external comparison is unaffected.
        ##   `ball_frozen_steps` fires the existing escape on actual zero
        ##     DISPLACEMENT rather than on cursor non-advance, which is the
        ##     physical condition it was written for; it only ever fires EARLIER
        ##     than `ball_stall_steps`, never later.
        self._ball_frozen_steps = int(getattr(args, 'ball_frozen_steps', 0))
        self._ball_frozen_eps = float(getattr(args, 'ball_frozen_eps', 0.02))
        self._ball_min_carrot_wu = float(getattr(args, 'ball_min_carrot_wu', 0.0))
        self._ball_wall_slide = bool(int(getattr(args, 'ball_wall_slide', 0)))
        self._ball_slide_wu = float(getattr(args, 'ball_slide_wu', 1.0))
        self._ball_frozen = 0              # consecutive steps with |dxy| < eps
        self._ball_frozen_escapes = 0      # escapes the frozen trip fired early
        self._ball_carrot_floor_hits = 0   # carrot pushed out by the floor
        self._ball_slides = 0              # steps the command was deflected
        self._ball_ep_esc0 = 0             # per-episode baselines for the two
        self._ball_ep_reacq0 = 0           # cumulative counters above

        ## ---- deadlock (zigzag) detector ----
        self._dl_win = int(getattr(args, 'ball_deadlock_win', 100))
        self._dl_disp = float(getattr(args, 'ball_deadlock_disp', 0.5))
        self._dl_umin = float(getattr(args, 'ball_deadlock_umin', 0.3))
        self._ball_log_npz = bool(int(getattr(args, 'ball_log_npz', 0)))

        ## ---- ball has no yaw: route/probe/band scans must be world-frame ----
        self._wrap_zero_yaw_synth()

        ## ---- decoded-plan bookkeeping + per-episode logs ----
        self._ball_plan_xy = None         # (H, 2) decoded targets of current plan
        self._ball_new_plan = False
        self._ball_plan_start_wp = 0
        self._ball_prev_env_xy = None     # for the 4-D state's finite-diff vel
        self._ball_ep = -1                # episode counter (reset() driven)
        self._ball_ep_log = None
        self._ball_ep_summaries = []
        self._ball_decode_stats = []
        self._ball_plans = []             # every plan of the CURRENT episode
        self._ball_true_goal_xy = None    # LOG ONLY (_ball_plan_goal_stats)

        ## wrap plan generation (AFTER gridnav's wrapper, so nav bypass plans
        ## also pass through here): decode every produced scan plan to xy once.
        self._ball_real_gen = self.policy.gen_cond_stgl
        self.policy.gen_cond_stgl = self._ball_gen_cond_stgl

        utils.print_color(
            f'[ball] pos_source={self._pos_source} pid=(kp={self._pid.kp} '
            f'kd={self._pid.kd} ki={self._pid.ki} ki_zone={self._pid.ki_zone} '
            f'pos_lpf={self._pid.pos_lpf}) lookahead={self._ball_lookahead} '
            f'decode_stride={self._ball_decode_stride} '
            f'decode_mode={self._ball_decode_mode}(rmax={self._ball_decode_rmax}) '
            f'waypoints={int(self._ball_waypoints)} '
            f'pursuit={self._ball_pursuit}(win={self._ball_pursuit_win}) '
            f'los={int(self._ball_los)}(margin={self._ball_los_margin}) '
            f'leash={self._ball_cursor_leash_wu}(back={self._ball_pursuit_back}) '
            f'max_carrot={self._ball_max_carrot_wu} '
            f'min_carrot={self._ball_min_carrot_wu} '
            f'frozen_steps={self._ball_frozen_steps}(eps={self._ball_frozen_eps}) '
            f'wall_slide={int(self._ball_wall_slide)}(free>={self._ball_slide_wu}) '
            f'deadlock=(win={self._dl_win} disp={self._dl_disp} umin={self._dl_umin})',
            c='g')

    def _wrap_zero_yaw_synth(self):
        """Re-render gridnav route/probe/band scans at yaw=0 (the ball's frame).

        The RouteScanSynthesizer casts scans at the route-tangent heading (an
        ant convention); ball training scans are world-frame, so executing
        tangent-frame scans would hand the executor/localizer out-of-
        distribution goals. Wraps in place; inert when gridnav is off."""
        synth = getattr(self, '_nav_synth', None)
        if synth is None:
            return
        scanner = self.scanner

        def _zero(route):
            if route is None:
                return route
            xy = np.asarray(route['xy'], dtype=np.float64)
            route['yaw'] = np.zeros(len(xy), dtype=np.float32)
            route['scans'] = scanner.scan_batch(
                xy, np.zeros(len(xy))).astype(np.float32)
            return route

        _synthesize, _probe, _band = synth.synthesize, synth.probe, synth.band_along_route

        def synthesize(*a, **kw):
            return _zero(_synthesize(*a, **kw))

        def probe(*a, **kw):
            return _zero(_probe(*a, **kw))

        def band_along_route(route, k, band_step_wu):
            band = _band(route, k, band_step_wu)
            ## re-render the K band poses at yaw 0
            xy = np.asarray(route['xy'], dtype=np.float64)
            stride = max(int(round(float(band_step_wu) / synth.step_wu)), 1)
            idx = [len(xy) - 1 - j * stride for j in range(int(k))]
            idx = np.clip(np.asarray(idx[::-1], dtype=int), 0, len(xy) - 1)
            return scanner.scan_batch(xy[idx], np.zeros(len(idx))).astype(np.float32)

        synth.synthesize, synth.probe, synth.band_along_route = \
            synthesize, probe, band_along_route
        utils.print_color('[ball] gridnav synth wrapped: world-frame (yaw=0) scans', c='c')

    # ================= normalizers =================
    def _setup_normalizers(self):
        """Arm B: the ant path (normalizers from the inv-dyn run's dataset,
        train = trailing LiDAR slice). Arm A: no inv-dyn run exists, so rebuild
        the PLANNER's own dataset from its saved dataset_config.pkl -- its
        normalizer is bit-exact the training normalizer (the LiDAR ray-cast is
        disk-cached, so this is minutes, not hours)."""
        if getattr(self.args, 'inv_model_path', '') or '':
            super()._setup_normalizers()          # ant logic; lidar_offset = 4 (xyv)
            assert self.lidar_offset == N_XYV, (
                f'ball inv-dyn obs must be xyv_lidar (lidar_offset 4), '
                f'got offset {self.lidar_offset}')
            return
        run_dir = os.path.join(self.args.logbase, self.args_train.dataset,
                               self.args_train.exp_name)
        utils.print_color(
            f'[ball] Arm A: rebuilding planner dataset normalizer from {run_dir}', c='c')
        dset_cfg = load_config(run_dir, 'dataset_config.pkl')
        self.n_beams = int(self.args_train.dataset_config['lidar_config']['n_beams'])
        dset = dset_cfg()
        self.train_normalizer = dset.normalizer
        self.full_normalizer = dset.normalizer    # same object: obs == the scan
        self.lidar_offset = 0
        try:
            del dset
        except Exception:  # noqa: BLE001
            pass

    def check_is_same_nmlizer(self):
        if self.train_normalizer is self.full_normalizer:
            return                                  # Arm A: identical by construction
        super().check_is_same_nmlizer()

    # ================= inv-dyn (Arm B only) =================
    def load_inv_model(self):
        """Arm B: generic inv-dyn loading (the base's pointmaze branch would
        skip it). Arm A: the base pointmaze no-model stub."""
        self.inv_model_path = str(getattr(self.args, 'inv_model_path', '') or '')
        if not self._ball_arm_b:
            self.inv_epoch = None
            self.inv_model = SimpleNamespace()
            self.inv_model.training = None
            utils.print_color('\n[ball] Arm A: *** No Inv Dyn Model (PID executor) ***\n', c='y')
        else:
            from diffuser.ogb_task.og_inv_dyn.og_invdyn_helpers import ogb_load_invdyn_maze_v1
            inv_model, inv_ema, inv_epoch = ogb_load_invdyn_maze_v1(
                self.inv_model_path, epoch=self.args.inv_epoch)
            self.inv_epoch = inv_epoch
            self.inv_model = inv_model
            utils.print_color(f'\n[ball] Arm B inv-dyn loaded: {self.inv_model.training=}\n', c='y')

        ## v2 auto-detection (identical to the base): goal encoding + action
        ## chunk come from the RUN'S OWN trainer pkl; defaults = legacy.
        self._inv_goal_enc = 'abs'
        self._inv_act_chunk = 1
        try:
            _tr = load_config(self.inv_model_path, 'inv_trainer_config.pkl')
            _td = _tr._dict.get('trainer_dict', {}) or {}
            self._inv_goal_enc = str(_td.get('goal_encoding', 'abs'))
            self._inv_act_chunk = int(_td.get('act_chunk', 1) or 1)
        except Exception:  # noqa: BLE001
            pass
        self._chunk_buf = []
        self._chunk_repl = None
        _cx = int(getattr(self.args, 'act_chunk_exec', 0) or 0)
        self._act_chunk_exec = _cx if _cx > 0 else max(self._inv_act_chunk // 2, 1)
        if self._inv_goal_enc != 'abs' or self._inv_act_chunk > 1:
            utils.print_color(
                f'[ball inv v2] goal_encoding={self._inv_goal_enc} '
                f'act_chunk={self._inv_act_chunk} (exec {self._act_chunk_exec}/query)', c='y')

    # ================= live obs bridges (no yaw) =================
    def _obs_xy_yaw(self, obs):
        """2-D pointmaze obs -> (x, y, yaw=0). The ant version reads a torso
        quaternion at obs[3:7], which does not exist here."""
        o = np.asarray(obs, dtype=np.float64).reshape(-1)
        return float(o[0]), float(o[1]), 0.0

    def _nav_proprio_yaw(self, st=None):
        return 0.0

    def _ball_env_vel(self, xy):
        """Finite-difference velocity of the TRUE position (dataset convention:
        v_t = xy_t - xy_{t-1}, v = 0 at episode start)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        v = np.zeros(2) if self._ball_prev_env_xy is None else xy - self._ball_prev_env_xy
        self._ball_prev_env_xy = xy.copy()
        return v

    def _obs_to_invdyn_obs(self, obs):
        """live 2-D obs -> the 34-D [x, y, vx, vy, lidar] composite (Arm B)."""
        x, y, _ = self._obs_xy_yaw(obs)
        lidar = self._obs_to_scan(obs)
        v = self._ball_env_vel(np.array([x, y]))
        head = np.array([x, y, v[0], v[1]], dtype=np.float32)
        return np.concatenate([head, lidar]).astype(np.float32)

    def _get_invdyn_obs_nm(self, obs_cur):
        if self._ball_arm_b:
            return super()._get_invdyn_obs_nm(obs_cur)   # composite via override above
        ## Arm A: the value is only consumed by the pred_inv branch (not taken);
        ## return the normalized current scan so shapes stay sane for logging.
        return self.train_normalizer.normalize(self._obs_to_scan(obs_cur), 'observations')

    # ================= per-segment aim poses =================
    def _ball_pos_for_bfs(self, st_scan=None):
        """Where the planner believes it is, for the waypoint BFS.

        Deliberately the SAME source the PID executor uses (``loc_source``), so
        turning waypoints on cannot smuggle ground-truth position into an L3
        (belief) run.

        ``st_scan`` is the START scan the sampler was handed. It exists because
        the belief is refreshed in ``_lidar_arrival_step``, which runs AFTER the
        first plan of an episode is sampled -- so an unseeded belief used to make
        plan 0 go out with no waypoints, in 100% of the episodes of every L3 run
        (2026-07-31). That is the pre-waypoint control
        arm, and it cost the first plan outright: tail_gd p50 17.18 wu vs L2's
        0.59 (the offline waypoint A/B is 17.6 -> 0.5) and 0/300 first-plan
        successes vs 51/300. Localizing the start scan seeds the BFS from LiDAR
        alone -- the same round trip the goal in _ball_add_waypoints already
        takes -- so the guard against a ground-truth position leak still holds.
        """
        if self._pos_source == 'belief':
            if self._ball_belief_est is not None:
                return np.asarray(self._ball_belief_est, dtype=np.float64).reshape(2)
            if st_scan is None:
                return None                         # nothing to localize: decline
            xy = self.loc.localize_single(np.asarray(st_scan, dtype=np.float32))['xy']
            self._ball_wp_scan_seeds += 1
            utils.print_color('[ball waypoints] belief not seeded yet -> BFS start '
                              'localized from the START SCAN (LiDAR only)', c='y')
            return np.asarray(xy, dtype=np.float64).reshape(2)
        return np.asarray(self.env.get_xy(), dtype=np.float64).reshape(2)

    def _ball_add_waypoints(self, g_cond):
        """Give every intermediate composed segment a goal-stgl at a maze
        shortest-path waypoint, instead of the ``stgl_cond={}`` it gets by
        default. Mutates ``g_cond`` in place; a no-op unless enabled.

        The start is the agent's CURRENT position, so the waypoints are rebuilt
        from scratch on every replan -- unlike the offline dump, where they are
        computed once from the episode start.
        """
        n_comp = int(getattr(self.policy, 'n_comp', 1))
        if not self._ball_waypoints or n_comp < 2:
            return                                  # nothing to aim at
        if getattr(self, 'nav_mode', '') in ('bfs', 'tf_exec'):
            return                                  # gridnav bypass: no diffusion plan to aim
        ## (1+K, n_probs, D): row 0 is the START scan, row 1 the goal scan.
        st_gl = np.asarray(g_cond['st_gl'], dtype=np.float32)
        pos = self._ball_pos_for_bfs(st_gl[0, 0])
        if pos is None:
            utils.print_color('[ball waypoints] belief not seeded yet -> plan without '
                              'waypoints (this replan only)', c='y')
            return
        ## the goal in world coords, from the goal SCAN the sampler was given --
        ## the same round trip _pid_lidar_act uses, so no extra privilege.
        gl_scan = st_gl[1, 0]
        goal_xy = np.asarray(self.loc.localize_single(gl_scan)['xy'], dtype=np.float64).reshape(2)
        wps, wps_xy = make_waypoints(self.scanner, pos[None], goal_xy[None], n_comp)
        g_cond['waypoints'] = wps                   # (n_comp-1, 1, n_beams)
        self._ball_wp_calls += 1
        self._ball_last_wp_xy = wps_xy[:, 0]        # (n_comp-1, 2), for logging
        ## A waypoint inside a wall means the BFS fell back to a straight line
        ## (unreachable pair) -- that synthesizes a scan from a pose the robot
        ## can never occupy, so the segment is aimed at nonsense. Loud, counted.
        walled = [i for i, (x, y) in enumerate(wps_xy[:, 0])
                  if self._ball_xy_in_wall(x, y)]
        if walled:
            self._ball_wp_fallbacks += 1
            utils.print_color(
                f'[ball waypoints] {len(walled)}/{n_comp - 1} aim poses are INSIDE WALLS '
                f'(indices {walled}) -- the BFS found no path from the current position '
                f'to the goal and fell back to straight-line interpolation. Those segments '
                f'are aimed at an unrealizable scan.', c='r')

    def _ball_xy_in_wall(self, x, y):
        ci, cj = self.scanner.world_to_cell(np.array([x]), np.array([y]))
        i, j = int(np.ravel(ci)[0]), int(np.ravel(cj)[0])
        mp = np.asarray(self.scanner.maze_map)
        return not (0 <= i < mp.shape[0] and 0 <= j < mp.shape[1]) or mp[i, j] == 1

    # ================= plan decode (Arm A's "inverse observation model") ====
    def _ball_gen_cond_stgl(self, g_cond=None, b_s=None, **kw):
        """Wrapper over policy.gen_cond_stgl (stacked on top of gridnav's, if
        any): after ANY plan is produced, decode its waypoint scans to (x, y)
        targets once, so the per-step controller only indexes an array."""
        ## must run HERE, not where g_cond is built: n_comp is only final after
        ## the replan branch has adjusted it, and waypoint count is n_comp-1.
        if g_cond is not None:
            self._ball_add_waypoints(g_cond)
        m_out = self._ball_real_gen(g_cond=g_cond, b_s=b_s, **kw)
        try:
            nav_xy = getattr(self, '_nav_plan_xy', None)
            if getattr(self, 'nav_mode', '') in ('bfs', 'tf_exec') and nav_xy is not None:
                ## gridnav plans carry their own (x, y) ribbon -- no decode needed
                self._ball_plan_xy = np.asarray(nav_xy, dtype=np.float64).copy()
                src = f'nav/{self.nav_mode}'
            else:
                self._ball_plan_xy = self._decode_plan_xy(np.asarray(m_out.pick_traj))
                src = 'grid-decode'
            self._ball_new_plan = True
            self._ball_record_plan()
            utils.print_color(
                f'[ball decode] plan -> {len(self._ball_plan_xy)} xy targets ({src})', c='c')
        except Exception as e:  # noqa: BLE001
            self._ball_plan_xy = None
            utils.print_color(f'[ball decode] FAILED ({e}) -> per-step single-scan decode', c='y')
        return m_out

    def _ball_record_plan(self):
        """Snapshot the plan just decoded, for the CONTENT question.

        2026-07-31 (`latest+12`): the executor is closed -- it rides the plan to
        its end on 95% of episodes and still only succeeds on 49% of those, so
        the residual is whether the PLAN ends at the goal. That could not be
        answered from the existing npz: 99% of episodes replan (p50 = 2), so
        `argmax(cursor)` indexes whichever plan reached deepest, and the LOS
        pull-back means `target` is often not the carrot. The proxy built on
        those two mistakes scored corr(tail, final_gd) = 0.14. Hence: keep EVERY
        plan, tagged with the control step it took effect on, so a plan can be
        matched to the segment of the timeline that was actually chasing it.

        Cost is ~14 KB per plan and ~3 plans per episode, so this is always on;
        only the npz arrays are gated behind LIDAR_BALL_LOG_NPZ.
        """
        try:
            if self._ball_plan_xy is None:
                return
            L = self._ball_ep_log
            step0 = len(L['true_xy']) if L is not None else 0
            self._ball_plans.append(dict(
                step0=int(step0),
                xy=np.asarray(self._ball_plan_xy, dtype=np.float64).copy()))
        except Exception:  # noqa: BLE001
            pass                                     # diagnostics never break a run

    def _ball_plan_goal_stats(self):
        """Per-plan geometry against the TRUE goal -- LOG ONLY, never control.

        `self._ball_true_goal_xy` is the episode's real goal, stashed in the
        reset hook. It must NEVER reach the controller: the whole point of the
        L2/L3 rungs is that the goal is derived from LiDAR, and reading the true
        one anywhere on the control path silently converts L3 into L1. It is
        used here, and only here, to answer "does the decoded plan END at the
        goal?" -- which is a question about the planner, not an input to it.

        Two numbers per plan, and the pair is the point:
          `tail_gd`    -- distance from the plan's LAST point to the goal.
          `closest_gd` -- the plan's closest approach to the goal anywhere.
        tail large + closest small  => the plan reaches the goal and runs PAST
        it (a tail/termination defect, `latest+2`'s "goal lives in the TAIL").
        tail large + closest large  => the plan never goes there at all (a route
        defect). They call for completely different fixes.
        """
        g = self._ball_true_goal_xy
        if g is None or not self._ball_plans:
            return {}
        tails, closes = [], []
        for p in self._ball_plans:
            xy = p['xy']
            if len(xy) == 0:
                continue
            d = np.linalg.norm(xy - g, axis=1)
            tails.append(float(d[-1]))
            closes.append(float(d.min()))
        if not tails:
            return {}
        return dict(
            n_plans=len(tails),
            ## the LAST plan is the one the episode ended under
            plan_tail_gd_last=tails[-1], plan_closest_gd_last=closes[-1],
            ## and the best any plan managed, so a good plan replaced by a bad
            ## one is distinguishable from never having had one
            plan_tail_gd_best=min(tails), plan_closest_gd_best=min(closes),
        )

    def _decode_plan_xy(self, scans):
        """(H, n_beams) plan scans -> (H, 2) xy targets.

        Sequence-consistent decode: the histogram filter is run along the plan
        (strided for speed), so self-similar corridors are disambiguated by the
        plan's own motion; the strided estimates are then linearly interpolated
        back to all H frames and lightly smoothed (PID targets should not jump).
        """
        H = scans.shape[0]
        assert scans.ndim == 2 and scans.shape[1] == self.n_beams, scans.shape
        stride = self._ball_decode_stride
        idx = np.unique(np.concatenate([np.arange(0, H, stride), [H - 1]]))
        ms = float(getattr(self.args, 'loc_motion_sigma', 1.0))
        if self._ball_decode_mode == 'viterbi':
            ## max-product over PATHS: the filter's per-step argmax can hop
            ## between posterior modes, and the carrot then crosses a wall the
            ## ball cannot follow. See GridLocalizer.localize_viterbi.
            r = self.loc.localize_viterbi(
                scans[idx].astype(np.float32), motion_sigma_cells=ms,
                r_max_cells=float(self._ball_decode_rmax))
        else:
            r = self.loc.localize_sequence(
                scans[idx].astype(np.float32),
                motion_sigma_cells=ms, theta_blur=0)
        map_xy = np.asarray(r['map_xy'], dtype=np.float64)         # (n_dec, 2)
        full = np.stack([np.interp(np.arange(H), idx, map_xy[:, 0]),
                         np.interp(np.arange(H), idx, map_xy[:, 1])], axis=1)
        if H >= 3:                                                 # light smoothing
            k = np.ones(3) / 3.0
            pad = np.pad(full, ((1, 1), (0, 0)), mode='edge')
            full = np.stack([np.convolve(pad[:, 0], k, 'valid'),
                             np.convolve(pad[:, 1], k, 'valid')], axis=1)
        ## decode-quality stat: fraction of consecutive decoded steps that jump
        ## more than one maze cell (should be ~0 for a healthy plan+decode)
        jumps = np.linalg.norm(np.diff(map_xy, axis=0), axis=1)
        frac_jump = float((jumps > self.scanner.maze_unit).mean()) if len(jumps) else 0.0
        self._ball_decode_stats.append(dict(n=H, frac_cell_jumps=frac_jump))
        return full

    # ================= pursuit =================
    def _ball_advance_cursor(self, pos):
        """Monotone progress cursor into the decoded plan (pure pursuit).

        The time parameterisation advances the plan index once per control step,
        so the target runs away whenever the ball is slower than the plan's own
        pace -- and the plan is generated at DATA pace, which dawdles (arc 0.192
        wu/frame but only 0.0755 wu/frame of actual maze progress). Measured on
        the 2026-07-30 L1 smoke, distance to the current subgoal grew monotonely
        to ~25 wu, i.e. the agent was chasing a point most of a maze away.

        Instead: snap to the nearest plan index AT OR AHEAD of where we already
        are, searching only a bounded window forward. Forward-only keeps the
        agent from falling back into a loop it has already run; the bounded
        window keeps it from teleporting across a plan that revisits a corridor.
        """
        plan = self._ball_plan_xy
        lo = int(self._ball_cursor)
        hi = min(lo + int(self._ball_pursuit_win) + 1, len(plan))
        if hi > lo:
            d = np.linalg.norm(plan[lo:hi] - np.asarray(pos, dtype=np.float64), axis=1)
            self._ball_cursor = lo + int(np.argmin(d))

        ## ---- STALL ESCAPE ----
        ## Progress pursuit only advances when the ball gets NEAR a plan point,
        ## so an unreachable one traps it forever: the decoded plan still carries
        ## ~5 residual jumps, and when the cursor lands just before one the
        ## target sits across a wall, the PID drives into that wall, the cursor
        ## never advances, and the episode is over. Measured on one benchmark run: the
        ## one successful episode consumed 100% of its plan with no deadlock,
        ## while all four failures deadlocked with the target stranded 2.6-8.8 wu
        ## away and only 5-55% of the plan consumed. Time-parameterised pursuit
        ## never had this failure (its index moved on regardless), which is why
        ## it out-travelled progress even while it zigzagged.
        ## So: if the cursor has not moved for `stall_steps`, walk it forward
        ## anyway until it re-acquires. Counted, never silent.
        if self._ball_cursor > lo:
            self._ball_stall = 0
        else:
            self._ball_stall += 1
            ## FROZEN TRIP (2026-08-01). `_ball_stall` counts CURSOR non-advance;
            ## the condition the escape was written for is the BALL not moving,
            ## and the two differ by up to `stall_steps` of dead time -- 18.6% of
            ## the 1000-step budget sits in the first min(run, 50) steps of a
            ## frozen run. Tripping on measured displacement fires the same
            ## escape earlier and never later.
            ## ⚠️ The escape it fires is the BLIND MARCH, which is what produces
            ## the B-pinned family, so firing it sooner can trade A-idle time for
            ## more of it. Run this knob on top of `min_carrot_wu` + `wall_slide`,
            ## never on its own.
            frozen_trip = (self._ball_frozen_steps > 0
                           and self._ball_frozen >= self._ball_frozen_steps)
            if (self._ball_stall_steps > 0
                    and self._ball_stall >= self._ball_stall_steps) or frozen_trip:
                self._ball_frozen_escapes += int(
                    frozen_trip and not (self._ball_stall_steps > 0
                                         and self._ball_stall >= self._ball_stall_steps))
                ## LOS SEEK. The +1/step march below frees a cursor that is stuck
                ## on an unreachable point, but it does so blind: it walks the
                ## carrot down the plan one index per control step whether or not
                ## the ball can follow, so a trapped ball ends up chasing a point
                ## in another corridor while pressed into a wall. A benchmark run:
                ## 13155 escapes over 20 episodes (~26% of control steps, ~658 of
                ## a 888-index plan per episode) and target_dist_mean rose from
                ## 0.22 wu to 7-19 wu, while plan_frac still read 1.00 -- the
                ## CURSOR finished the plan, the ball did not. Seeking the
                ## furthest VISIBLE point instead re-acquires in one step, or
                ## falls through to the old march when nothing is visible.
                nxt = self._ball_los_seek(pos, lo, hi)
                if nxt is not None:
                    self._ball_cursor = nxt
                    self._ball_los_seeks += 1
                    self._ball_stall = 0
                else:
                    ## CURSOR LEASH. The march below is the original escape and
                    ## it is correct while the cursor is still NEAR the ball --
                    ## it frees a cursor pinned on a locally unreachable point.
                    ## What it must never do is keep walking once the ball has
                    ## been left behind: `lo` only ever increases and the LOS
                    ## seek only looks FORWARD, so nothing can bring the carrot
                    ## back and the loop runs to the end of the plan (
                    ## 11750 marches, runs of 40-145 consecutive steps, ball ->
                    ## target p50 6.3 wu). So the leash gates the point we are
                    ## about to march TO -- not the one we are on, which a
                    ## single decode jump can put a maze away from its
                    ## successor -- and past it we re-acquire BACKWARD instead,
                    ## the one place the monotone cursor may retreat.
                    nxt = min(lo + 1, len(plan) - 1)
                    back = None
                    if self._ball_cursor_leash_wu > 0 and np.linalg.norm(
                            plan[nxt] - np.asarray(pos, dtype=np.float64)
                    ) > self._ball_cursor_leash_wu:
                        back = self._ball_cursor_reacquire(pos, lo)
                    if back is not None:
                        self._ball_cursor = back
                        self._ball_cursor_reacquires += 1
                        self._ball_stall = 0
                    else:
                        ## nothing within reach behind either: escape anyway, so
                        ## a leashed run is never MORE stuck than the control arm
                        self._ball_cursor = nxt
                        self._ball_stall_escapes += 1
        self._ball_cursor_dist = float(np.linalg.norm(
            plan[int(self._ball_cursor)] - np.asarray(pos, dtype=np.float64)))
        return int(self._ball_cursor)

    def _ball_cursor_reacquire(self, pos, lo):
        """Nearest plan index at or behind the cursor that is WITHIN the leash of
        the ball, or None (leash off, or nothing in the backward window is in
        reach -- in which case the caller escapes forward as before). Prefers a
        point with clear line of sight; `lo` itself is a legal answer and means
        'hold here', which is the correct move when the cursor is already the
        closest thing the ball can drive at."""
        leash = self._ball_cursor_leash_wu
        if leash <= 0:
            return None
        plan = self._ball_plan_xy
        pos = np.asarray(pos, dtype=np.float64)
        st = max(0, lo - int(self._ball_pursuit_back))
        cand = plan[st:lo + 1]
        d = np.linalg.norm(cand - pos, axis=1)
        near = d <= leash
        if not near.any():
            return None
        if self._ball_los:
            vis = np.flatnonzero(np.logical_and(
                near, LOS_CLEAR(self.scanner, pos, cand, margin=self._ball_los_margin)))
            if len(vis):
                return int(st + vis[int(np.argmin(d[vis]))])
        idx = np.flatnonzero(near)
        return int(st + idx[int(np.argmin(d[idx]))])

    def _ball_los_seek(self, pos, lo, hi):
        """Furthest plan index in ``(lo, hi)`` with clear line of sight from
        ``pos``, or None (LOS off / window empty / nothing visible)."""
        if not self._ball_los or hi <= lo + 1:
            return None
        ok = LOS_CLEAR(self.scanner, pos, self._ball_plan_xy[lo + 1:hi],
                       margin=self._ball_los_margin)
        idx = np.flatnonzero(ok)
        return int(lo + 1 + idx[-1]) if len(idx) else None

    def _ball_lookahead_idx(self, rel, pos=None):
        """Index of the pursuit target, ``rel`` = the cursor.

        Index-based lookahead is pace-dependent, which is precisely the thing
        that breaks here: the plan advances ~0.1 wu per frame, so the historical
        ``lookahead=4`` is only ~0.4 wu ahead of the agent. Under the old TIME
        parameterisation that did not matter -- the index itself ran ahead and
        supplied the carrot -- but under progress pursuit a 0.4 wu carrot makes
        the ball crawl. So when ``ball_lookahead_wu > 0`` the target is the first
        plan point at least that far along the plan from the cursor, measured by
        cumulative arc length; the index lookahead stays as the fallback.
        """
        plan = self._ball_plan_xy
        if self._ball_lookahead_wu <= 0:
            tgt = int(np.clip(rel + self._ball_lookahead, 0, len(plan) - 1))
        else:
            seg = np.linalg.norm(np.diff(plan[rel:], axis=0), axis=1)
            if not len(seg):
                return len(plan) - 1
            adv = int(np.searchsorted(np.cumsum(seg), self._ball_lookahead_wu) + 1)
            tgt = int(min(rel + adv, len(plan) - 1))
        ## ---- CARROT FLOOR (2026-08-01) ----
        ## ARC along the plan is not DISTANCE from the ball. The decoded plan is
        ## the model's own pace, which dawdles (0.192 wu/frame of arc but only
        ## 0.0755 wu/frame of maze progress), so `lookahead_wu` of arc can be a
        ## wiggle that ends 0.2 wu from where the ball is standing. Measured on
        ## Baseline arm: 17.7% of the 1000-step budget is spent frozen with the
        ## carrot 0.19 wu away and |u| ~ 0 -- a fixed point, because the cursor
        ## is an argmin over the ball's own position and cannot advance while the
        ## ball does not move. So extend the nominal target forward until it is
        ## at least `min_carrot_wu` AWAY. Bounded by the plan; 0 = off.
        ## The forward search is bounded so the LOS window below stays small (it
        ## raycasts every candidate, every control step); if nothing inside the
        ## bound clears the floor, keep the historical target.
        if self._ball_min_carrot_wu > 0 and pos is not None and tgt < len(plan) - 1:
            hi_f = min(len(plan), tgt + 2 * int(self._ball_pursuit_win) + 1)
            d_fwd = np.linalg.norm(plan[tgt:hi_f] - np.asarray(pos, dtype=np.float64),
                                   axis=1)
            far = np.flatnonzero(d_fwd >= self._ball_min_carrot_wu)
            if len(far):
                nxt = int(tgt + far[0])
                self._ball_carrot_floor_hits += int(nxt > tgt)
                tgt = nxt
        ## ---- LOS PULL-BACK + EUCLIDEAN CARROT CAP ----
        ## The carrot above is chosen by arc length along the plan, which says
        ## nothing about whether a wall stands between the ball and it -- nor
        ## how far away it actually is. Walk back from the nominal target to the
        ## cursor and take the furthest point that satisfies both predicates:
        ## visible (LOS), and within `ball_max_carrot_wu` in a straight line.
        ## The distance predicate matters because ARC is not DISTANCE: the
        ## decoded plan carries ~10-14 cell jumps per 888 frames, so 2 wu of arc
        ## can be 50 wu of real distance (target dist p90 25.5 wu,
        ## 54.3% of steps beyond one maze cell), and a saturated PID then drives
        ## at a point across the maze.
        if (self._ball_los or self._ball_max_carrot_wu > 0) \
                and pos is not None and tgt > rel:
            win = plan[rel:tgt + 1]
            ok = (LOS_CLEAR(self.scanner, pos, win, margin=self._ball_los_margin)
                  if self._ball_los else np.ones(len(win), dtype=bool))
            if self._ball_max_carrot_wu > 0:
                near = np.linalg.norm(win - np.asarray(pos, dtype=np.float64),
                                      axis=1) <= self._ball_max_carrot_wu
                ok = np.logical_and(ok, near)
            idx = np.flatnonzero(ok)
            if len(idx):
                ## The pull-back takes the FURTHEST admissible point, but that can
                ## still be one the ball is standing on when the plan loiters --
                ## the A-idle fixed point above. Prefer the furthest admissible
                ## point that also clears the floor; fall back to the old choice
                ## when the wall leaves nothing else (then the escape handles it).
                if self._ball_min_carrot_wu > 0:
                    far = np.linalg.norm(
                        win - np.asarray(pos, dtype=np.float64), axis=1
                    ) >= self._ball_min_carrot_wu
                    pref = np.flatnonzero(ok & far)
                    if len(pref):
                        idx = pref
                new = int(rel + idx[-1])
                self._ball_los_pullbacks += int(new < tgt)
                return new
            self._ball_los_blind += 1
            ## Nothing qualifies. Falling through to the far nominal target is
            ## what produced the runaway; with a cap set, hold at the cursor --
            ## which the leash keeps within reach of the ball.
            if self._ball_max_carrot_wu > 0:
                return rel
        return tgt

    def _ball_slide_cmd(self, pos, target, u):
        """Deflect a saturated command that is producing no motion along a free
        heading read from the LIVE SCAN. Returns ``u`` unchanged unless every
        precondition holds.

        The B-pinned family (32.9% of the 1000-step budget, worth +19.0 pts):
        the cursor has been marched out of reach, the carrot sits p50 27 wu away,
        the pure-P law saturates (|u| p50 1.414 = both axes clipped) and drives
        straight into the wall between them. The ball is not stuck because it is
        weak -- it is stuck because the ONLY direction it is being told to go is
        blocked. On 84.7% of these steps some beam is open >= `slide_wu` AND
        still points goal-ward, so a tangent exists; taking it converts dead time
        into progress.

        SCAN ONLY. `scanner.scan` is the agent's own sensor reading at its own
        believed position -- the same call that produces the observation the
        planner conditions on. No occupancy map, no BFS, no true position: this
        must not widen the map-derived asymmetry that the log already flags
        against the paper's method.

        A backward re-acquire cannot substitute here: only 16.6% of pinned steps
        have a plan point within 3 wu behind the cursor (31.1% searching the whole
        plan), which is why `cursor_leash_wu` measured 3x NEGATIVE.
        """
        if not self._ball_wall_slide or getattr(self, 'scanner', None) is None:
            return u
        u = np.asarray(u, dtype=np.float64).reshape(2)
        clip = float(getattr(self._pid, 'out_clip', 1.0))
        ## only when the ball is DEMONSTRABLY stuck at full effort: a frozen
        ## trace plus a saturated command. Anything less and the PID is fine.
        if (self._ball_frozen < max(self._ball_frozen_steps, 1)
                or np.abs(u).max() < 0.99 * clip):
            return u
        d = np.asarray(target, dtype=np.float64).reshape(2) \
            - np.asarray(pos, dtype=np.float64).reshape(2)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return u
        d = d / n
        r = np.asarray(self.scanner.scan(float(pos[0]), float(pos[1]), 0.0),
                       dtype=np.float64).reshape(-1)
        th = np.asarray(self.scanner.beam_offsets, dtype=np.float64)  # yaw = 0
        if len(th) != len(r):
            return u
        proj = np.cos(th) * d[0] + np.sin(th) * d[1]
        ok = (r >= self._ball_slide_wu) & (proj > 0.0)
        if not ok.any():
            return u                      # boxed in: leave the control arm alone
        k = int(np.flatnonzero(ok)[int(np.argmax(proj[ok]))])
        dirn = np.array([np.cos(th[k]), np.sin(th[k])], dtype=np.float64)
        ## same magnitude convention as the PID: drive the dominant axis to the
        ## clip so the deflected command is no weaker than the one it replaces.
        m = float(np.abs(dirn).max())
        self._ball_slides += 1
        return np.clip(dirn / max(m, 1e-9) * clip, -clip, clip)

    # ================= the PID executor (Arm A) =================
    def _pid_lidar_act(self, obs_cur, goal_cur, wp_idx, fused_traj_ep, i_et):
        """One control step: waypoint scan -> xy target -> PID action.

        Called from the base rollout loop's 'pid_lidar' branch. ``goal_cur`` is
        the (already nav-advanced) waypoint SCAN; the xy target comes from the
        decoded plan cache (fallback: single-scan decode of goal_cur)."""
        ## --- position source (the ladder knob) ---
        ## Read BEFORE picking the target: progress pursuit needs to know where
        ## the agent is, and it must use the same (possibly belief) position the
        ## PID closes on -- not the truth.
        true_xy = np.asarray(self.env.get_xy(), dtype=np.float64)
        if self._pos_source == 'belief' and self._ball_belief_est is not None:
            pos = np.asarray(self._ball_belief_est, dtype=np.float64)
        else:
            pos = true_xy
        ## keep the 4-D state's velocity stream warm even in Arm A (logging + parity)
        ## -- and it is exactly the per-step TRUE displacement the frozen trip and
        ## the wall slide key on (v = xy_t - xy_{t-1}, zero on the first step).
        had_prev = self._ball_prev_env_xy is not None
        v_env = self._ball_env_vel(true_xy)
        if had_prev and float(np.linalg.norm(v_env)) < self._ball_frozen_eps:
            self._ball_frozen += 1
        else:
            self._ball_frozen = 0

        ## --- plan-relative waypoint bookkeeping (mirrors gridnav pursuit) ---
        if self._ball_new_plan:
            self._ball_new_plan = False
            self._ball_plan_start_wp = int(wp_idx)
            self._ball_cursor = 0
            self._ball_stall = 0
            self._ball_frozen = 0
        if self._ball_plan_xy is not None:
            if self._ball_pursuit == 'progress':
                rel = self._ball_advance_cursor(pos)
                rel = self._ball_lookahead_idx(rel, pos)
            else:
                ## TIME parameterisation: the plan index advances one per control
                ## step regardless of where the agent actually is. BALL.md 7.2
                ## specifies progress-based pursuit instead; this branch is kept
                ## as the control arm because every rollout before 2026-07-30
                ## used it -- so it keeps the plain index lookahead, unchanged.
                rel = int(wp_idx) - self._ball_plan_start_wp + self._ball_lookahead
            rel = int(np.clip(rel, 0, len(self._ball_plan_xy) - 1))
            target = self._ball_plan_xy[rel]
        else:                                       # decode failed: 1-scan fallback
            target = np.asarray(
                self.loc.localize_single(np.asarray(goal_cur, dtype=np.float32))['xy'],
                dtype=np.float64)

        u = self._pid.step(pos, target)
        u = self._ball_slide_cmd(pos, target, u)

        ## --- subgoal marker (we HAVE an xy target, unlike the ant) ---
        if self.is_use_subgoal_marker:
            try:
                self.env.set_subgoal_waypnt(np.asarray(target, dtype=np.float64))
            except Exception:  # noqa: BLE001
                pass

        ## --- per-step log + deadlock detector (on TRUE motion) ---
        L = self._ball_ep_log
        if L is not None:
            L['cursor'].append(int(self._ball_cursor))
            L['true_xy'].append(true_xy.copy())
            L['pos_used'].append(np.asarray(pos, dtype=np.float64).copy())
            L['target'].append(np.asarray(target, dtype=np.float64).copy())
            L['u_norm'].append(float(np.linalg.norm(u)))
            L['loc_err'].append(float(np.linalg.norm(pos - true_xy)))
            L['belief_conf'].append(float(self._ball_belief_conf))
            ## ||pos - plan[cursor]||. Without it the 2026-07-31 post-mortem
            ## could measure how far the CARROT was but not how far the CURSOR
            ## had run, which is the quantity the leash bounds.
            L['cursor_dist'].append(float(self._ball_cursor_dist))
            w = self._dl_win
            if (L['deadlock_step'] < 0 and len(L['true_xy']) >= w):
                seg = np.asarray(L['true_xy'][-w:])
                disp = float(np.linalg.norm(seg[-1] - seg[0]))
                u_mean = float(np.mean(L['u_norm'][-w:]))
                if disp < self._dl_disp and u_mean > self._dl_umin:
                    L['deadlock_step'] = int(i_et)
                    utils.print_color(
                        f'[ball DEADLOCK] ep step {i_et}: net disp {disp:.2f} wu over '
                        f'{w} steps with mean|u|={u_mean:.2f} -> zigzag/oscillation. '
                        f'Policy: pause and consult the team (artifact pack '
                        f'{"saved" if self._ball_log_npz else "off: LIDAR_BALL_LOG_NPZ=1 to save"}).',
                        c='r')
                    if self._ball_log_npz:
                        self._ball_dump_artifact(i_et)
        return np.asarray(u, dtype=np.float32)

    def _ball_dump_artifact(self, i_et):
        """Deadlock artifact pack: the last window of the episode timeline + the
        current belief, for the consult-the-team escalation (§5)."""
        try:
            L = self._ball_ep_log
            path = os.path.join(self.savepath, f'deadlock_ep{self._ball_ep}_et{i_et}.npz')
            np.savez_compressed(
                path,
                true_xy=np.asarray(L['true_xy']), pos_used=np.asarray(L['pos_used']),
                target=np.asarray(L['target']), u_norm=np.asarray(L['u_norm']),
                loc_err=np.asarray(L['loc_err']),
                belief=np.asarray(self._ball_live.belief, dtype=np.float32),
                plan_xy=(np.asarray(self._ball_plan_xy)
                         if self._ball_plan_xy is not None else np.zeros((0, 2))))
            utils.print_color(f'[ball DEADLOCK] artifact -> {path}', c='r')
        except Exception as e:  # noqa: BLE001
            utils.print_color(f'[ball DEADLOCK] artifact dump failed ({e})', c='y')

    # ================= per-episode hooks =================
    def _lidar_arrival_reset(self, gl_pos):
        ## flush the finished episode's aggregates BEFORE the base resets state
        self._ball_flush_ep()
        super()._lidar_arrival_reset(gl_pos)
        self._ball_ep += 1
        self._pid.reset()
        self._ball_prev_env_xy = None
        self._ball_plan_xy = None
        self._ball_new_plan = False
        self._ball_plan_start_wp = 0
        self._ball_live.reset()
        self._ball_belief_est = None
        self._ball_belief_conf = 0.0
        self._ball_ep_log = dict(cursor=[], true_xy=[], pos_used=[], target=[], u_norm=[],
                                 loc_err=[], belief_conf=[], cursor_dist=[],
                                 deadlock_step=-1)
        ## every plan this episode decodes, and the true goal to score them
        ## against. LOG ONLY -- see _ball_plan_goal_stats on why the true goal
        ## must not leak onto the control path.
        self._ball_plans = []
        try:
            self._ball_true_goal_xy = np.asarray(gl_pos, dtype=np.float64)[:2].copy()
        except Exception:  # noqa: BLE001
            self._ball_true_goal_xy = None
        ## per-episode baselines for the cumulative escape/re-acquire counters
        self._ball_ep_esc0 = int(self._ball_stall_escapes)
        self._ball_ep_reacq0 = int(self._ball_cursor_reacquires)
        self._ball_cursor_dist = 0.0

    def _lidar_arrival_step(self, obs):
        r = super()._lidar_arrival_step(obs)        # nav tracking + arrival detector
        ## refresh the ball's own belief estimate (shared with nav's if running,
        ## but kept separate so L3 works with gridnav OFF too)
        try:
            nav_live = getattr(self, '_nav_live', None)
            if getattr(self, 'nav_mode', '') and nav_live is not None and nav_live.est_xy is not None:
                self._ball_belief_est = np.asarray(nav_live.est_xy, dtype=np.float64)
                self._ball_belief_conf = float(getattr(nav_live, 'belief', np.zeros(1)).max())
            else:
                st = self._ball_live.update(self._obs_to_scan(obs))
                self._ball_belief_est = np.asarray(st['est_xy'], dtype=np.float64)
                self._ball_belief_conf = float(st['conf'])
        except Exception as e:  # noqa: BLE001
            utils.print_color(f'[ball] belief update err ({e})', c='y')
        return r

    def _ball_flush_ep(self):
        L = getattr(self, '_ball_ep_log', None)
        if not L or not L['true_xy']:
            return
        loc = np.asarray(L['loc_err'])
        xy = np.asarray(L['true_xy'])
        tgt_d = np.linalg.norm(np.asarray(L['target']) - xy, axis=1)
        ## How much ROUTE did the ball actually cover? Every episode runs the
        ## full step budget, so the question is whether it ran out of plan or ran
        ## out of steps. `plan_frac` is how far the pursuit cursor got along the
        ## current plan; `arc` / `net` are the ball's own motion in wu.
        cur = np.asarray(L['cursor']) if L.get('cursor') else np.zeros(len(xy), int)
        cur_d = np.asarray(L['cursor_dist']) if L.get('cursor_dist') else np.zeros(len(xy))
        n_plan = len(self._ball_plan_xy) if self._ball_plan_xy is not None else 0
        self._ball_ep_summaries.append(dict(
            ep=int(self._ball_ep),
            n_steps=int(len(L['true_xy'])),
            arc_wu=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0,
            net_disp_wu=float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else 0.0,
            speed_wu_per_step=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).mean())
            if len(xy) > 1 else 0.0,
            cursor_max=int(cur.max()) if len(cur) else 0,
            plan_frac=float(cur.max() / max(n_plan - 1, 1)) if len(cur) else 0.0,
            loc_err_mean=float(loc.mean()), loc_err_max=float(loc.max()),
            ## WHEN did the belief go wrong, not just how far. A benchmark run (L3) split
            ## 12 episodes at loc_err_mean 0.36 wu (4/12 success) against 8 at
            ## 8.92 wu with loc_err_max up to 59 wu (0/8) -- but mean/max alone
            ## cannot tell INITIALISATION LOCK-IN (the uniform prior's first MAP
            ## picks an aliased twin and the filter never leaves it) from
            ## MID-EPISODE DIVERGENCE, and both patterns are present there
            ## (ep5 mean 10.6/max 59.1 vs ep4 mean 4.0/max 58.3). `frac_lost` is
            ## the honest summary of the gate; `lost_first_step` is the mechanism.
            ## Threshold 4.0 wu == one maze cell == "in the wrong place".
            loc_err_p50=float(np.median(loc)),
            loc_err_first100_mean=float(loc[:100].mean()),
            loc_err_frac_lost=float((loc > 4.0).mean()),
            loc_lost_first_step=int(np.argmax(loc > 4.0)) if bool((loc > 4.0).any()) else -1,
            ## TRACKING LAG in wu: how far the PID target sits from where the
            ## agent actually is. This -- not the log's `obs_subgl_dist`, which
            ## is a 30-beam SCAN-space L2 against the base loop's own
            ## time-advanced subgoal -- is the quantity that says whether
            ## pursuit is working. Under 'time' it should grow without bound.
            target_dist_mean=float(tgt_d.mean()),
            target_dist_p90=float(np.quantile(tgt_d, 0.9)),
            target_dist_max=float(tgt_d.max()),
            ## CURSOR lag, the quantity the leash bounds. `target_dist_*` above
            ## is the carrot; the carrot can never be pulled nearer than the
            ## cursor, so this is the floor on the whole tracking problem. Read
            ## the two together: a carrot cap with a runaway cursor is inert.
            cursor_dist_mean=float(cur_d.mean()),
            cursor_dist_p90=float(np.quantile(cur_d, 0.9)),
            cursor_dist_max=float(cur_d.max()),
            n_stall_escapes=int(self._ball_stall_escapes - self._ball_ep_esc0),
            n_cursor_reacquires=int(self._ball_cursor_reacquires - self._ball_ep_reacq0),
            belief_conf_mean=float(np.mean(L['belief_conf'])),
            deadlock_step=int(L['deadlock_step']),
            ## does the PLAN end at the goal? -- the only open question once the
            ## executor closed (2026-07-31). Absent when the goal was not
            ## captured, so read with `in`, never with .get(k, default).
            **self._ball_plan_goal_stats(),
        ))
        if self._ball_log_npz:
            try:
                path = os.path.join(self.savepath, f'ball_ep{self._ball_ep}_timeline.npz')
                out = {k: np.asarray(v) for k, v in L.items() if k != 'deadlock_step'}
                ## Plans are ragged (a replan can shorten the horizon), so store
                ## them CSR-style: plans_xy[plans_ptr[i]:plans_ptr[i+1]] is plan
                ## i, which took effect at control step plans_step[i]. Slice the
                ## timeline by plans_step to get the steps that chased each one.
                plans = [p['xy'] for p in self._ball_plans]
                out['plans_xy'] = (np.concatenate(plans, axis=0) if plans
                                   else np.zeros((0, 2)))
                out['plans_ptr'] = np.cumsum([0] + [len(p) for p in plans])
                out['plans_step'] = np.asarray([p['step0'] for p in self._ball_plans], int)
                out['true_goal_xy'] = (self._ball_true_goal_xy
                                       if self._ball_true_goal_xy is not None
                                       else np.zeros(2))
                np.savez_compressed(path, **out)
            except Exception:  # noqa: BLE001
                pass

    # ================= json =================
    def update_j_data(self, json_data):
        """extend the rollout json with the ball executor / ladder metadata"""
        self._ball_flush_ep()                        # last episode
        json_data = super().update_j_data(json_data)
        json_data.update([
            ('ball_executor_arm', 'B_invdyn' if self._ball_arm_b else 'A_decode_pid'),
            ('ball_pos_source', self._pos_source),
            ('ball_pid', dict(kp=self._pid.kp, kd=self._pid.kd, ki=self._pid.ki,
                              ki_zone=self._pid.ki_zone, pos_lpf=self._pid.pos_lpf)),
            ('ball_lookahead', self._ball_lookahead),
            ('ball_decode_stride', self._ball_decode_stride),
            ('ball_decode_mode', self._ball_decode_mode),
            ('ball_decode_rmax', self._ball_decode_rmax),
            ('ball_waypoints', int(self._ball_waypoints)),
            ('ball_wp_calls', int(self._ball_wp_calls)),
            ('ball_wp_fallbacks', int(self._ball_wp_fallbacks)),
            ('ball_wp_scan_seeds', int(self._ball_wp_scan_seeds)),
            ('ball_pursuit', self._ball_pursuit),
            ('ball_pursuit_win', int(self._ball_pursuit_win)),
            ('ball_lookahead_wu', float(self._ball_lookahead_wu)),
            ('ball_stall_steps', int(self._ball_stall_steps)),
            ('ball_stall_escapes', int(self._ball_stall_escapes)),
            ('ball_los', int(self._ball_los)),
            ('ball_los_margin', float(self._ball_los_margin)),
            ('ball_los_pullbacks', int(self._ball_los_pullbacks)),
            ('ball_los_seeks', int(self._ball_los_seeks)),
            ('ball_los_blind', int(self._ball_los_blind)),
            ('ball_cursor_leash_wu', float(self._ball_cursor_leash_wu)),
            ('ball_max_carrot_wu', float(self._ball_max_carrot_wu)),
            ('ball_pursuit_back', int(self._ball_pursuit_back)),
            ('ball_cursor_reacquires', int(self._ball_cursor_reacquires)),
            ('ball_min_carrot_wu', float(self._ball_min_carrot_wu)),
            ('ball_carrot_floor_hits', int(self._ball_carrot_floor_hits)),
            ('ball_frozen_steps', int(self._ball_frozen_steps)),
            ('ball_frozen_eps', float(self._ball_frozen_eps)),
            ('ball_frozen_escapes', int(self._ball_frozen_escapes)),
            ('ball_wall_slide', int(self._ball_wall_slide)),
            ('ball_slide_wu', float(self._ball_slide_wu)),
            ('ball_slides', int(self._ball_slides)),
            ('ball_goal_tol_env', float(getattr(self.env, '_goal_tol', float('nan')))),
            ('ball_grid', dict(xy_step=float(self.loc_grid.xy_step),
                               n_theta=int(self.loc_grid.n_theta),
                               n_poses=int(self.loc_grid.n_poses))),
            ('ball_loc_motion_sigma', float(getattr(self.args, 'loc_motion_sigma', 1.0))),
            ('ball_loc_subcell', int(self._ball_subcell)),
            ('ball_deadlock_cfg', dict(win=self._dl_win, disp=self._dl_disp,
                                       umin=self._dl_umin)),
            ('ball_n_deadlocks', int(sum(1 for s in self._ball_ep_summaries
                                         if s['deadlock_step'] >= 0))),
            ('ball_ep_summaries', self._ball_ep_summaries),
            ('ball_decode_stats', self._ball_decode_stats[-64:]),
        ])
        return json_data
