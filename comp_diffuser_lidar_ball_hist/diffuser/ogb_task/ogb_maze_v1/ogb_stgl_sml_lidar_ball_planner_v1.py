"""
BALL (OGBench pointmaze) LiDAR closed-loop rollout -- the whole executor, in one file.

The planner denoises sequences of 30-beam LiDAR scans; it never receives ``(x, y)``.
This class closes the loop around it:

    scan_t = scanner.scan(x, y, 0.0)                     # 30 beams, 2*pi, yaw == 0
    plan   = policy.gen_cond_stgl(start scan, goal scan, [history])
    xy     = GridLocalizer.localize_sequence(plan[::2])  # forward filter, interpolated
    target = progress cursor -> arc lookahead -> LOS pull-back
    u      = clip(5 * (target - p_hat), -1, 1)           # PID2D, kp=5, kd=ki=0

Position source is the ladder knob (``LIDAR_LOC_SOURCE``):

    'env'    -- the PID closes on the true position. Isolates planner + decode
                quality from perception. ("L1")
    'belief' -- the PID closes on the live histogram filter's MAP estimate, which
                is derived from the observed scans alone. The full True-LiDAR loop
                and the deliverable. ("L3")

There is no third rung here, and that is deliberate. The previous fork's L2
("goal localized from its scan") differed from L1 only through machinery this repo
does not have -- an arrival detector, a BFS navigator, a plan re-ranker -- so in this
control loop L1 and L2 would be the *same rollout*. (Consistently, the fork measured
them as flat: 55.0 vs 55.0, p = 1.0, with 55/100 episodes bit-identical.) Goal-scan
localizability is real but it is an offline property, so it is *logged* per episode
(decoded goal cell, rank of the true goal cell) rather than dressed up as a rung.

What this file deliberately does NOT contain, versus the previous fork: Viterbi plan
decode, BFS per-segment waypoints, goal-band K>1, goal/grid plan re-ranking, arrival
detection + verify-and-hop, the BFS grid navigator, cursor leash / max-carrot /
min-carrot, wall-slide, frozen-step escapes, sub-cell belief interpolation, a
position low-pass filter, and the learned inverse-dynamics arm. It is the vanilla
LiDAR L3 loop plus the three fixes that made it work at all (progress pursuit, LOS
pull-back, belief seeding before plan 0).
"""

import os
from collections import deque

import numpy as np

import diffuser.utils as utils
from diffuser.utils import load_config
from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env
from diffuser.datasets.lidar.lidar_grid import load_or_build_grid
from diffuser.datasets.lidar.lidar_localization import GridLocalizer, scan_distance
from diffuser.datasets.lidar.lidar_arrival import LiveLocalizer
from diffuser.datasets.lidar.pid_controller import PID2D
from diffuser.datasets.lidar.hist_window import left_pad_history
from diffuser.pl_eval.lidar_eval.lidar_grid_common import los_clear as LOS_CLEAR

## Hard velocity clamp of the OGBench point agent, measured directly: full
## throttle in open corridor gives exactly 0.2000 wu/step, sustained. It is the
## ceiling every pace number in `_flush_ep` is expressed against.
BALL_SPEED_CAP_WU = 0.2
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_planner_v1 import (
    OgB_Stgl_Sml_MazeEnvPlanner_V1,
)


class OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1(OgB_Stgl_Sml_MazeEnvPlanner_V1):
    """Closed-loop rollout for the LiDAR pointmaze (ball) planner."""

    # ================= setup =================
    def setup_load(self, ld_config):
        super().setup_load(ld_config)
        args, args_train = self.args, self.args_train
        assert 'pointmaze' in args_train.dataset, args_train.dataset

        ## The planner observation is the 30-D scan, NOT the 2-D (x, y) placeholder
        ## `obs_select_dim=(0, 1)` that the LiDAR config carries for API shape.
        ## dfu_ndim sizes the fused-plan buffer, so it must be the real plan width.
        self.dfu_ndim = self.diffusion.observation_dim
        self.act_control = 'pid_lidar'

        ## the sensor: same geometry that produced the training scans
        self.lidar_config = args_train.dataset_config['lidar_config']
        self.n_beams = int(self.lidar_config['n_beams'])
        assert self.dfu_ndim == self.n_beams, \
            f'planner plan dim ({self.dfu_ndim}) must equal n_beams ({self.n_beams})'
        self.scanner = make_scanner_for_env(self.env.name, self.lidar_config)

        ## the reference grid + localizer: scan -> (x, y). The ball has no heading,
        ## so n_theta MUST be 1 -- the whole point of the pivot to the point mass.
        self.loc_grid = load_or_build_grid(
            self.env.name, self.lidar_config,
            xy_step=float(getattr(args, 'loc_grid_xy_step', 0.5)),
            n_theta=int(getattr(args, 'loc_grid_n_theta', 1)),
            cache_root=getattr(args, 'loc_grid_cache', 'data/ogb_maze/lidar_grid'))
        assert self.loc_grid.n_theta == 1, (
            f'ball grid must be built with n_theta=1 (got {self.loc_grid.n_theta}); '
            f'set LIDAR_LOC_NTHETA=1')
        self.loc = GridLocalizer(
            self.loc_grid, sigma=float(getattr(args, 'loc_grid_sigma', 1.0)))

        ## ---- position source: the ladder knob ----
        self._pos_source = str(getattr(args, 'loc_source', 'env')).lower()
        assert self._pos_source in ('env', 'belief'), self._pos_source
        self._loc_motion_sigma = float(getattr(args, 'loc_motion_sigma', 1.0))
        ## eps_mix: uniform floor in the predict step. `loc_err_p50` is bimodal (0.25 wu
        ## on 86% of episodes, ~45 wu on 14%) -- a wrong-hypothesis latch the pure
        ## product update cannot escape. 0.0 = off = bit-identical.
        self._loc_eps_mix = float(getattr(args, 'loc_eps_mix', 0.0))
        self._live = LiveLocalizer(self.loc, motion_sigma_cells=self._loc_motion_sigma,
                                   theta_blur=0, eps_mix=self._loc_eps_mix)

        ## ---- executor ----
        ## kp=5 is upstream's own pointmaze P-law, bit for bit. kd/ki stay 0 and the
        ## position low-pass stays 0: the PID saturates on ~90% of control steps, so
        ## there is no gain here to tune, and the low-pass was itself convicted as the
        ## single cause of the fork's "L3 localization failures" (it is a belief<->
        ## control feedback loop, invisible open-loop).
        self._pid = PID2D(kp=float(getattr(args, 'ball_kp', 5.0)), kd=0.0, ki=0.0,
                          pos_lpf=0.0)

        ## ---- pursuit ----
        ## progress cursor: nearest plan index at or ahead of the ball, searched in a
        ## bounded forward window. The time parameterisation (index += 1 per control
        ## step) runs away, because the plan is generated at DATA pace.
        self._pursuit_win = int(getattr(args, 'ball_pursuit_win', 60))
        ## the carrot sits this far ALONG the plan (arc length) from the cursor.
        ## An index lookahead is pace-dependent and ~0.4 wu, which makes the ball crawl.
        self._lookahead_wu = float(getattr(args, 'ball_lookahead_wu', 2.0))
        self._los_margin = float(getattr(args, 'ball_los_margin', 0.6))
        ## Stalled control steps before the SAFE (line-of-sight) recovery is attempted.
        ## 50 was measured as the loop's largest budget sink (see `_advance_cursor`);
        ## 5 frees ~35% of the step budget on the periodic arms, ~78% on ada_dist.
        self._stall_steps = int(getattr(args, 'ball_stall_steps', 5))
        ## Stalled steps before the UNSAFE blind `cursor+1` walk. Deliberately still 50:
        ## this one advances with nothing visible and desyncs the cursor if run hot.
        self._blind_walk_steps = int(getattr(args, 'ball_blind_walk_steps', 50))
        ## ---- LOS shortcut: the arc the ball never needed to drive ----
        ## Measured 2026-09-03 on the 80 dumped r24/hist16 plans: the decoded plan is
        ## **315 wu of arc** (p50 311) against a **200 wu** physical budget (0.2 wu/step
        ## x 1000), so **77 of 80 plans cannot be driven at all**, and a flawless
        ## never-stalling ball finishes only 64% of the median plan. That is why the
        ## bench sits at ~0 with every mechanism verified working.
        ##
        ## The plan wanders (6.2x the direct distance, p50) but stays in FREE SPACE, so
        ## most of the arc is removable by line-of-sight string-pulling. Greedy pull
        ## with the same `los_clear` this loop already calls, over the same 80 plans:
        ##     shipped (no shortcut)   p50 arc 312 wu    4/80 fit the budget
        ##     shortcut win=200        p50 arc 163 wu   54/80 fit
        ##     shortcut win=888        p50 arc  80 wu   80/80 fit
        ## The pulled path is 1.7x direct -- a plausible maze route, not a wall-cut.
        ##
        ## This is a SEPARATE window from `pursuit_win` on purpose. Widening
        ## `pursuit_win` would also widen the nearest-point snap above, which is the
        ## documented cursor-teleport risk ("a plan that revisits a corridor"). The
        ## shortcut is safe by construction: `_los_seek` can only ever return a point
        ## the ball can drive at in a STRAIGHT LINE from where it actually is.
        ## 0 disables it, which is bit-identical to the pre-2026-09-03 executor.
        self._los_shortcut_win = int(getattr(args, 'ball_los_shortcut_win', 0))
        ## Candidate stride for the shortcut probe. The window can be the whole plan,
        ## and `los_clear` raycasts every candidate every control step; a stride of 4
        ## costs ~1/4 the rays and loses at most 4 plan indices (~1.4 wu of arc) of
        ## shortcut. Never applied to `_los_seek`'s stall recovery, which is narrow.
        self._los_shortcut_stride = max(int(getattr(args, 'ball_los_shortcut_stride', 4)), 1)
        ## ---- plan string-pull (the corrected form of the shortcut) ----
        ## 0 disables it, which is bit-identical to the pre-2026-09-03 executor.
        self._pull_win = int(getattr(args, 'ball_pull_win', 0))
        self._pull_stride = max(int(getattr(args, 'ball_pull_stride', 4)), 1)
        self._pull_resample_wu = float(getattr(args, 'ball_pull_resample_wu', 0.4))
        ## ---- the fork's two executor guards (comp_diffuser_ball) ----
        ## v2 never adopted these and reproduced the exact failure they fix in the
        ## benchmark runs (cursor ran away from the ball, carrot_dist_p50 1.95 -> 6.4).
        ##   max_carrot_wu -- cap the carrot by STRAIGHT-LINE distance, not just by
        ##     plan ARC. Arc is not distance: the decoded plan carries ~25 wall-
        ##     crossing breaks per 888 frames, so 2 wu of arc can be 50 wu of real
        ##     distance, and a saturated PID then drives at a point across the maze.
        ##   cursor_leash_wu -- never blind-march the cursor to a point further than
        ##     this from the ball; re-acquire BACKWARD over `pursuit_back` indices
        ##     instead. That is the only place the monotone cursor may retreat.
        ## Both 0 = OFF = bit-identical to the shipped executor.
        self._max_carrot_wu = float(getattr(args, 'ball_max_carrot_wu', 0.0))
        self._cursor_leash_wu = float(getattr(args, 'ball_cursor_leash_wu', 0.0))
        self._pursuit_back = int(getattr(args, 'ball_pursuit_back', 15))
        self._cursor_reacquires = 0
        ## ---- adaptive composition depth ----
        ## The fork clips n_comp to the ROUTE LENGTH; v2 is fixed at 8, so a 9 wu hop
        ## gets the same ~300 wu plan as a 63 wu route. Sized here from the BELIEF
        ## position and the planner's OWN decoded goal scan -- no BFS, no map route,
        ## nothing the loop was not already given. 0 = OFF.
        self._ada_ncomp = int(getattr(args, 'ball_ada_ncomp', 0))
        self._ada_ncomp_cells = float(getattr(args, 'ball_ada_ncomp_cells', 3.4))
        self._ncomp_used = []
        ## ---- goal capture: the terminal 12 wu ----
        ## Measured 2026-09-04 over every bench arm ever run: ~64 of 300 episodes come
        ## within 3 cells of the goal and 3-6 of them convert, and that `<12wu` count is
        ## 61-67 in EVERY arm -- unmoved by history, cadence, stall timers, carrot caps,
        ## the LOS shortcut or ada_ncomp. The agent is not lost when it gets there: the
        ## goal decoded from the goal SCAN alone is 0.26 wu p50 (true cell ranked #1 in
        ## 79% of episodes) and the belief is 0.25 wu p50 on 86% of episodes.
        ##
        ## What loses it is mechanical: `cursor_max` p50 is **215 of 888**, so the ball
        ## is beside the goal while the pursuit cursor aims at plan[217] somewhere else,
        ## and it leaves -- `final_goal_dist` p50 42 wu against `min_goal_dist` p50 36,
        ## closest approach at step ~400 of 1000. It drives PAST the goal chasing the
        ## early sixth of a wandering plan.
        ##
        ## So: when the agent's OWN decoded goal is within `capture_wu` AND in clear line
        ## of sight, stop chasing plan indices and drive straight at it.
        ##
        ## This is STEERING, NOT SELF-REPORTING. It never declares success -- the env
        ## does, exactly as before -- so it is not the `LidarArrivalDetector` /
        ## verify-and-hop rule that `lidar_arrival.py` deliberately omits. It reads only
        ## what the loop already holds: `_pos()` (the BELIEF estimate at L3), the argmin
        ## decode of the goal scan the task handed over, and `los_clear`, which the
        ## shipped executor already raycasts every control step for the LOS pullback and
        ## the stall recovery. No Viterbi, no BFS, no map route, no true coordinate.
        ## It carries the same caveat `ada_ncomp` does: it consumes a decoded goal cell,
        ## so the "there is no L2 here" note does not hold for a capture run.
        ## 0 = OFF = bit-identical to the shipped executor.
        self._capture_wu = float(getattr(args, 'ball_capture_wu', 0.0))
        ## Hysteresis, on DISTANCE ONLY. Releasing on transient loss of line of sight
        ## would flicker at every corner the ball rounds while closing.
        self._capture_release_wu = float(
            getattr(args, 'ball_capture_release_wu', 0.0)) or (1.5 * self._capture_wu)
        ## ---- goal re-entry: the tier that makes capture reachable at all ----
        ## Capture alone is nearly useless, and a 2-minute CPU census says why.
        ## Fraction of FREE positions at a given distance from the goal that can SEE it
        ## (all 5 eval goals, r24 grid, the same `los_clear` this loop calls):
        ##
        ##   dist   0-2   2-4   4-6   6-8  8-12  12-16  16-20 wu
        ##   ALL   1.00  0.74  0.63  0.37  0.19   0.07   0.01
        ##   blk4  1.00  0.68  0.61  0.30  0.06   0.00   0.00
        ##
        ## So at the 12 wu shelf where 64/300 episodes sit, the goal is visible from
        ## only ~19% of positions -- 6% on block 4. Euclidean proximity is NOT
        ## reachability in a maze: the ball is 9 wu away AROUND A CORNER. A smoke on
        ## block 4 confirmed it: 0 engagements in 4 episodes, while sitting
        ## 8-9.7 wu from a goal it decoded to 0.2 wu with the true cell ranked #1.
        ##
        ## What gets through the corner is the PLAN -- and its approach to the goal is
        ## never executed, because `cursor_max` p50 is 215 of 888. So: when the ball is
        ## within `reentry_wu` of its decoded goal but cannot see it, snap the cursor
        ## FORWARD into the plan's own goal region and pursue normally from there. The
        ## plan's last frame decodes within 1 wu of the goal on 92% of episodes, so
        ## forward pursuit from that region heads at the goal through the plan's corner.
        ##
        ## This is NOT the refuted FIX 5 shortcut, and the difference is the guard:
        ## FIX 5 jumped to the furthest index VISIBLE FROM THE BALL, anywhere in the
        ## plan, every step -- so `carrot_dist_p50` went 1.95 -> 6.4 and cursor progress
        ## became a phantom. This jumps only into the GOAL REGION, only while the ball
        ## is already within `reentry_wu` of the goal, lands on the index NEAREST THE
        ## BALL, and is refused unless that index is closer to the ball than the cursor
        ## already is. `carrot_dist_p50` is bounded by construction, and stays a gate.
        ## 0 = OFF.
        ## ---- capture's OWN line-of-sight margin ----
        ## `los_margin = 0.6` is refusing sightlines the ball could actually drive.
        ## Fraction of free positions that can see the goal, by margin (CPU census over
        ## all 5 eval goals, r24 grid):
        ##
        ##   margin   0-4   4-6   6-8  8-12  12-16 wu
        ##     0.6   0.831 0.628 0.366 0.186  0.075
        ##     0.4   0.936 0.860 0.457 0.237  0.097
        ##     0.0   1.000 1.000 0.559 0.297  0.141
        ##
        ## At 0.6 it blocks 17% of sightlines at POINT-BLANK range and a third of them
        ## at 4-6 wu, which is exactly the band where capture converts 11/11. Loosening
        ## it is worth +25-37% of capture opportunity.
        ##
        ## Deliberately a SEPARATE knob from `_los_margin`: that one also governs the
        ## carrot pullback and the stall recovery, so lowering it globally would change
        ## the whole controller and confound the arm. Defaults to `_los_margin`, which
        ## is bit-identical to the shipped loop.
        self._capture_margin = float(getattr(args, 'ball_capture_margin', -1.0))
        if self._capture_margin < 0:
            self._capture_margin = self._los_margin
        ## Safety valve for the looser margin: a capture that clips a corner would aim
        ## the saturated PID at a wall, and release is on DISTANCE only, so nothing
        ## would ever free it. Release after this many captured steps without the
        ## distance to the goal falling. 0 = off.
        self._capture_stuck_steps = int(getattr(args, 'ball_capture_stuck_steps', 0))
        self._capture_releases_stuck = 0
        self._reentry_wu = float(getattr(args, 'ball_goal_reentry_wu', 0.0))
        self._reentry_hops = 0
        self._reentry_ep_hops = 0
        self._captured = False
        ## engagements and EPISODES are different: capture can release and re-engage
        ## within one episode, and the pre-registered gate is stated in episodes.
        self._capture_engagements = 0
        self._capture_episodes = 0
        self._capture_steps = 0
        self._capture_near_min = float('inf')
        self._capture_in_range_steps = 0
        self._capture_blocked_steps = 0
        ## plan decode: every `decode_stride`-th frame is localized, then interpolated
        self._decode_stride = max(int(getattr(args, 'ball_decode_stride', 2)), 1)
        ## which decoder turns the scan plan into (x, y). See `_decode_plan_xy`.
        self._decode_mode = str(getattr(args, 'ball_decode', 'argmin')).lower()
        assert self._decode_mode in ('argmin', 'filter', 'viterbi'), (
            f'unknown LIDAR_DECODE={self._decode_mode!r}; expected '
            f'argmin|filter|viterbi. An unrecognised decoder must never fall back '
            f'silently -- the choice changes every downstream number.')
        ## ---- viterbi decode: the globally-consistent option ----
        ## argmin is memoryless and filter's motion model is a Gaussian on the lattice,
        ## which cannot express "stay in the corridor you are in". Both therefore fail
        ## G2a's jump bar (RETRAIN_2026-08-29 §6). `viterbi` keeps the same emission
        ## model and replaces the motion model with a hard free-space connectivity
        ## constraint. The graph is derived from `loc_grid`, which is already built.
        self._vit_max_step = float(getattr(args, 'ball_decode_max_step', 3.0))
        self._vit_beam = int(getattr(args, 'ball_decode_beam', 2048))
        self._vit_w_step = float(getattr(args, 'ball_decode_w_step', 0.0))
        self._vit_sigma_e = float(getattr(args, 'ball_decode_sigma_e', 1.0))
        self._vit_graph = None      # built lazily on first decode, then reused
        self._decode_motion_sigma = float(
            getattr(args, 'ball_decode_motion_sigma',
                    getattr(args, 'loc_motion_sigma', 1.0)))
        ## argmin is memoryless, so an aliased frame teleports the decode. Left raw
        ## that inflates the plan's ARC LENGTH 4.3x (918 wu vs 213), and `_carrot_idx`
        ## advances by 2.0 wu of arc -- so the carrot would barely move and the ball
        ## would crawl. A median (not a mean) removes isolated spikes without smearing
        ## corners. Measured over 40 real plans: width 15 gives arc 326 wu,
        ## frac_cell_jumps 0.027 (raw 0.077), free_frac 0.979, endpoint unchanged.
        self._decode_median = max(int(getattr(args, 'ball_decode_median', 15)), 1)

        ## ---- FIX 11: confidence-triggered replan (thr 0 = off = bit-identical) ----
        ## The argmin decode scores every plan frame against every free pose and keeps
        ## only the winner. The rest is a free confidence: the MARGIN to the best pose
        ## more than one cell away. A small margin is a near-tie with a far lookalike,
        ## which is the aliased frame a teleport goes through. decode_confidence_census
        ## (2026-10-18, 80 r24 plans): AUC 0.92 for break frames, and a run of >=3
        ## decoded frames under 0.3 starts at or before the plan's FIRST wall-crossing
        ## break in 72/80 plans. REPAIRING those frames is refuted (a time-interpolated
        ## refill cuts walls: breaks 24.8 -> 38-51). This only DETECTS them: once the
        ## cursor reaches the first flagged run, replan from where the ball is instead
        ## of thrashing at the break until the next scheduled replan. No map graph.
        self._conf_repl_thr = float(getattr(args, 'ball_conf_repl_thr', 0.0))
        self._conf_repl_run = max(int(getattr(args, 'ball_conf_repl_run', 3)), 1)
        ## Minimum control steps since the last plan before the trigger may fire, and
        ## the first plan frame a flagged run may start at. Without it an ambiguous
        ## plan HEAD would refire on every step -- the ada_dist latch (RETRAIN §8.4).
        self._conf_repl_min_gap = max(int(getattr(args, 'ball_conf_repl_min_gap', 20)), 1)
        ## replans added to the periodic budget, so these cannot starve the schedule
        self._conf_repl_extra = max(int(getattr(args, 'ball_conf_repl_extra', 10)), 0)
        if self._conf_repl_thr > 0:
            assert self._decode_mode == 'argmin', (
                'LIDAR_BALL_CONF_REPL_THR reads the ARGMIN distance array; '
                f'decode={self._decode_mode} has no such margin')
            assert self._pull_win == 0, (
                'the flag is a plan INDEX; the string-pull resamples the plan and would '
                'misalign it -- run CONF_REPL with PULL_WIN=0')
            assert self.is_replan == 'periodic', (
                'the confidence trigger rides on the periodic loop; with REPL_EVERY=0 '
                '(ada_dist) it would silently never fire')
        self._plan_flag_idx = None      # plan index of the current plan's first flagged run
        self._conf_repls = 0            # all fires, all episodes
        self._conf_ep_repls = 0         # fires this episode
        self._conf_ep_flags = []        # this episode: flag index per decoded plan (-1 = none)
        self._conf_ep_fire_steps = []   # this episode: control step of each fire
        self._conf_ep_flag_lead = []    # this episode: cursor - flag at each fire

        ## ---- observation history ----
        ## hist_len comes from the CHECKPOINT (the trained encoder's window), never
        ## from a flag, so a history run can never feed a length the model never saw.
        ## `ball_hist` then selects the ARM: 1 -> feed the observed scans, 0 -> drop
        ## them, which is the no-history control the same weights also serve.
        self.hist_len = int(getattr(self.diffusion.model, 'hist_len', 0))
        self._hist_on = bool(int(getattr(args, 'ball_hist', 0))) and self.hist_len > 0
        if int(getattr(args, 'ball_hist', 0)) and self.hist_len == 0:
            raise ValueError(
                'LIDAR_BALL_HIST=1 but this checkpoint has no history encoder '
                '(network_config["hist_model_config"] was None at training time). '
                'Refusing to run a history arm against a no-history model.')
        self._hist = deque(maxlen=max(self.hist_len, 1))

        ## ---- per-episode state ----
        self._plan_xy = None        # (H, 2) decoded targets of the current plan
        self._ada_no_plan = 0       # ada_dist calls with no decoded plan (see _get_ada_dist)
        self._ada_dist_last = float('nan')
        self._new_plan = False
        self._cursor = 0
        self._stall = 0
        self._stall_escapes = 0
        self._blind_walks = 0
        self._los_pullbacks = 0
        self._los_blind = 0
        self._pull_stats = []          # (arc_before, arc_after, n_before, n_after)
        self._los_shortcuts = 0        # times the shortcut advanced the cursor
        self._los_shortcut_gain = 0    # total plan indices it skipped
        self._obs_step = -1         # guards the once-per-step sensing
        self._goal_scan_cur = None
        self._true_goal_xy = None   # LOG ONLY
        self._ep_log = None
        self._ep_summaries = []
        self._decode_stats = []
        ## ---- replan probe (LIDAR_BALL_PROBE=1) ----
        ## OFF by default so a bench run is bit-identical without it. Answers three
        ## questions the existing diagnostics cannot: does the history tensor reach
        ## the model, how degenerate is the deque at each replan, and is the forward
        ## filter's decode error concentrated at the plan TAIL (which replanning
        ## discards) rather than the HEAD (which the executor pursues).
        self._probe_on = bool(int(os.environ.get('LIDAR_BALL_PROBE', 0)))
        self._replan_probe = []

        ## decode every produced plan to (x, y) once, and inject the history: both
        ## must happen where n_comp is already final, i.e. around the actual call.
        self._real_gen = self.policy.gen_cond_stgl
        self.policy.gen_cond_stgl = self._gen_cond_stgl

        utils.print_color(
            f'[ball] pos_source={self._pos_source} kp={self._pid.kp} '
            f'hist={"on" if self._hist_on else "off"}(len={self.hist_len}) '
            f'pursuit_win={self._pursuit_win} lookahead_wu={self._lookahead_wu} '
            f'stall_steps={self._stall_steps} blind_walk_steps={self._blind_walk_steps} '
            f'los_shortcut_win={self._los_shortcut_win}'
            f'(stride={self._los_shortcut_stride}) '
            f'pull_win={self._pull_win}(stride={self._pull_stride},'
            f'resample={self._pull_resample_wu}) '
            f'max_carrot={self._max_carrot_wu} leash={self._cursor_leash_wu}'
            f'(back={self._pursuit_back}) ada_ncomp={self._ada_ncomp} '
            f'capture_wu={self._capture_wu}'
            f'(release={self._capture_release_wu}) '
            f'reentry_wu={self._reentry_wu} '
            f'capture_margin={self._capture_margin}'
            f'(stuck={self._capture_stuck_steps}) '
            f'loc_eps_mix={self._loc_eps_mix} '
            f'conf_repl_thr={self._conf_repl_thr}(run={self._conf_repl_run},'
            f'min_gap={self._conf_repl_min_gap},extra={self._conf_repl_extra}) '
            f'decode_stride={self._decode_stride} '
            f'grid={self.loc_grid.n_poses} poses '
            f'scanner={self.scanner.config_dict()}', c='g')

    def _setup_normalizers(self):
        """Rebuild the PLANNER's own dataset from its saved ``dataset_config.pkl``.

        The observation IS the scan, so there is exactly one normalizer and the
        base class's `train` / `full` split degenerates. The LiDAR ray-cast is
        disk-cached, so this costs seconds after the first run."""
        run_dir = os.path.join(self.args.logbase, self.args_train.dataset,
                               self.args_train.exp_name)
        utils.print_color(
            f'[ball] rebuilding planner dataset normalizer from {run_dir}', c='c')
        dset = load_config(run_dir, 'dataset_config.pkl')()
        self.train_normalizer = dset.normalizer
        self.full_normalizer = dset.normalizer   # same object: obs == the scan
        del dset

    def check_is_same_nmlizer(self):
        assert self.train_normalizer is self.full_normalizer

    # ================= sensing =================
    def _obs_to_scan(self, obs):
        """Live env observation -> the 30-D world-frame scan the model was trained on."""
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        return self.scanner.scan(float(obs[0]), float(obs[1]), 0.0).astype(np.float32)

    def _observe(self, obs_cur, i_et):
        """Once-per-control-step sensing: push the scan onto the history and advance
        the belief filter. Idempotent within a step, so it is safe to call from both
        the per-step hook and the (re)plan hook."""
        if i_et is not None and i_et == self._obs_step:
            return self._hist[-1]
        scan = self._obs_to_scan(obs_cur)
        self._hist.append(scan)
        if self._pos_source == 'belief':
            r = self._live.update(scan)
            self._belief_xy = np.asarray(r['est_xy'], dtype=np.float64)
            self._belief_conf = float(r['conf'])
        if i_et is not None:
            self._obs_step = i_et
        return scan

    def _pos(self):
        if self._pos_source == 'belief' and getattr(self, '_belief_xy', None) is not None:
            return self._belief_xy
        return np.asarray(self.env.get_xy(), dtype=np.float64)

    def _goal_scan(self, gl_pos):
        """Goal POSITION -> the goal scan. This is how the task is handed to the
        agent: not as a coordinate, but as what the goal looks like. yaw == 0."""
        return self.scanner.scan(float(gl_pos[0]), float(gl_pos[1]), 0.0).astype(np.float32)

    # ================= base-class hooks =================
    def _ep_reset_hook(self, obs_cur, gl_pos):
        """New episode: clear the belief and the history, then SEED both from the
        start scan -- before plan 0 is sampled. In the previous fork the belief was
        refreshed by a step that ran after the first plan, so plan 0 of 100% of L3
        episodes was sampled with no localization at all; fixing that was worth +15
        points. Doing it in a reset hook makes the ordering structural."""
        self._flush_ep()          # close out the PREVIOUS episode's diagnostics
        self._live.reset()
        self._hist.clear()
        self._obs_step = -1
        self._plan_xy = None
        self._new_plan = False
        self._cursor = 0
        self._stall = 0
        self._belief_xy = None
        self._belief_conf = 0.0
        self._ada_no_plan = 0
        self._ada_dist_last = float('nan')
        self._true_goal_xy = np.asarray(gl_pos[:2], dtype=np.float64)
        self._goal_scan_cur = self._goal_scan(gl_pos)
        self._cursor_reacquires = 0
        ## Where the agent BELIEVES the goal is, decoded from the goal SCAN it was
        ## handed -- the same `localize_single` the decode-failure fallback already
        ## uses. Only `ball_ada_ncomp` consumes it; with that off nothing reads it,
        ## and the loop is unchanged.
        ##
        ## NOTE this makes the control loop consume a decoded goal cell for the first
        ## time, so the standing "there is no L2 here" note in plan_ball_bench100 no
        ## longer holds for an ada_ncomp run: goal-decode error becomes a real error
        ## source. It is NOT privileged -- the task hands the agent a goal scan and
        ## decoding it is the agent's own inference (`_goal_scan_diag` scores that
        ## decode at 92% within 1 wu) -- but it must be stated, not assumed.
        self._goal_xy_est = None
        self._captured = False
        self._capture_first_step = -1
        self._capture_dist_at_engage = float('nan')
        self._capture_step_cnt = 0
        self._capture_near_min = float('inf')
        self._capture_in_range_steps = 0
        self._capture_blocked_steps = 0
        self._capture_best_d = float('inf')
        self._capture_no_gain = 0
        self._reentry_ep_hops = 0
        self._plan_flag_idx = None
        self._conf_ep_repls = 0
        self._conf_ep_flags = []
        self._conf_ep_fire_steps = []
        self._conf_ep_flag_lead = []
        if self._ada_ncomp or self._capture_wu > 0 or self._reentry_wu > 0:
            try:
                self._goal_xy_est = np.asarray(
                    self.loc.localize_single(
                        np.asarray(self._goal_scan_cur, dtype=np.float32))['xy'],
                    dtype=np.float64)
            except Exception as e:  # noqa: BLE001
                utils.print_color(f'[ball goal decode] failed ({e}) -> keeping the '
                                  f'configured n_comp, capture disabled this episode',
                                  c='y')

        start_scan = self._observe(obs_cur, i_et=None)
        ## left-pad with the start scan: at t=0 the agent genuinely has no past, and
        ## repeating the first frame is exactly the padding the training windows use.
        ## Built with the SAME helper the dataset uses, so the two cannot drift apart.
        self._hist.clear()
        self._hist.extend(left_pad_history(np.empty((0, self.n_beams), dtype=np.float32),
                                           self._hist.maxlen, fill=start_scan))

        self._ep_log = dict(true_xy=[], pos_used=[], target=[], cursor=[],
                            u_norm=[], loc_err=[], belief_conf=[],
                            goal_decode=self._goal_scan_diag(gl_pos))

        ## the first per-step sensing call of this episode must NOT re-observe the
        ## start scan: it is already in the deque and already in the belief, and
        ## folding the same measurement in twice would sharpen the posterior on
        ## nothing. _next_step_idx() hands out this id next.
        self._obs_step = getattr(self, '_step_counter', -1) + 1

    def _get_invdyn_obs_nm(self, obs_cur):
        """Reused as the per-step sensing hook: the base loop calls this once at the
        top of every control step, before the replan branch and before the executor.
        There is no inverse-dynamics model in this repo, so the returned value is
        only the normalized scan (kept for interface compatibility)."""
        scan = self._observe(obs_cur, i_et=self._next_step_idx())
        return self.train_normalizer.normalize(scan, 'observations')

    def _next_step_idx(self):
        """The base loop does not pass i_et to _get_invdyn_obs_nm, so derive a
        monotone step id: this hook is called exactly once per control step."""
        self._step_counter = getattr(self, '_step_counter', -1) + 1
        return self._step_counter

    def _get_planner_stgl_input(self, obs_cur, gl_pos):
        """Condition the planner on (current scan, goal scan). No position, ever."""
        scan = self._hist[-1] if len(self._hist) else self._observe(obs_cur, None)
        if self._goal_scan_cur is None:
            self._goal_scan_cur = self._goal_scan(gl_pos)
        return scan, self._goal_scan_cur

    def _get_subgoal_log_dist(self, obs_cur, goal_cur):
        """Progress metric for the periodic log line: scan-space distance between
        what the ball sees and the waypoint scan it is chasing."""
        return float(np.linalg.norm(self._obs_to_scan(obs_cur) - np.asarray(goal_cur)))

    def _get_ada_dist(self, obs_cur, goal_cur, wp_idx):
        """`ada_dist` in WORLD units, from decoded cells only.

        The base formula slices the raw obs and the plan waypoint with the same
        indices, which means "position vs subgoal position" only when the plan is
        in (x, y). Here the plan is a 30-beam scan and `obs_select_dim` is the
        (0, 1) placeholder the LiDAR config carries for API shape, so the base
        version compared world (x, y) against beams 0 and 1 -- a quantity that
        essentially always exceeds `thres=1`. Measured: every ada episode replanned
        at steps [1..10] and then ran open-loop for the rest of the budget.

        The replacement keeps the paper's SEMANTICS -- schedule deviation, "am I
        where the plan says I should be by now" -- and reads it off the decoded
        plan, i.e. the same `GridLocalizer` output the PID already closes on. No
        privileged position enters: `self._pos()` is the belief estimate and
        `self._plan_xy` is the decoded plan.

        Deliberately indexed by `wp_idx`, NOT by the pursuit cursor. `_advance_cursor`
        picks the cursor as an argmin of distance over a forward window, so
        ||pos - plan[cursor]|| is near zero by construction and a cursor-based
        trigger could never fire.

        Decode failure -> 0.0, i.e. cond_1 stays silent and plan exhaustion
        (cond_2) remains the only trigger. Counted, never silent."""
        plan = self._plan_xy
        if plan is None or not len(plan):
            self._ada_no_plan += 1
            return 0.0
        i = int(min(wp_idx, len(plan) - 1))
        d = float(np.linalg.norm(np.asarray(self._pos(), dtype=np.float64) - plan[i]))
        self._ada_dist_last = d
        return d

    # ================= plan generation + decode =================
    def _gen_cond_stgl(self, g_cond=None, b_s=None, **kw):
        """Wrapper around ``policy.gen_cond_stgl``: attach the observation history
        on the way in, decode the produced scan plan to (x, y) on the way out."""
        if g_cond is not None and self._hist_on:
            ## (n_probs=1, hist_len, n_beams), UNNORMALIZED -- the policy normalizes
            ## it with the same normalizer/key as the start/goal scans.
            g_cond['hist'] = np.asarray(self._hist, dtype=np.float32)[None]
        if self._probe_on:
            self._probe_rec = self._probe_hist_rec(g_cond)
        ## ---- adaptive composition depth ----
        ## Restored under try/finally so a raising sampler can never leave the policy
        ## on a clipped n_comp for every later episode.
        n_prev = getattr(self.policy, 'n_comp', None)
        try:
            if self._ada_ncomp and n_prev:
                nc = self._ada_ncomp_for_now(n_prev)
                if nc is not None:
                    self.policy.n_comp = nc
                    self._ncomp_used.append(int(nc))
            m_out = self._real_gen(g_cond=g_cond, b_s=b_s, **kw)
        finally:
            if n_prev is not None:
                self.policy.n_comp = n_prev
        try:
            ## the ladder-appropriate current pose, for the viterbi start anchor;
            ## `_pos()` returns the BELIEF at L3, so this leaks nothing.
            try:
                self._decode_anchor_xy = self._pos()
            except Exception:  # noqa: BLE001  -- offline, no env
                self._decode_anchor_xy = None
            ## a failed decode must not leave the PREVIOUS plan's flag armed
            self._plan_flag_idx = None
            self._plan_xy = self._decode_plan_xy(np.asarray(m_out.pick_traj))
            self._new_plan = True
            if self._probe_on:
                self._probe_flush(np.asarray(m_out.pick_traj))
            utils.print_color(
                f'[ball decode] plan -> {len(self._plan_xy)} xy targets '
                f'(hist={"on" if self._hist_on else "off"})', c='c')
        except Exception as e:  # noqa: BLE001
            self._plan_xy = None
            utils.print_color(
                f'[ball decode] FAILED ({e}) -> per-step single-scan decode', c='y')
        return m_out

    def _ada_ncomp_for_now(self, n_max):
        """Composition depth for the plan about to be sampled, from the distance the
        ball still has to cover.

        Fixed `n_comp=8` emits 888 frames -- about 300 wu of plan arc -- for EVERY
        problem, whether the goal is 9 wu away or 63. The fork clipped this to the
        route length and scored 77%/90% on the two short blocks where v2 scores ~0.

        The distance is measured between two things the loop already holds and is
        already allowed to use: `self._pos()` (the BELIEF estimate at L3) and the
        argmin decode of the GOAL SCAN it was handed. **No BFS, no map route, no true
        coordinate** -- this is deliberately not the fork's `_nav_subgoal_band`, which
        clipped along a BFS route through the true occupancy map.

        Straight-line distance UNDERSTATES a maze route, so this only ever clips a
        plan that is far longer than even a generous detour needs. Returns None when
        the goal scan has not been decoded yet, in which case the caller keeps the
        configured depth.
        """
        gl = getattr(self, '_goal_xy_est', None)
        if gl is None:
            return None
        try:
            pos = np.asarray(self._pos(), dtype=np.float64)
        except Exception:  # noqa: BLE001
            return None
        dist_cells = float(np.linalg.norm(pos - np.asarray(gl, dtype=np.float64))) \
            / float(self.scanner.maze_unit)
        nc = int(np.clip(int(np.ceil(dist_cells / self._ada_ncomp_cells)) + 1,
                         2, int(n_max)))
        utils.print_color(
            f'[ball ada_ncomp] {dist_cells:.1f} cells to goal -> n_comp={nc} '
            f'(of {int(n_max)})', c='g')
        return nc

    @staticmethod
    def _median_dejitter(xy, w):
        """Rolling median over the decoded poses. `w<3` is a no-op.

        Edge-padded so the endpoints survive -- `plan[-1]` is the goal anchor and
        `plan[0]` is what `head_err_vs_true` reads.
        """
        if w < 3 or len(xy) < w:
            return xy
        pad = np.pad(xy, ((w // 2, w // 2), (0, 0)), mode='edge')
        return np.stack(
            [np.median(np.lib.stride_tricks.sliding_window_view(pad[:, k], w), axis=-1)
             for k in (0, 1)], axis=1)

    def _decode_viterbi(self, sub):
        """Strided scans -> poses, via beam-Viterbi over the free-space graph.

        Both anchors are information the planner already holds, never the benchmark's
        coordinates:

        * START -- `self._pos()`, i.e. the belief estimate at L3 and the env pose at L1.
          Whatever the ladder rung grants, and nothing more.
        * END -- the per-frame argmin decode of the plan's LAST scan. The task is handed
          to the agent as a goal *scan*, not a coordinate (`_goal_scan`), and the model
          reproduces that scan to ~1e-7; argmin reads it correctly on 97.5% of problems
          (G2b). Connectivity and destination are carried by different evidence -- the
          graph carries the route, the goal scan carries the endpoint -- and an
          unanchored Viterbi trades the second away for the first.
        """
        from diffuser.datasets.lidar.lidar_viterbi import (
            FreeSpaceGraph, viterbi_decode_xy)
        if self._vit_graph is None or self._vit_graph.max_step != self._vit_max_step:
            self._vit_graph = FreeSpaceGraph(self.loc.grid, max_step=self._vit_max_step)
            utils.print_color(f'[ball decode] {self._vit_graph}', c='c')
        ## end anchor: what argmin alone concludes from the final scan
        d_end = scan_distance(sub[-1:], self.loc.grid.free_scans)[0]
        end_xy = np.asarray(self.loc.grid.free_xy[int(np.argmin(d_end))],
                            dtype=np.float64)
        ## start anchor: absent offline (the gate has no episode), which is correct --
        ## it must never be silently replaced by a true coordinate.
        start_xy = getattr(self, '_decode_anchor_xy', None)
        return viterbi_decode_xy(
            sub, self.loc.grid, self._vit_graph,
            sigma_e=self._vit_sigma_e, w_step=self._vit_w_step,
            beam=self._vit_beam, start_xy=start_xy, end_xy=end_xy)

    def _decode_plan_xy(self, scans):
        """(H, n_beams) plan scans -> (H, 2) xy targets.

        Sequence-consistent decode: the histogram FILTER is run along the plan
        (strided for speed), so self-similar corridors are disambiguated by the
        plan's own motion; the strided estimates are interpolated back to all H
        frames and lightly smoothed (PID targets should not jump).

        Two decoders, selected by `LIDAR_DECODE`:

        * ``argmin`` (default) -- per-frame nearest grid pose, memoryless. No motion
          model, so nothing to miscalibrate and no belief to lock in.
        * ``filter`` -- the forward histogram filter, sequence-consistent in
          principle. Every number produced before 2026-08-24 used this.

        The default changed on evidence, not taste. `plan_motion_census` decoded 40
        REAL generated plans spanning all 5 blocks and scored the two frames whose
        truth is exact (frame 0 = the inpainted start, frame -1 = the goal scan, which
        the model reproduces to 1e-07). Endpoints within one cell: filter 48% at the
        shipped sigma=1.0, 82% at its best sigma=4.0, **argmin 90%**. The filter's
        worst per-block p90 is 40 wu; argmin's is 0.6 wu everywhere except block 1,
        whose 50% is single-scan aliasing at max_range=12 that no decoder can fix.

        `decode_bench` separately showed the filter is NOT intrinsically broken -- on
        scanner-rendered trajectories it scores 0.21 wu and beats argmin under heavy
        noise. It is specifically the *generated* plans it cannot hold, and the
        scan-space seam blending once suspected for that was measured and cleared."""
        H = scans.shape[0]
        assert scans.ndim == 2 and scans.shape[1] == self.n_beams, scans.shape
        idx = np.unique(np.concatenate([np.arange(0, H, self._decode_stride), [H - 1]]))
        sub = scans[idx].astype(np.float32)
        if self._decode_mode == 'argmin':
            ## (n_dec, M) scan distance to every free pose, then the nearest.
            d = scan_distance(sub, self.loc.grid.free_scans)
            k_best = np.argmin(d, axis=1)
            map_xy = np.asarray(self.loc.grid.free_xy[k_best], dtype=np.float64)
            if getattr(self, '_conf_repl_thr', 0.0) > 0:
                self._plan_flag_idx = self._conf_flag(d, k_best, idx)
            map_xy = self._median_dejitter(map_xy, self._decode_median)
        elif self._decode_mode == 'viterbi':
            map_xy = self._decode_viterbi(sub)
            ## NO median de-jitter here. The median exists to erase the isolated
            ## teleports argmin produces; this decode cannot produce them, because a
            ## teleport is not an admissible transition. Running it anyway would only
            ## round the corners the constraint just earned.
        else:
            r = self.loc.localize_sequence(
                sub, motion_sigma_cells=self._decode_motion_sigma, theta_blur=0)
            map_xy = np.asarray(r['map_xy'], dtype=np.float64)      # (n_dec, 2)
        full = np.stack([np.interp(np.arange(H), idx, map_xy[:, 0]),
                         np.interp(np.arange(H), idx, map_xy[:, 1])], axis=1)
        if H >= 3:                                                  # light smoothing
            k = np.ones(3) / 3.0
            pad = np.pad(full, ((1, 1), (0, 0)), mode='edge')
            full = np.stack([np.convolve(pad[:, 0], k, 'valid'),
                             np.convolve(pad[:, 1], k, 'valid')], axis=1)
            ## The 3-tap mean is edge-padded, so it averages the last frame with its
            ## neighbour and drags the two frames that carry real information:
            ## frame 0 (the inpainted current scan -- what `head_err_vs_true` and the
            ## ada trigger at wp_idx=0 read) and frame -1 (the goal anchor -- what G2
            ## scores). Measured over 40 real plans: unpinned costs 15 points of
            ## endpoint accuracy (90% -> 75% within a cell) and 2 of head accuracy,
            ## for no gain in arc length or free-cell fraction. Pin them back.
            ##
            ## NOT applied under `filter`: that path is the frozen reproduction of
            ## every number produced before 2026-08-24 and must stay bit-identical.
            if self._decode_mode != 'filter':
                full[0], full[-1] = map_xy[0], map_xy[-1]
        ## decode-quality stat: share of consecutive decoded steps that jump more
        ## than one maze cell (~0 for a healthy plan + decode)
        jumps = np.linalg.norm(np.diff(map_xy, axis=0), axis=1)
        self._decode_stats.append(dict(
            n=int(H), decode=self._decode_mode,
            decode_median=int(self._decode_median),
            frac_cell_jumps=float((jumps > self.scanner.maze_unit).mean()) if len(jumps) else 0.0,
            end_xy=[float(full[-1, 0]), float(full[-1, 1])]))
        return self._pull_plan(full)

    def _conf_flag(self, d, k_best, idx):
        """First plan index where the argmin decode stops being trustworthy, or None.

        Per decoded frame, margin = (best distance among poses more than one cell from
        the winner) - (the winner's distance). Flag the first run of
        `_conf_repl_run` consecutive decoded frames with margin < `_conf_repl_thr`
        that starts at plan index >= `_conf_repl_min_gap`. Returns that run's FIRST
        plan index, which is what the cursor is compared against."""
        free_xy = np.asarray(self.loc.grid.free_xy, dtype=np.float32)
        far = np.linalg.norm(free_xy[None, :, :] - free_xy[k_best][:, None, :],
                             axis=-1) > float(self.scanner.maze_unit)
        d_far = np.where(far, d, np.inf).min(axis=1)
        margin = d_far - d[np.arange(len(k_best)), k_best]
        low = margin < self._conf_repl_thr
        R = self._conf_repl_run
        flag = None
        if len(low) >= R:
            run = np.convolve(low.astype(np.int32), np.ones(R, np.int32), 'valid') >= R
            ok = np.flatnonzero(run & (np.asarray(idx[:len(run)]) >= self._conf_repl_min_gap))
            if len(ok):
                flag = int(idx[ok[0]])
        self._conf_ep_flags.append(-1 if flag is None else flag)
        return flag

    def _extra_replan_due(self, wp_idx, prev_dfu_wp_idx):
        """FIX 11: replan once the pursuit cursor reaches the current plan's first
        low-confidence run. Never while captured (the plan is not being followed), and
        never sooner than `_conf_repl_min_gap` steps after the last plan."""
        if self._conf_repl_thr <= 0 or self._plan_flag_idx is None or self._captured:
            return False
        if wp_idx - prev_dfu_wp_idx < self._conf_repl_min_gap:
            return False
        if self._cursor < self._plan_flag_idx:
            return False
        self._conf_repls += 1
        self._conf_ep_repls += 1
        self._conf_ep_fire_steps.append(int(wp_idx))
        self._conf_ep_flag_lead.append(int(self._cursor - self._plan_flag_idx))
        ## consumed: the replan that follows decodes a fresh plan and sets a new flag
        self._plan_flag_idx = None
        return True

    def _extra_replan_budget(self):
        return self._conf_repl_extra if self._conf_repl_thr > 0 else 0

    def _pull_plan(self, xy):
        """String-pull the decoded plan: replace wiggles with straight shots.

        THIS is the operation the 80-plan census measured (arc p50 312 -> 80 wu, and
        80/80 plans inside the 200 wu budget). It runs plan-point to plan-point, so the
        result is still a CONNECTED ROUTE from the plan's start to the plan's goal --
        only shorter. That is the whole difference from the refuted runtime cursor
        shortcut (`_advance_cursor`), which pulled from the BALL's position and let the
        cursor teleport past the route.

        Why the plan needs this at all: the decoded plan is ~315 wu of arc (p50 311)
        against a **200 wu** physical budget (0.2 wu/step x 1000 steps), so 77 of 80
        plans could not be driven even by a flawless controller. The plan wanders 6.2x
        the direct distance but stays in free space, so the wander is removable.

        The pulled polyline is RESAMPLED at `pull_resample_wu` because the pursuit
        cursor advances by nearest-plan-point: a bare polyline with 10 wu straights
        would leave the cursor stranded between corners. Endpoints are pinned -- the
        last point is the goal anchor that G2b scores and that the whole episode aims
        at, and it must survive resampling exactly.
        """
        if self._pull_win <= 0 or len(xy) < 3:
            return xy
        ## DROP THE DECODE'S WALL POINTS FIRST. argmin+median leaves ~1.8% of the
        ## plan inside wall cells, and those points are exactly where the greedy walk
        ## degenerates: `los_clear` from inside a wall sees NOTHING, so the pull falls
        ## back to `i+1` and appends the wall points one by one as corners. Measured:
        ## 12 of 23 kept segments were that fallback and the kept corners were 29% in
        ## walls, against 1.8% in the plan it started from -- the pull was
        ## *preferentially selecting* the decode's worst points. Anchors are exempt:
        ## plan[-1] is the goal anchor and must survive whatever the decode did.
        _i, _j = self.scanner.world_to_cell(xy[:, 0], xy[:, 1])
        _free = ~self.scanner._is_wall(_i, _j)
        _free[0] = _free[-1] = True
        xy = xy[_free]
        n = len(xy)
        if n < 3:
            return xy
        keep = [0]
        i = 0
        while i < n - 1:
            hi = min(i + self._pull_win + 1, n)
            j = None
            if hi > i + 1:
                cand = np.arange(i + 1, hi, self._pull_stride)
                if len(cand):
                    ok = LOS_CLEAR(self.scanner, xy[i], xy[cand], margin=self._los_margin)
                    idx = np.flatnonzero(ok)
                    if len(idx):
                        j = int(cand[idx[-1]])
            i = j if (j is not None and j > i) else i + 1
            keep.append(i)
        pulled = xy[keep]
        if keep[-1] != n - 1:                    # never drop the goal anchor
            pulled = np.concatenate([pulled, xy[-1:]], axis=0)
        ## DROP ZERO-LENGTH SEGMENTS BEFORE RESAMPLING. `np.interp` requires a
        ## strictly increasing `xp`, and the decoded plan is **43% repeated
        ## consecutive points** (the median-15 de-jitter emits long constant runs),
        ## which survive into the pulled polyline as duplicate cumulative-arc values.
        ## Interpolating against those produced points scattered off the route:
        ## `free_frac` 0.988 -> 0.726, i.e. **27% of the pulled path inside walls**,
        ## while the arc and the endpoint both looked perfectly healthy. Keep the
        ## first point and every point that actually advances the arc.
        d_seg = np.linalg.norm(np.diff(pulled, axis=0), axis=1)
        pulled = np.concatenate([pulled[:1], pulled[1:][d_seg > 1e-9]], axis=0)
        if len(pulled) < 2:
            return xy
        ## resample at a fixed arc spacing so the cursor always has a nearby point
        seg = np.linalg.norm(np.diff(pulled, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(arc[-1])
        if total <= 1e-9:
            return xy
        m = max(int(np.ceil(total / self._pull_resample_wu)) + 1, 2)
        t = np.linspace(0.0, total, m)
        out = np.stack([np.interp(t, arc, pulled[:, 0]),
                        np.interp(t, arc, pulled[:, 1])], axis=1)
        out[0], out[-1] = pulled[0], pulled[-1]                 # pin both anchors
        self._pull_stats.append((float(np.sum(seg)), total, len(xy), len(out)))
        return out

    def _goal_scan_diag(self, gl_pos):
        """LOG ONLY: where the goal SCAN alone says the goal is, and how the true
        goal cell ranks in that posterior. This is the quantity the previous fork
        promoted to an eval rung (L2); here it is a diagnostic, because nothing in
        this control loop consumes a decoded goal cell."""
        try:
            g = self.loc_grid
            ## (n_row, n_col) posterior over position (theta is length 1 for the ball)
            post = np.asarray(self.loc.posterior_xy(self._goal_scan(gl_pos)),
                              dtype=np.float64)
            flat = post.reshape(-1)
            order = np.argsort(-flat)
            true_xy = np.asarray(gl_pos[:2], dtype=np.float64)
            r0, c0 = np.unravel_index(int(order[0]), post.shape)
            map_xy = np.array([g.x_vals[c0], g.y_vals[r0]], dtype=np.float64)
            ## rank of the true goal's own lattice cell in the posterior ordering
            ri = int(np.clip(round((true_xy[1] - g.y_vals[0]) / g.xy_step), 0, g.n_row - 1))
            ci = int(np.clip(round((true_xy[0] - g.x_vals[0]) / g.xy_step), 0, g.n_col - 1))
            true_flat = ri * post.shape[1] + ci
            rank = int(np.flatnonzero(order == true_flat)[0])
            return dict(map_xy=[float(map_xy[0]), float(map_xy[1])],
                        map_err_wu=float(np.linalg.norm(map_xy - true_xy)),
                        true_rank=rank, top1_mass=float(flat[order[0]]))
        except Exception as e:  # noqa: BLE001
            return dict(error=str(e))

    # ================= pursuit =================
    def _advance_cursor(self, pos):
        """Monotone progress cursor into the decoded plan (pure pursuit).

        Snap to the nearest plan index AT OR AHEAD of where we already are, searching
        only a bounded window forward: forward-only keeps the ball out of a loop it
        has already run, the bounded window keeps it from teleporting across a plan
        that revisits a corridor.

        STALL ESCAPE. Progress pursuit only advances when the ball gets NEAR a plan
        point, so an unreachable one traps it forever -- the decoded plan carries
        residual jumps, and a cursor sitting just before one aims across a wall.

        TWO RECOVERIES, TWO RATES (RETRAIN §8.5). These were
        one branch on one `stall_steps=50` timer, and that timer was the single largest
        sink in the whole loop: `frozen_frac ~= stall_escapes * 50 / total_steps` fits the
        12 bench runs at **r=0.99** (ratio 0.92-0.97 on the periodic arms), i.e. the
        "frozen" time was not the ball being stuck, it WAS this constant. In the ada_dist
        arms the escape was even the primary progress mechanism -- 17.4 escapes/ep x ~15
        indices = 261 against an observed `cursor_max` of 267 -- throttled to one per 50
        steps. They are split because they carry opposite risks:

          * `_los_seek` (SAFE, fires at `stall_steps`, now 5). It can only ever select a
            point with clear line of sight from the ball, so firing it often cannot
            desync the cursor. Retried every step while stalled; `los_clear` is one
            vectorised numpy call and `_carrot_idx` already pays one per step anyway.
          * blind forward walk (UNSAFE, stays at `blind_walk_steps`, 50). `cursor+1` with
            nothing visible advances on faith. Run it at 5 and it walks ~200 indices per
            episode; once it outruns the ball by more than `pursuit_win` the forward-only
            window can never re-snap and the plan is lost for good. It stays slow, and is
            counted separately so the two can never again be read as one number."""
        plan = self._plan_xy
        lo = int(self._cursor)
        hi = min(lo + self._pursuit_win + 1, len(plan))
        if hi > lo:
            d = np.linalg.norm(plan[lo:hi] - np.asarray(pos, dtype=np.float64), axis=1)
            self._cursor = lo + int(np.argmin(d))

        ## ---- LOS cursor shortcut -- REFUTED by benchmark runs. Kept only so the
        ## runs that produced the refutation stay reproducible; leave it at 0. ----
        ## It seeks the furthest index visible FROM THE BALL, which is NOT the
        ## string-pull that was measured offline (that pulls plan-point to
        ## plan-point). LOS guarantees the ball can drive AT a point, never that the
        ## point is AHEAD OF IT ALONG THE ROUTE, so a late index that wanders back
        ## near the ball teleports the monotone cursor past the route in between.
        ## Measured at win=888: cursor_max 215 -> 751 of 888 and 4113 indices skipped
        ## per episode, while `min_goal_dist` moved 32.1 -> 30.2 and
        ## corr(cursor_max, min_goal_dist) = -0.16. `carrot_dist_p50` 1.95 -> 6.4 is
        ## the cursor running away from the ball. Cursor progress became a phantom.
        ## The correct operation is `_pull_plan`, applied to the PLAN at decode time.
        if self._los_shortcut_win > 0:
            s_hi = min(self._cursor + self._los_shortcut_win + 1, len(plan))
            nxt = self._los_seek(pos, int(self._cursor), s_hi,
                                 stride=self._los_shortcut_stride)
            if nxt is not None and nxt > self._cursor:
                self._los_shortcuts += 1
                self._los_shortcut_gain += int(nxt - self._cursor)
                self._cursor = nxt

        if self._cursor > lo:
            self._stall = 0
        else:
            self._stall += 1
            if 0 < self._stall_steps <= self._stall:
                nxt = self._los_seek(pos, lo, hi)
                if nxt is not None:
                    self._cursor = nxt
                    self._stall_escapes += 1
                    self._stall = 0
                elif self._stall >= self._blind_walk_steps:
                    ## nothing visible for a long time: advance on faith, slowly.
                    ## CURSOR LEASH -- gate the point we are about to march TO (not
                    ## the one we are on, which a single decode break can put a maze
                    ## away from its successor). Past the leash, re-acquire BACKWARD.
                    nxt = min(lo + 1, len(plan) - 1)
                    back = None
                    if self._cursor_leash_wu > 0 and np.linalg.norm(
                            plan[nxt] - np.asarray(pos, dtype=np.float64)
                    ) > self._cursor_leash_wu:
                        back = self._cursor_reacquire(pos, lo)
                    if back is not None:
                        self._cursor = back
                        self._cursor_reacquires += 1
                    else:
                        ## nothing in reach behind either: escape anyway, so a
                        ## leashed run is never MORE stuck than the control arm
                        self._cursor = nxt
                        self._blind_walks += 1
                    self._stall = 0
                ## else: keep the cursor, keep accumulating, retry _los_seek next step
        return int(self._cursor)

    def _los_seek(self, pos, lo, hi, stride=1):
        """Furthest plan index in ``(lo, hi)`` with clear line of sight from ``pos``,
        or None when the window is empty / nothing is visible.

        ``stride`` subsamples the candidates. It exists only for the wide LOS shortcut,
        where the window can be the whole plan and the raycast runs every control step;
        the stall recovery calls this with stride=1 and is unaffected."""
        if hi <= lo + 1:
            return None
        cand = np.arange(lo + 1, hi, max(int(stride), 1))
        if not len(cand):
            return None
        ok = LOS_CLEAR(self.scanner, pos, self._plan_xy[cand], margin=self._los_margin)
        idx = np.flatnonzero(ok)
        return int(cand[idx[-1]]) if len(idx) else None

    def _cursor_reacquire(self, pos, lo):
        """Nearest plan index at or BEHIND the cursor that is within the leash of the
        ball, or None (leash off, or nothing behind is in reach -- the caller then
        escapes forward as before, so a leashed run is never more stuck).

        Prefers a point with clear line of sight. `lo` itself is a legal answer and
        means "hold here", which is correct when the cursor is already the closest
        thing the ball can drive at."""
        if self._cursor_leash_wu <= 0:
            return None
        plan = self._plan_xy
        pos = np.asarray(pos, dtype=np.float64)
        st = max(0, lo - int(self._pursuit_back))
        cand = plan[st:lo + 1]
        if not len(cand):
            return None
        d = np.linalg.norm(cand - pos, axis=1)
        near = d <= self._cursor_leash_wu
        if not near.any():
            return None
        vis = np.flatnonzero(np.logical_and(
            near, LOS_CLEAR(self.scanner, pos, cand, margin=self._los_margin)))
        if len(vis):
            return int(st + vis[int(np.argmin(d[vis]))])
        idx = np.flatnonzero(near)
        return int(st + idx[int(np.argmin(d[idx]))])

    def _carrot_idx(self, cur, pos):
        """Index of the pursuit target: `lookahead_wu` of plan ARC ahead of the
        cursor, then pulled BACK to the furthest point that is actually visible.

        Never aim at a plan point the ball cannot drive at in a straight line: the
        PID is pure proportional, so an occluded target means pressing into a wall
        until the episode ends."""
        plan = self._plan_xy
        seg = np.linalg.norm(np.diff(plan[cur:], axis=0), axis=1)
        if not len(seg):
            return len(plan) - 1
        adv = int(np.searchsorted(np.cumsum(seg), self._lookahead_wu) + 1)
        tgt = int(min(cur + adv, len(plan) - 1))

        if tgt > cur:
            win = plan[cur:tgt + 1]
            ok = LOS_CLEAR(self.scanner, pos, win, margin=self._los_margin)
            ## EUCLIDEAN CARROT CAP. LOS alone says a point is reachable in a
            ## straight line, never that it is NEAR. Across one of the plan's
            ## breaks, 2 wu of arc is up to 50 wu of real distance.
            if self._max_carrot_wu > 0:
                near = (np.linalg.norm(win - np.asarray(pos, dtype=np.float64),
                                       axis=1) <= self._max_carrot_wu)
                ok = np.logical_and(ok, near)
            idx = np.flatnonzero(ok)
            if len(idx):
                new = int(cur + idx[-1])
                self._los_pullbacks += int(new < tgt)
                return new
            self._los_blind += 1
            ## Nothing qualifies. Falling through to the far nominal target is what
            ## aims the saturated PID at a wall and holds it there for the rest of
            ## the episode (RETRAIN 8.5 fix #4). With a cap set, HOLD AT THE CURSOR
            ## and let the stall machinery handle it.
            if self._max_carrot_wu > 0:
                return cur
        return tgt

    def _capture_target(self, pos):
        """The decoded goal, once the ball is close enough to drive straight at it.

        Returns the capture target, or None to leave the pursuit cursor in charge.

        ENGAGE when the belief position is within `capture_wu` of the agent's own
        decoded goal AND that goal is in clear line of sight. RELEASE only when the
        distance exceeds `capture_release_wu` -- distance-only hysteresis, because a
        corner briefly occludes the goal and releasing on that would flicker the target
        between the goal and a plan index every few steps.

        `_goal_xy_est` is the argmin decode of the GOAL SCAN (`_ep_reset_hook`), never
        `gl_pos` and never `env.get_xy()`. That distinction is the whole point: the task
        hands the agent what the goal looks like, and decoding it is the agent's own
        inference. A test asserts it.
        """
        if self._capture_wu <= 0 or self._goal_xy_est is None:
            return None
        pos = np.asarray(pos, dtype=np.float64)
        gl = np.asarray(self._goal_xy_est, dtype=np.float64)
        d = float(np.linalg.norm(pos - gl))
        ## ---- the instrument that tells a NULL apart from a BUG ----
        ## An arm that never engages has two completely different explanations, and
        ## the engagement counter alone cannot separate them: the ball may never have
        ## come within the radius of its decoded goal at all, or it may have been
        ## there repeatedly and been refused by the line-of-sight guard every time.
        ## The first says capture had no chance and the blocker is upstream; the
        ## second says the radius or the margin is wrong. Record both.
        self._capture_near_min = min(getattr(self, '_capture_near_min', float('inf')), d)
        if self._captured:
            if d > self._capture_release_wu:
                self._captured = False
                return None
            ## NO-PROGRESS RELEASE. With a loose margin a capture can clip a corner and
            ## aim the saturated PID at a wall; release is on distance only, so without
            ## this nothing would ever free it. Also covers a badly decoded goal.
            if self._capture_stuck_steps > 0:
                if d < self._capture_best_d - 1e-3:
                    self._capture_best_d = d
                    self._capture_no_gain = 0
                else:
                    self._capture_no_gain += 1
                    if self._capture_no_gain >= self._capture_stuck_steps:
                        self._captured = False
                        self._capture_releases_stuck += 1
                        return None
        else:
            if d > self._capture_wu:
                return None
            self._capture_in_range_steps += 1
            if not bool(LOS_CLEAR(self.scanner, pos, gl[None],
                                  margin=self._capture_margin)[0]):
                self._capture_blocked_steps += 1
                return None
            self._captured = True
            self._capture_best_d = d
            self._capture_no_gain = 0
            self._capture_engagements += 1
            if self._capture_first_step < 0:
                self._capture_episodes += 1
                self._capture_dist_at_engage = d
                ## WITHIN-EPISODE step index. `_step_counter` is monotone across the
                ## whole run, so using it here would make the diagnostic unreadable.
                ## `_ep_log['true_xy']` is appended AFTER this call, so its current
                ## length is exactly the 0-based index of the step being acted on.
                L = self._ep_log
                self._capture_first_step = len(L['true_xy']) if L else 0
            utils.print_color(
                f'[ball capture] engaged at {d:.2f} wu from the decoded goal', c='g')
        self._capture_steps += 1
        self._capture_step_cnt += 1
        return gl

    def _goal_reentry(self, pos, cur):
        """Snap the cursor forward into the plan's own approach to the decoded goal.

        Returns the new cursor index, or None to leave it alone.

        Fires only while the ball is within `reentry_wu` of its decoded goal. The
        candidate set is the plan's GOAL REGION -- indices ahead of the cursor whose
        decoded position is within `reentry_wu` of the decoded goal -- and the chosen
        index is the one NEAREST THE BALL in that set, accepted only if it is closer to
        the ball than the cursor already is. That last clause is what keeps this from
        becoming FIX 5: `carrot_dist` can only fall, so the cursor cannot run away from
        the ball and turn progress into a phantom.

        Monotone, like every cursor move outside the leash: forward only, so it cannot
        send the ball back around a loop it has already run.
        """
        if self._reentry_wu <= 0 or self._goal_xy_est is None:
            return None
        plan = self._plan_xy
        if plan is None or cur >= len(plan) - 1:
            return None
        pos = np.asarray(pos, dtype=np.float64)
        gl = np.asarray(self._goal_xy_est, dtype=np.float64)
        if float(np.linalg.norm(pos - gl)) > self._reentry_wu:
            return None
        ahead = np.arange(cur + 1, len(plan))
        near_goal = ahead[np.linalg.norm(plan[ahead] - gl, axis=1) <= self._reentry_wu]
        if not len(near_goal):
            return None
        d_ball = np.linalg.norm(plan[near_goal] - pos, axis=1)
        if float(d_ball.min()) >= float(np.linalg.norm(plan[cur] - pos)):
            return None
        self._reentry_hops += 1
        self._reentry_ep_hops += 1
        return int(near_goal[int(np.argmin(d_ball))])

    # ================= one control step =================
    def _pid_lidar_act(self, obs_cur, goal_cur, wp_idx, fused_traj_ep, i_et):
        """waypoint scan -> xy target -> PID action."""
        true_xy = np.asarray(self.env.get_xy(), dtype=np.float64)
        pos = self._pos()

        if self._new_plan:
            ## a replan is the only event that resets the pursuit cursor -- which is
            ## why replan CADENCE is a control lever and not just a compute knob.
            self._new_plan = False
            self._cursor = 0
            self._stall = 0

        ## The cursor is advanced even while captured: it keeps `cursor_max` and the
        ## stall machinery live, and leaves a clean path back to plan pursuit if the
        ## capture releases.
        if self._plan_xy is not None and len(self._plan_xy):
            cur = self._advance_cursor(pos)
            ## re-enter the plan at its approach to the goal BEFORE picking the carrot,
            ## so the carrot comes from the goal region rather than from index ~215
            re = self._goal_reentry(pos, cur)
            if re is not None:
                self._cursor = cur = re
            plan_target = self._plan_xy[self._carrot_idx(cur, pos)]
        else:                                   # decode failed: 1-scan fallback
            plan_target = np.asarray(
                self.loc.localize_single(np.asarray(goal_cur, dtype=np.float32))['xy'],
                dtype=np.float64)
        cap_target = self._capture_target(pos)
        target = plan_target if cap_target is None else cap_target

        u = self._pid.step(pos, target)

        if self.is_use_subgoal_marker:
            try:
                self.env.set_subgoal_waypnt(np.asarray(target, dtype=np.float64))
            except Exception:  # noqa: BLE001
                pass

        L = self._ep_log
        if L is not None:
            L['true_xy'].append(true_xy.copy())
            L['pos_used'].append(np.asarray(pos, dtype=np.float64).copy())
            L['target'].append(np.asarray(target, dtype=np.float64).copy())
            L['cursor'].append(int(self._cursor))
            L['u_norm'].append(float(np.linalg.norm(u)))
            L['loc_err'].append(float(np.linalg.norm(pos - true_xy)))
            L['belief_conf'].append(float(getattr(self, '_belief_conf', 0.0)))
        return np.asarray(u, dtype=np.float32)


    # ================= replan probe =================
    def _probe_hist_rec(self, g_cond):
        """What the history channel actually carries at THIS replan.

        The training distribution is 63.7% fully-degenerate windows (32 copies of
        the episode's frame 0) and only 8.2% fully-real, so a rollout's history is
        in-distribution at t=0 and drifts OUT of it as real frames accumulate.
        `real_frac` is that drift, measured per replan."""
        rec = dict(step=int(getattr(self, '_step_counter', -1)),
                   hist_on=bool(self._hist_on))
        h = np.asarray(self._hist, dtype=np.float32) if len(self._hist) else None
        if h is not None and h.ndim == 2 and len(h):
            ## the left-pad signature: a run of leading frames identical to frame 0
            dup = 1
            while dup < len(h) and np.array_equal(h[dup], h[0]):
                dup += 1
            rec.update(n_frames=int(len(h)),
                       n_leading_dup=int(dup),
                       real_frac=float(1.0 - dup / len(h)),
                       n_unique=int(len(np.unique(h, axis=0))))
        ## did the tensor actually reach the model? this repo has already shipped one
        ## bug where g_cond['hist'] was silently overwritten with an empty array.
        gh = None if g_cond is None else g_cond.get('hist', None)
        if gh is None:
            rec['reached_model'] = False
        else:
            gh = np.asarray(gh)
            rec.update(reached_model=bool(gh.size > 0 and np.isfinite(gh).all()),
                       hist_shape=list(gh.shape))
        return rec

    def _probe_flush(self, scans):
        """Filter vs per-frame argmin along the plan, binned head / mid / tail.

        The forward filter starts from a UNIFORM prior, so at frame 0 it IS the
        per-frame argmin and can only diverge as the belief sharpens. If the
        offline gates' endpoint failure is a tail artefact, the head bin -- the
        only part the executor pursues before the next replan resets the cursor --
        should show near-zero disagreement."""
        rec = getattr(self, '_probe_rec', None) or dict(
            step=int(getattr(self, '_step_counter', -1)))
        self._probe_rec = None
        try:
            H = int(scans.shape[0])
            xy = self._plan_xy
            win = max(int(self._pursuit_win), 1) if hasattr(self, '_pursuit_win') else 60
            bins = dict(head=np.arange(0, min(win, H)),
                        mid=np.arange(max(H // 2 - win // 2, 0), min(H // 2 + win // 2, H)),
                        tail=np.arange(max(H - win, 0), H))
            for name, ix in bins.items():
                if not len(ix):
                    continue
                sub = ix[np.linspace(0, len(ix) - 1, min(len(ix), 24)).astype(int)]
                d = [float(np.linalg.norm(
                        np.asarray(self.loc.localize_single(
                            scans[int(k)].astype(np.float32))['xy'], dtype=np.float64)
                        - xy[int(k)]))
                     for k in sub]
                d = np.asarray(d)
                rec[f'{name}_argmin_gap_p50'] = float(np.percentile(d, 50))
                rec[f'{name}_agree_1cell'] = float((d <= self.scanner.maze_unit).mean())
            ## the two errors that matter operationally
            true_xy = np.asarray(self.env.get_xy(), dtype=np.float64)
            rec['head_err_vs_true'] = float(np.linalg.norm(xy[0] - true_xy))
            if self._true_goal_xy is not None:
                rec['end_err_vs_goal'] = float(np.linalg.norm(xy[-1] - self._true_goal_xy))
        except Exception as e:  # noqa: BLE001  -- a probe must never kill a rollout
            rec['probe_error'] = repr(e)
        self._replan_probe.append(rec)

    # ================= per-episode summary =================
    def _flush_ep(self):
        """Close out one episode's diagnostics. These are the MECHANISM GATES --
        read them before the success rate: an arm whose mechanism did not move did
        not run, whatever it scored."""
        L = self._ep_log
        self._ep_log = None
        if not L or not len(L['true_xy']):
            return
        true_xy = np.asarray(L['true_xy'])
        disp = (np.linalg.norm(np.diff(true_xy, axis=0), axis=1)
                if len(true_xy) > 1 else np.zeros(0))
        self._ep_summaries.append(dict(
            n_steps=int(len(true_xy)),
            ## share of control steps that move the ball ~zero. The previous fork
            ## measured 51% of the 1000-step budget here and deleting it was worth
            ## ~29 points of ceiling, so this is the first number to look at.
            frozen_frac=float((disp < 0.02).mean()) if len(disp) else 0.0,
            loc_err_p50=float(np.percentile(L['loc_err'], 50)) if L['loc_err'] else 0.0,
            loc_err_max=float(np.max(L['loc_err'])) if L['loc_err'] else 0.0,
            belief_conf_p50=(float(np.percentile(L['belief_conf'], 50))
                             if L['belief_conf'] else 0.0),
            cursor_max=int(np.max(L['cursor'])) if L['cursor'] else 0,
            goal_decode=L['goal_decode'],
            ## ada_dist calls that found no decoded plan and so could not fire
            ## cond_1. Non-zero means the trigger was partly blind this episode.
            ada_no_plan=int(getattr(self, '_ada_no_plan', 0)),
            ## ---- PACE (added 2026-08-30, RETRAIN §8.6) ----
            ## The ball's free-run speed in this env is a hard 0.2 wu/step (velocity
            ## clamp, measured directly), so 1000 steps buys AT MOST ~200 wu of arc
            ## against a decoded plan that is ~300 wu long. bench100 actually achieved
            ## ~0.015 wu/step -- about **8% of the ceiling** -- and nothing in the
            ## summary said so, because `u_norm` and `target` were logged per step and
            ## then dropped here. Without these three fields you cannot tell a ball
            ## that is BLOCKED (saturated command, no motion) from one that is being
            ## ASKED TO CRAWL (tiny command because the carrot was pulled back onto
            ## the ball). They are opposite defects with opposite fixes.
            speed_p50=float(np.percentile(disp, 50)) if len(disp) else 0.0,
            speed_frac_of_cap=(float(np.mean(disp)) / BALL_SPEED_CAP_WU
                               if len(disp) else 0.0),
            ## commanded |u|. Saturation is sqrt(2)~=1.414 for a 2-vector clipped
            ## componentwise to [-1, 1]; kp=5 means any carrot >~0.28 wu away saturates.
            u_norm_p50=float(np.percentile(L['u_norm'], 50)) if L['u_norm'] else 0.0,
            u_sat_frac=(float(np.mean(np.asarray(L['u_norm']) > 1.40))
                        if L['u_norm'] else 0.0),
            ## how far ahead the carrot actually sat. `lookahead_wu` asks for 2.0; the
            ## LOS pullback can drag it back onto the ball, and THAT is what a small
            ## u_norm would mean.
            carrot_dist_p50=(float(np.percentile(
                np.linalg.norm(np.asarray(L['target']) - np.asarray(L['pos_used']),
                               axis=1), 50))
                if len(L['target']) and len(L['target']) == len(L['pos_used']) else 0.0),
            ## ---- MECHANISM GATE for goal capture ----
            ## Pre-registered: `capture_engaged` should be true on 15-30% of episodes
            ## (the measured `<12wu` shelf is 21%), and among engaged episodes the
            ## drift `final_goal_dist - min_goal_dist` (today +4.9 wu) should go to ~0.
            ## Outside 10-40% the radius is mis-set: re-tune, do NOT read the score.
            capture_engaged=bool(getattr(self, '_capture_first_step', -1) >= 0),
            capture_first_step=int(getattr(self, '_capture_first_step', -1)),
            capture_dist_at_engage=float(
                getattr(self, '_capture_dist_at_engage', float('nan'))),
            capture_frac=(float(getattr(self, '_capture_step_cnt', 0)) / len(true_xy)
                          if len(true_xy) else 0.0),
            ## closest the ball ever came to its OWN decoded goal, and how often it
            ## was inside the radius but refused by the LOS guard. `capture_engaged`
            ## false with `capture_near_min` >> capture_wu means capture never had a
            ## chance and the blocker is upstream; false with a small `near_min` and a
            ## large `blocked` share means the guard or the margin is what to look at.
            capture_near_min=float(getattr(self, '_capture_near_min', float('inf'))),
            capture_in_range_steps=int(getattr(self, '_capture_in_range_steps', 0)),
            capture_blocked_steps=int(getattr(self, '_capture_blocked_steps', 0)),
            reentry_hops=int(getattr(self, '_reentry_ep_hops', 0)),
            ## ---- MECHANISM GATE for FIX 11 (confidence-triggered replan) ----
            ## `conf_repls` > 0 means it fired. `conf_flags` is the flagged plan index
            ## of every plan decoded this episode (-1 = none), so the share of plans
            ## flagged and where they flag can be read without a rerun.
            ## `conf_flag_lead` is cursor - flag at each fire: ~0 means it fired on
            ## arrival at the run, large means the cursor had jumped past it.
            **({} if getattr(self, '_conf_repl_thr', 0.0) <= 0 else dict(
                conf_repls=int(self._conf_ep_repls),
                conf_fire_steps=list(self._conf_ep_fire_steps),
                conf_flag_lead=list(self._conf_ep_flag_lead),
                conf_flags=list(self._conf_ep_flags))),
        ))

    def update_j_data(self, json_data):
        """Fold the executor's diagnostics into the rollout json."""
        json_data = super().update_j_data(json_data)
        self._flush_ep()                     # the last episode never saw a reset
        json_data['ball_decode'] = self._decode_mode
        json_data['ball_pos_source'] = self._pos_source
        json_data['ball_hist_on'] = bool(self._hist_on)
        json_data['ball_hist_len'] = int(self.hist_len)
        json_data['ball_ep_summaries'] = self._ep_summaries
        json_data['ball_decode_stats'] = self._decode_stats
        if self._probe_on:
            json_data['ball_replan_probe'] = self._replan_probe
        json_data['ball_stall_escapes'] = int(self._stall_escapes)
        json_data['ball_blind_walks'] = int(self._blind_walks)
        json_data['ball_stall_steps'] = int(self._stall_steps)
        json_data['ball_blind_walk_steps'] = int(self._blind_walk_steps)
        json_data['ball_los_pullbacks'] = int(self._los_pullbacks)
        json_data['ball_los_blind'] = int(self._los_blind)
        ## MECHANISM GATE for the LOS shortcut: `ball_los_shortcut_win` records the
        ## operating point, `ball_los_shortcuts` that it fired, and `_gain` the plan
        ## indices it actually skipped. A run with win>0 and shortcuts==0 did not run
        ## the arm, whatever it scored.
        json_data['ball_los_shortcut_win'] = int(self._los_shortcut_win)
        json_data['ball_los_shortcuts'] = int(self._los_shortcuts)
        json_data['ball_los_shortcut_gain'] = int(self._los_shortcut_gain)
        ## MECHANISM GATE for the plan string-pull: arc before/after, per decode.
        json_data['ball_pull_win'] = int(self._pull_win)
        ## MECHANISM GATES for the three fork guards.
        json_data['ball_max_carrot_wu'] = float(self._max_carrot_wu)
        json_data['ball_cursor_leash_wu'] = float(self._cursor_leash_wu)
        json_data['ball_cursor_reacquires'] = int(self._cursor_reacquires)
        json_data['ball_ada_ncomp'] = int(self._ada_ncomp)
        ## MECHANISM GATE for goal capture: the operating point, and that it fired.
        ## `capture_wu > 0` with `capture_episodes == 0` did not run the arm.
        json_data['ball_capture_wu'] = float(self._capture_wu)
        json_data['ball_capture_release_wu'] = float(self._capture_release_wu)
        json_data['ball_capture_episodes'] = int(self._capture_episodes)
        json_data['ball_capture_engagements'] = int(self._capture_engagements)
        json_data['ball_goal_reentry_wu'] = float(self._reentry_wu)
        json_data['ball_reentry_hops'] = int(self._reentry_hops)
        json_data['ball_capture_margin'] = float(self._capture_margin)
        json_data['ball_capture_stuck_steps'] = int(self._capture_stuck_steps)
        json_data['ball_capture_releases_stuck'] = int(self._capture_releases_stuck)
        json_data['ball_capture_steps'] = int(self._capture_steps)
        json_data['ball_loc_eps_mix'] = float(self._loc_eps_mix)
        if self._conf_repl_thr > 0:
            json_data['ball_conf_repl_thr'] = float(self._conf_repl_thr)
            json_data['ball_conf_repl_run'] = int(self._conf_repl_run)
            json_data['ball_conf_repl_min_gap'] = int(self._conf_repl_min_gap)
            json_data['ball_conf_repl_extra'] = int(self._conf_repl_extra)
            json_data['ball_conf_repls'] = int(self._conf_repls)
        if self._ncomp_used:
            _nu = np.asarray(self._ncomp_used, dtype=np.float64)
            json_data['ball_ncomp_used_p50'] = float(np.median(_nu))
            json_data['ball_ncomp_used_min'] = int(_nu.min())
            json_data['ball_ncomp_used_max'] = int(_nu.max())
            json_data['ball_ncomp_n_plans'] = int(len(_nu))
        if self._pull_stats:
            _pb = np.asarray([a for a, _, _, _ in self._pull_stats], dtype=np.float64)
            _pa = np.asarray([b for _, b, _, _ in self._pull_stats], dtype=np.float64)
            json_data['ball_pull_arc_before_p50'] = float(np.median(_pb))
            json_data['ball_pull_arc_after_p50'] = float(np.median(_pa))
            json_data['ball_pull_n_decodes'] = int(len(_pb))
        return json_data
