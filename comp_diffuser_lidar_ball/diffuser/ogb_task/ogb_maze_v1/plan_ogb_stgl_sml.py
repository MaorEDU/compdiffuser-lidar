import sys, os; sys.path.append('./')
os.environ['PYOPENGL_PLATFORM'] = 'egl' ## enable GPU rendering in mujoco
os.environ['MUJOCO_GL'] = 'egl'
import pdb, torch, copy, pdb, json
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)
##
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from datetime import datetime
import os.path as osp
import diffuser.utils as utils
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_planner_v1 import OgB_Stgl_Sml_MazeEnvPlanner_V1




class Parser(utils.Parser):
    dataset: str = None
    config: str = None
    ## should not put any existing variable in config file here
    pl_seeds: str = '-1' # no seed
    plan_n_ep: int = -100 ## all if -100, auto parse to int

def main(args_train, args):

    #---------------------------------- setup ----------------------------------#

    ld_config = dict()

    ## LiDAR planner: observations are 30-beam scans, so use the LiDAR rollout
    ## subclass (inserts scanner.scan for the planner conditioning + the 34-D
    ## inv-dyn obs). Selected by the presence of a lidar_config in the dataset.
    is_lidar = args_train.dataset_config.get('lidar_config', None) is not None
    if is_lidar and 'pointmaze' in str(args_train.dataset).lower():
        ## BALL: pointmaze LiDAR rollout (decode + PID executor / optional
        ## learned inv-dyn; see ogb_stgl_sml_lidar_pnt_planner_v1 + BALL.md).
        from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_pnt_planner_v1 import \
            OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1
        ogmz_planner = OgB_Stgl_Sml_Lidar_Pnt_MazeEnvPlanner_V1(args_train, args=args)
    elif is_lidar:
        from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_planner_v1 import \
            OgB_Stgl_Sml_Lidar_MazeEnvPlanner_V1
        ogmz_planner = OgB_Stgl_Sml_Lidar_MazeEnvPlanner_V1(args_train, args=args)
    else:
        ogmz_planner = OgB_Stgl_Sml_MazeEnvPlanner_V1(args_train, args=args)
    ogmz_planner.setup_load( ld_config=ld_config )

    #---------------------------- start planning -----------------------------#

    pl_seeds = args.pl_seeds

    from diffuser.datasets.d4rl import Is_OgB_Robot_Env
    if len(pl_seeds) == 1:
        ## plan_n_ep
        if Is_OgB_Robot_Env:
            if pl_seeds[0] == -1: ## no seed
                avg_result_dict = ogmz_planner.ogb_plan_once(pl_seed=None,)
            else:
                avg_result_dict = ogmz_planner.ogb_plan_once(pl_seed=pl_seeds[0])
        
    else:
        utils.print_color(f'{args.pl_seeds=}')
        raise NotImplementedError ## can impl plan_multi_run

    ## might prevent the final exception before the program finishes
    ogmz_planner.env.close()
    del ogmz_planner.env
    del ogmz_planner.renderer.env
    
    return avg_result_dict


if __name__ == '__main__':
    ## training args
    args_train = Parser().parse_args('diffusion')
    args = Parser().parse_args('plan')
    ## 1. get epoch to eval on, by default all
    loadpath = args.logbase, args.dataset, args_train.exp_name

    args.pl_seeds = utils.parse_seeds_str(args.pl_seeds) ## a list of int
    args.n_batch_acc_probs = 4 ##
    
    ### --- Hyper-parameters Setup ---
    from diffuser.datasets.d4rl import Is_OgB_Robot_Env
    assert Is_OgB_Robot_Env
    
    

    ## Default
    args.is_replan = None ## placeholder, should be replaced in the if code blcok below
    args.n_act_per_waypnt = 2
    args.is_save_pkl = False
    args.is_rd_agv = False

    ## the state dimension used in the diffusion models
    dfu_ndim = len(args_train.dataset_config['obs_select_dim'])

    ## LiDAR planner: the diffusion observation is a LiDAR scan, not (x, y), so we
    ## branch on the presence of a lidar_config (obs_select_dim is a placeholder
    ## (0, 1) in the LiDAR config and would otherwise collide with the 2-D branch).
    is_lidar_cfg = args_train.dataset_config.get('lidar_config', None) is not None


    ## ---------------------------------------
    ## ----------- Ant Maze Stitch -----------
    if 'antmaze' in args.dataset.lower() and 'stitch' in  args.dataset.lower():
        if 'giant' in args.dataset:

            ## Ant Maze Giant -- LiDAR closed-loop rollout (Option D)
            if is_lidar_cfg:
                ## First working video: one composed LiDAR plan per episode, the
                ## inv-dyn chases its scans. Adaptive replanning and the subgoal
                ## marker need per-waypoint (x, y), which a LiDAR plan does not
                ## carry, so they are disabled here (see LIDAR_ROLLOUT_VIDEO_TODO.md
                ## §3.2/§4/§7).
                args.ep_st_idx = 0
                repl_wp_cfg = {}
                args.n_act_per_waypnt = 1
                args.ev_cp_infer_t_type = 'interleave'
                args.ev_n_comp = 9
                args.rd_resol = 300
                args.is_use_subgoal_marker = False

                ## ---- tunable rollout knobs (env-driven so the baseline and the
                ## ---- variants run without re-editing; defaults == the first run) ----
                ## Goal-scan heading: '0.0' (fixed, default) or 'goal' (use the goal
                ## state's real heading -- sharper but pose-dependent conditioning).
                _yaw = os.environ.get('LIDAR_GOAL_YAW', '0.0')
                try:
                    args.lidar_goal_yaw = float(_yaw)
                except ValueError:
                    args.lidar_goal_yaw = _yaw                  # e.g. 'goal'

                ## Replanning: OFF by default (one plan per episode). Set
                ## LIDAR_REPLAN=1 to re-condition on the CURRENT scan whenever the
                ## current plan is consumed -- helps cross the giant maze, where a
                ## single H160 plan can't reach the goal. (mode 'periodic', added to
                ## ogb_stgl_sml_planner_v1.py; scan-space safe, keeps full n_comp.)
                if os.environ.get('LIDAR_REPLAN', '0') == '1':
                    args.is_replan = 'periodic'
                    args.repl_ada_dist_cfg = dict(
                        n_max_steps=int(os.environ.get('LIDAR_N_MAX_STEPS', '2500')),
                        max_n_repl=int(os.environ.get('LIDAR_MAX_N_REPL', '15')),
                        ## LIDAR_REPL_EVERY=N (>0): also replan every N consumed
                        ## waypoints, not only when the whole plan is used up.
                        ## 0 (default) keeps the original "replan when plan consumed".
                        repl_every=int(os.environ.get('LIDAR_REPL_EVERY', '0')),
                    )
                else:
                    args.is_replan = False
                    ## no-replan path: only n_max_steps is read from this dict
                    args.repl_ada_dist_cfg = dict(n_max_steps=2000)

                ## Multi-scan goal-band spacing (eval-only): override the config's
                ## goal_band_step so it can be SWEPT without editing the config or
                ## retraining. NOTE: goal_band_k (the trained band width) is NOT
                ## overridable here -- it must match the planner checkpoint, so it
                ## stays config-driven.
                args.goal_band_step = float(
                    os.environ.get('LIDAR_GOAL_BAND_STEP', getattr(args, 'goal_band_step', 4.0)))

                ## Teacher-forced REAL goal band (decisive diagnostic): LIDAR_TF_BAND=1
                ## replaces the SYNTHESIZED band with the TRUE last-K scans of a real
                ## dataset trajectory ending at the goal -> isolates "band concept" from
                ## "eval-time synthesis is OOD". LIDAR_TF_NPZ overrides the default
                ## ~/.ogbench/data/<env>-val.npz.
                args.tf_band = int(os.environ.get('LIDAR_TF_BAND', '0'))
                args.tf_npz = os.environ.get('LIDAR_TF_NPZ', '')
                ## Wide-baseline teacher band: frames between band scans (S>1 spaces
                ## the K real scans over SEVERAL cells of the approach instead of the
                ## last K consecutive frames). Eval-only; ignored unless LIDAR_TF_BAND=1.
                args.tf_band_stride = int(os.environ.get('LIDAR_TF_BAND_STRIDE', '1'))

                ## --- localization-guided plan re-rank (eval-only) ------
                ## Activated by LIDAR_PICK_TYPE=grid (set globally below). These
                ## configure the (x, y, theta) reference grid used to DECODE each
                ## candidate plan's route and pick the one that actually reaches the
                ## goal cell (maze graph-distance). Read only when pick_type=='grid';
                ## otherwise inert -> default rollout is unchanged.
                args.loc_grid_xy_step = float(os.environ.get('LIDAR_LOC_XYSTEP', '1.0'))
                args.loc_grid_n_theta = int(os.environ.get('LIDAR_LOC_NTHETA', '12'))
                args.loc_grid_sigma = float(os.environ.get('LIDAR_LOC_SIGMA', '1.0'))
                args.loc_n_decode = int(os.environ.get('LIDAR_LOC_SUBSAMPLE', '24'))
                args.loc_grid_cache = os.environ.get(
                    'LIDAR_LOC_CACHE_ROOT', 'data/ogb_maze/lidar_grid')
                ## Goal-cell source for the re-ranker's graph-distance field:
                ##   'true'  = env's true goal (x,y)->cell. DIAGNOSTIC ablation that
                ##             isolates whether SELECTION helps, given a CORRECT target.
                ##             (Re-ranker still only *chooses* among LiDAR plans; the
                ##             model sees no x,y. The true goal is used only to define
                ##             the scoring target cell -- clearly an ablation, not the
                ##             compliant final.)
                ##   'lidar' = localize the goal SCAN on the grid (pure LiDAR, compliant).
                ##             If the goal scan is pose-ambiguous the goal cell may land
                ##             on a look-alike -- that is the effect we are measuring.
                args.loc_goal_mode = os.environ.get('LIDAR_LOC_GOAL_MODE', 'true').lower()

                ## --- Stage-3: LiDAR self-localization + goal-arrival termination ---
                ## The ant decodes its own position from LiDAR each step (histogram filter
                ## on the (x,y,theta) grid) and ENDS the run when it enters the goal cell
                ## (scan matches the trained goal scan). LIDAR_ARRIVAL=1 enables it; default
                ## off -> rollout unchanged. Arrival ends the episode but does NOT fake env
                ## success, so ep_srate stays ground-truth and arrival precision is measurable.
                args.lidar_arrival = int(os.environ.get('LIDAR_ARRIVAL', '0'))
                args.lidar_arrival_ends = int(os.environ.get('LIDAR_ARRIVAL_ENDS', '1'))
                args.lidar_arrival_goal_mode = os.environ.get('LIDAR_ARRIVAL_GOAL_MODE', 'lidar').lower()
                args.lidar_arrival_persist = int(os.environ.get('LIDAR_ARRIVAL_PERSIST', '3'))

                ## --- GRID NAVIGATOR (likelihood grid drives the route; no retrain) ---
                ## LIDAR_NAV_MODE: '' off (default) | 'bfs' Plan A (the executed plan IS the
                ## grid-routed scan sequence; diffusion sampling bypassed) | 'subgoal' Plan B
                ## (diffusion conditioned on a grid-routed subgoal band ~N cells ahead) |
                ## 'tf_exec' executor diagnostic (plan = a REAL dataset scan sequence).
                args.nav_mode = os.environ.get('LIDAR_NAV_MODE', '').lower()
                ## goal-cell source: 'lidar' localize the goal scan BAND on the grid
                ## (pure LiDAR, compliant) | 'true' env goal cell (diagnostic ablation).
                args.nav_goal_mode = os.environ.get('LIDAR_NAV_GOAL_MODE', 'lidar').lower()
                args.nav_step_wu = float(os.environ.get('LIDAR_NAV_STEP_WU', '0.13'))
                args.nav_yaw_smooth = int(os.environ.get('LIDAR_NAV_YAW_SMOOTH', '31'))
                args.nav_subgoal_cells = float(os.environ.get('LIDAR_NAV_SUBGOAL_CELLS', '10'))
                args.nav_min_conf = float(os.environ.get('LIDAR_NAV_MIN_CONF', '0.02'))
                args.nav_probe = int(os.environ.get('LIDAR_NAV_PROBE', '1'))
                args.nav_goal_hold = int(os.environ.get('LIDAR_NAV_GOAL_HOLD', '40'))
                ## tf_exec: min remaining in-episode frames of the picked real segment
                args.nav_tf_min_frames = int(os.environ.get('LIDAR_NAV_TF_MIN', '120'))
                ## tf_exec heading gate: max |live yaw - segment-start yaw| in degrees
                ## (2026-07-05: unmatched headings put the inv-dyn's obs->goal yaw
                ## delta far outside its k=3-4 training distribution).
                args.nav_tf_max_dyaw = float(os.environ.get('LIDAR_NAV_TF_MAX_DYAW', '60'))
                ## pursuit v2 (2026-07-05 regression fix; see _nav_pick_wp_idx):
                ## cursor PACED +1 frame/step, localization only issues a bounded
                ## (<= nav_corr_max frames/step) correction and only when est_xy
                ## snaps onto the local plan window (<= nav_snap_mj). 0 = pure
                ## time pacing (the 21089 regime).
                args.nav_pursuit = int(os.environ.get('LIDAR_NAV_PURSUIT', '1'))
                args.nav_lookahead = int(os.environ.get('LIDAR_NAV_LOOKAHEAD', '12'))
                args.nav_pursuit_win = int(os.environ.get('LIDAR_NAV_PURSUIT_WIN', '40'))
                args.nav_pursuit_back = int(os.environ.get('LIDAR_NAV_PURSUIT_BACK', '15'))
                args.nav_snap_mj = float(os.environ.get('LIDAR_NAV_SNAP_MJ', '3.0'))
                args.nav_corr_max = int(os.environ.get('LIDAR_NAV_CORR_MAX', '2'))
                ## A2 ribbon source (2026-07-07): 'synth' (default, unchanged) |
                ## 'data' = stitch REAL dataset frames along the BFS route so the
                ## executed goal scans come from the inv-dyn's training manifold
                ## (real poses + gait wobble) instead of ray-cast lattice poses.
                ## Frames from LIDAR_TF_NPZ (default <env>-val.npz; point it at the
                ## full train npz for densest coverage). Dataset xy only picks the
                ## frames (tf_exec standard); the model still sees scans only.
                args.nav_ribbon = os.environ.get('LIDAR_NAV_RIBBON', 'synth').lower()
                args.nav_a2_seg_frames = int(os.environ.get('LIDAR_NAV_A2_SEG', '12'))
                args.nav_a2_snap_wu = float(os.environ.get('LIDAR_NAV_A2_SNAP_WU', '2.0'))
                ## v2b chunk executor (2026-07-07): actions played per model query
                ## for act_chunk inv-dyn runs. 0 = auto (chunk/2). Ignored (inert)
                ## for single-action checkpoints; encoding/chunk themselves are
                ## auto-detected from the inv run's trainer pkl at load time.
                args.act_chunk_exec = int(os.environ.get('LIDAR_ACT_CHUNK_EXEC', '0'))
                ## live-filter motion diffusion (lattice steps/scan; also used by arrival)
                args.loc_motion_sigma = float(os.environ.get('LIDAR_LOC_MOTION_SIGMA', '1.0'))
                ## goal band rendered along the BFS approach corridor instead of the
                ## straight-line most-open-beam walk (train/eval OOD fix; needs K>1).
                args.band_from_route = int(os.environ.get('LIDAR_BAND_FROM_ROUTE', '0'))
                ## goal-contrast guidance weight on the ACTION (2026-07-05): amplify
                ## the inv-dyn's weak goal channel, a = a_self + w*(a_goal - a_self).
                ## 1.0 = off (bit-exact legacy). Probe-measured channel: ~20% of var.
                args.act_cfg_w = float(os.environ.get('LIDAR_ACT_CFG_W', '1.0'))
                ## TOP-K goal-posterior diagnostic (memory 2026-07-08): log the top-K
                ## goal *cells* + scores + where the TRUE cell ranks, one print per ep.
                args.nav_topk = int(os.environ.get('LIDAR_NAV_TOPK', '5'))
                ## verify-and-hop (fix-1): don't trust a self-declared arrival -- match
                ## the live scan to the goal band tail; on mismatch reject the cell and
                ## re-route to the next posterior hypothesis (sequential decoy sweep).
                args.nav_verify_hop = int(os.environ.get('LIDAR_VERIFY_HOP', '0'))
                ## accept tolerance (per-beam RMS world-units) for the live-vs-goal scan
                args.nav_verify_tol = float(os.environ.get('LIDAR_VERIFY_TOL', '1.5'))
                ## max hypothesis visits per episode (budget ~5-6 at ~1300 steps/8000)
                args.nav_verify_hop_max = int(os.environ.get('LIDAR_VERIFY_MAX_HOPS', '6'))
                ## belief-converged gate (fix-2): require this MAP posterior mass before
                ## honoring an arrival (0 = off) -- skips step-61 spawn-phantom arrivals.
                args.nav_verify_min_conf = float(os.environ.get('LIDAR_VERIFY_MIN_CONF', '0.0'))

                utils.print_color(
                    f'[lidar rollout] is_replan={args.is_replan} '
                    f'lidar_goal_yaw={args.lidar_goal_yaw} '
                    f"repl_every={os.environ.get('LIDAR_REPL_EVERY', '0')} "
                    f"cond_w={os.environ.get('LIDAR_COND_W', '2.0')} "
                    f"pick_type={os.environ.get('LIDAR_PICK_TYPE', 'first')} "
                    f'goal_band_step={args.goal_band_step} tf_band={args.tf_band} '
                    f'tf_band_stride={args.tf_band_stride} '
                    f'loc_goal_mode={getattr(args, "loc_goal_mode", "-")} '
                    f'nav_topk={args.nav_topk} verify_hop={args.nav_verify_hop} '
                    f'verify_tol={args.nav_verify_tol} verify_max_hops={args.nav_verify_hop_max} '
                    f'verify_min_conf={args.nav_verify_min_conf}', c='c')
                ## LiDAR inverse-dynamics run dir (must exist on disk). Default path
                ## resolver keys off env+gl_dim and will NOT find this, so set it
                ## explicitly. Confirm the dir name matches your trained run.
                ## --- proprioception fix: obs=[full 29-D ant obs, lidar] (obs_dim 59) ---
                ## The rollout auto-detects the obs_layout from this run's dataset_config,
                ## so no other rollout change is needed. To reproduce the old (car-like,
                ## 0% success) baseline, point this back at '...xyo_g30d_invdyn_h12'.
                ## LIDAR_INV_MODEL_PATH (2026-07-07): switch inv-dyn checkpoints
                ## without a code edit (e.g. the wide-k goal-aware retrain
                ## '..._invdyn_h64_widek'). Default = the h12 run (bit-exact legacy).
                args.inv_model_path = os.environ.get(
                    'LIDAR_INV_MODEL_PATH',
                    'logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h12')
                args.inv_epoch = os.environ.get('LIDAR_INV_EPOCH', 'latest')

            ## Ant Maze Giant
            elif dfu_ndim == 2:
                ## NOTE: Set the eval start idx, by default starts from 0
                args.ep_st_idx = 0
                repl_wp_cfg = {}
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1 ## number of actions per waypoint
                # args.ev_cp_infer_t_type = 'same_t' ## ours, faster
                # args.ev_cp_infer_t_type = 'gsc' ## baselines
                
                args.ev_cp_infer_t_type = 'interleave' ## ours
                args.rd_resol = 300 ## website default: 1000
                # args.is_save_pkl = True ## save the plan/rollout trajs to a pkl file

                args.ev_n_comp = 9

                ## high resolution, etc., other eval/render hyperparameters
                # args.rd_resol = 1600
                # args.ep_st_idx = 20
                # args.is_rd_agv = True
                # args.is_use_subgoal_marker = False
                # args.vid_fps = 60
                ## --------------------------------

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## 0: no replan
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ##
                    n_max_steps=2000, ## 
                )
                args.inv_epoch = int(8e5)
            
            ## Ant Maze Giant Higher Dim
            elif dfu_ndim == 15:
                args.ev_n_comp = 9
                repl_wp_cfg = {}

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.ev_cp_infer_t_type = 'interleave'

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=15,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150, ##
                    n_max_steps=2000, ## 
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)
            ## Ant Maze Giant
            elif dfu_ndim == 29:
                args.ev_n_comp = 9 # 8
                repl_wp_cfg = {}
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1 ## important:
                args.ev_cp_infer_t_type = 'interleave'

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=15, ##
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150, ## 100?
                    n_max_steps=2000, ## 
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)
        
        ## Ant Maze Large Stitch
        elif 'large' in args.dataset:
            if dfu_ndim == 29:
                
                args.ev_cp_infer_t_type = 'interleave' ## gsc / same_t (parallel)
                args.rd_resol = 300 # 1000
                # args.is_save_pkl = True


                repl_wp_cfg = {}
                args.ev_n_comp = 5 ## 6,7 is also fine
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=4, ## 
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ## 100?
                    n_max_steps=1000, ## 
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)


            ## Ant Maze Large Stitch
            elif dfu_ndim == 15:
                ## ----------------
                args.ev_cp_infer_t_type = 'interleave'

                repl_wp_cfg = {}
                args.ev_n_comp = 6
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4, 
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ## 100?
                    n_max_steps=1000, ## 
                    used_idxs=(0,1),
                )
                # pdb.set_trace()
                args.inv_epoch = int(8e5)

            ## Ant Maze Large Stitch
            elif dfu_ndim == 2:
                repl_wp_cfg = {}
                
                args.ev_n_comp = 6 ##
                args.ev_cp_infer_t_type = 'interleave'

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=4, 
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ## 
                    n_max_steps=1000, ## 
                )

                args.inv_epoch = int(8e5)


        ## Ant Maze Medium
        elif 'medium' in args.dataset:
            if dfu_ndim == 29:

                ## ---------------------
                repl_wp_cfg = {}
                args.ev_n_comp = 3 ##
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150,
                    n_max_steps=2000,
                    used_idxs=(0,1),
                )
                # pdb.set_trace()
                args.inv_epoch = int(8e5)

            ## Ant Maze Medium
            elif dfu_ndim == 15:
                repl_wp_cfg = {}
                args.ev_n_comp = 3 ## 4
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                ## -- failure analysis vis --
                args.ev_cp_infer_t_type = 'interleave'
                # args.rd_resol = 1000
                # args.is_save_pkl = True
                # args.is_rd_agv = True
                args.is_use_subgoal_marker = True
                args.vid_fps = 60
                ## --------------------------


                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150,
                    n_max_steps=2000,
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)

            
            ## Ant Maze Medium
            elif dfu_ndim == 2:

                args.ev_n_comp = 3

                args.ev_cp_infer_t_type = 'interleave'
                # args.rd_resol = 1000
                # args.is_save_pkl = True
                # args.is_rd_agv = True
                # args.is_use_subgoal_marker = True ## False
                # args.vid_fps = 60

                # args.ep_st_idx = 40
                repl_wp_cfg = {}
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150,
                    n_max_steps=1000,
                )
                args.inv_epoch = int(8e5)

    
    ## ----------------------------------------------------------------
    ## ---------------------- AntMaze Explore -------------------------
    ## ----------------------------------------------------------------
    
    ## Ant Maze Explore
    elif 'explore' in  args.dataset.lower() and 'antmaze' in args.dataset.lower():
        ## Explore Large
        if 'large' in args.dataset:
            if dfu_ndim in [29, 15]:
                args.ev_cp_infer_t_type = 'interleave'
                repl_wp_cfg = {}
                args.ev_n_comp = 10
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=2,
                    ada_dist_minus_n_wp=0,
                    type='m_2',
                    cond_2_extra=150,
                    n_max_steps=2000,
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)

            ## Explore Large
            elif dfu_ndim == 2:
                args.ev_cp_infer_t_type = 'interleave' ## or 'gsc', 'same_t_p'
                # args.rd_resol = 400
                # args.is_save_pkl = True

                repl_wp_cfg = {}
                args.ev_n_comp = 6
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0,
                    thres=2, ##
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150,
                    n_max_steps=1000,
                )
                args.inv_epoch = int(8e5)

            else:
                raise NotImplementedError
        
        ## Explore Medium
        elif 'medium' in args.dataset:
            if dfu_ndim in [29,15]:
                repl_wp_cfg = {} 
                args.ev_n_comp = 5 ## 4

                args.ev_cp_infer_t_type = 'interleave'
                args.rd_resol = 1000
                args.is_save_pkl = True

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=2, ##
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150,
                    n_max_steps=1000, ## 2000
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)
            
            ## Explore Medium
            elif dfu_ndim == 2:
                repl_wp_cfg = {}
                args.ev_n_comp = 5 ## Used

                args.ev_cp_infer_t_type = 'interleave' ## or 'gsc', 'same_t_p'
                args.rd_resol = 1000
                args.is_save_pkl = True

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## replan
                    thres=2,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150,
                    n_max_steps=1000,
                )
                args.inv_epoch = int(8e5)


    ## ----------------------------------------------------------------
    ## ---------------------- Humanoid Stitch -------------------------
    ## ----------------------------------------------------------------

    ## Only Implemented 2D For Now
    elif 'humanoid' in args.dataset.lower():
        if 'giant' in args.dataset:
            args.ev_n_comp = 11
            args.inv_epoch = int(16e5)

            args.ev_cp_infer_t_type = 'interleave'
            args.rd_resol = 1000
            args.is_save_pkl = True
            args.ep_st_idx = 80

            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1 ## important:
            args.repl_ada_dist_cfg = dict(
                max_n_repl=10,
                thres=10,
                type='m_2',
                ada_dist_minus_n_wp=300,
                # ada_dist_minus_n_wp=400, ## TODO:
                cond_2_extra=150, ## 100?
                n_max_steps=8000, ## NEW Jan 10
            )

        ## Humanoid
        elif 'large' in args.dataset:
            if dfu_ndim == 23:
                raise NotImplementedError
            elif dfu_ndim == 2:
                ## Humanoid Large
                args.ev_n_comp = 6
                args.ev_cp_infer_t_type = 'interleave'
                # args.rd_resol = 400 ## or 1000
                # args.is_save_pkl = True
                # args.ep_st_idx = 80

                ## --------------------
                # args.ev_n_comp = 5 # 5
                # args.is_rd_agv = True
                # args.is_use_subgoal_marker = False
                # args.vid_fps = 120 ## humanoid
                # args.ep_st_idx = 60
                ## -------------------

                args.is_replan = 'ada_dist'
                repl_wp_cfg = {}
                args.n_act_per_waypnt = 1
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=10,
                    type='m_2',
                    ada_dist_minus_n_wp=300,
                    cond_2_extra=150,
                    n_max_steps=5000, ## default for eval
                )
                args.inv_epoch = int(8e5)
        
        ## Humanoid Medium
        elif 'medium' in args.dataset:
            if dfu_ndim == 23:
                raise NotImplementedError
            elif dfu_ndim == 2:
                ## Humanoid Medium
                args.ev_n_comp = 4

                args.ev_cp_infer_t_type = 'interleave' ## or 'gsc'
                # args.rd_resol = 1000
                # args.is_save_pkl = True
                # args.ep_st_idx = 80

                args.is_replan = 'ada_dist'
                repl_wp_cfg = {}
                
                args.n_act_per_waypnt = 1
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=10,
                    type='m_2',
                    ada_dist_minus_n_wp=300,
                    cond_2_extra=150,
                    n_max_steps=5000,
                )
            args.inv_epoch = int(8e5)
        else:
            raise NotImplementedError

    elif 'antsoccer' in args.dataset:
        ## Soccer
        if 'arena' in args.dataset:
            ## a 4D diffusion planner: ant x-y, ball x-y 
            if dfu_ndim == 4:
                repl_wp_cfg = {}
                args.ev_n_comp = 5

                args.ev_cp_infer_t_type = 'interleave'
                args.is_use_subgoal_marker = False
                args.ep_st_idx = 60
                ## teaser animation vis
                # args.rd_resol = 1200
                # args.is_rd_agv = True ## render agent view video
                # args.vid_fps = 60

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1 ##
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=50, ## 
                    n_max_steps=5000, ##
                    used_idxs=(0,1),
                )

                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

            elif dfu_ndim == 17:
                repl_wp_cfg = {}
                args.ev_n_comp = 5

                args.ev_cp_infer_t_type = 'interleave'
                # args.is_use_subgoal_marker = False
                
                args.ep_st_idx = 60
                ## teaser animation vis
                # args.rd_resol = 1200
                # args.is_rd_agv = True ## render agent view video
                # args.vid_fps = 60

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=50, ## 
                    n_max_steps=5000, ## ori default
                    used_idxs=(0,1),
                )

                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

        ## Soccer
        elif 'medium' in args.dataset:
            if dfu_ndim == 17:
                
                repl_wp_cfg = {}
                args.ev_n_comp = 6
                args.ep_st_idx = 0 ## 20
                args.is_replan = 'ada_dist'
                args.ev_cp_infer_t_type = 'interleave'
                args.rd_resol = 600
                args.is_save_pkl = True
                args.is_use_subgoal_marker = False


                args.n_act_per_waypnt = 2 ## 1 vis

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=6,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=10,
                    n_max_steps=5000,
                    used_idxs=(0,1,),
                )

                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

            ## ---------------------------------
            ## Soccer Medium
            elif dfu_ndim == 4:
                
                args.ev_cp_infer_t_type = 'interleave'
                args.rd_resol = 1000 # 600 # 1000
                args.is_save_pkl = True
                args.is_use_subgoal_marker = False
                ## for website
                args.ep_st_idx = 20
                args.is_rd_agv = True


                repl_wp_cfg = {}
                ## Jan 10 New Replan Method
                args.ev_n_comp = 8
                args.ev_n_comp = 5
                
                args.ev_n_comp = 6 ## Jan 17
                args.ev_n_comp = 7
                args.ev_n_comp = 8

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 2
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50, ## ?
                    cond_2_extra=50, ## 100?
                    n_max_steps=5000,
                    used_idxs=(0,1,),
                )

                # args.inv_epoch = int(12e5)
                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

    ## OGBench Point Maze Env
    elif 'pointmaze' in args.dataset.lower():

        ## ================= BALL + LiDAR closed-loop rollout =================
        ## (BALL.md Phases 5-6; the ball twin of the antmaze lidar branch above.
        ##  All knobs are env-driven so the L1-L4 eval-ladder rungs run without
        ##  code edits; defaults = L1/L2-style single-run with periodic replan.)
        if is_lidar_cfg:
            args.ep_st_idx = int(os.environ.get('LIDAR_EP_ST_IDX', '0'))
            repl_wp_cfg = {}
            args.n_act_per_waypnt = 1
            args.ev_cp_infer_t_type = 'interleave'
            ## composition count per maze size (matches the o2d branches below).
            ## NOTE on provisioning: the 20 rollout tasks (ev_probs block 0) need
            ## a BFS maze path of 120 wu, and at the data's PROGRESS pace (0.0755
            ## wu/frame) that is ~1590 frames => n_comp ~= 15. n_comp=8 is 888
            ## frames, i.e. ~1.8x UNDER-provisioned, so the plan can only satisfy
            ## its inpainted goal frame by covering ground faster than the data
            ## ever does. Overridable to test exactly that.
            _nc = os.environ.get('LIDAR_EV_N_COMP', '')
            if 'giant' in args.dataset:
                args.ev_n_comp = 8
            elif 'large' in args.dataset:
                args.ev_n_comp = 6
            else:
                args.ev_n_comp = 3
            if _nc:
                args.ev_n_comp = int(_nc)
            args.rd_resol = 300
            ## the ball rollout DECODES waypoint scans to (x, y), so the subgoal
            ## marker is meaningful again (the ant lidar branch had to disable it)
            args.is_use_subgoal_marker = True

            ## goal-scan heading: the ball has no yaw; scans are world-frame.
            args.lidar_goal_yaw = 0.0

            ## Replanning: ON by default (re-condition on the CURRENT scan when
            ## the plan is consumed). LIDAR_REPLAN=0 -> one plan per episode.
            if os.environ.get('LIDAR_REPLAN', '1') == '1':
                args.is_replan = 'periodic'
                args.repl_ada_dist_cfg = dict(
                    n_max_steps=int(os.environ.get('LIDAR_N_MAX_STEPS', '1000')),
                    max_n_repl=int(os.environ.get('LIDAR_MAX_N_REPL', '10')),
                    repl_every=int(os.environ.get('LIDAR_REPL_EVERY', '0')),
                )
            else:
                args.is_replan = False
                args.repl_ada_dist_cfg = dict(
                    n_max_steps=int(os.environ.get('LIDAR_N_MAX_STEPS', '1000')))

            ## goal band (inert while the planner was trained with K=1)
            args.goal_band_step = float(os.environ.get('LIDAR_GOAL_BAND_STEP', '4.0'))
            args.tf_band = int(os.environ.get('LIDAR_TF_BAND', '0'))
            args.tf_npz = os.environ.get('LIDAR_TF_NPZ', '')
            args.tf_band_stride = int(os.environ.get('LIDAR_TF_BAND_STRIDE', '1'))
            args.band_from_route = int(os.environ.get('LIDAR_BAND_FROM_ROUTE', '0'))

            ## --- localization grid (decode + rerank + arrival + gridnav) ---
            ## BALL DEFAULTS: no theta axis (n_theta=1) buys a 2x finer xy
            ## lattice than the ant grid for ~1/6 the poses.
            args.loc_grid_xy_step = float(os.environ.get('LIDAR_LOC_XYSTEP', '0.5'))
            args.loc_grid_n_theta = int(os.environ.get('LIDAR_LOC_NTHETA', '1'))
            args.loc_grid_sigma = float(os.environ.get('LIDAR_LOC_SIGMA', '1.0'))
            args.loc_n_decode = int(os.environ.get('LIDAR_LOC_SUBSAMPLE', '24'))
            args.loc_grid_cache = os.environ.get(
                'LIDAR_LOC_CACHE_ROOT', 'data/ogb_maze/lidar_grid')
            ## goal-cell source for the plan re-ranker ('true' = diagnostic
            ## ablation, 'lidar' = compliant); only read when LIDAR_PICK_TYPE=grid
            args.loc_goal_mode = os.environ.get('LIDAR_LOC_GOAL_MODE', 'true').lower()
            args.loc_motion_sigma = float(os.environ.get('LIDAR_LOC_MOTION_SIGMA', '1.0'))
            ## Sub-cell MAP: report the belief-weighted centroid of the +/-N cell window
            ## around the MAP instead of the raw lattice point. 0 = off (raw argmax).
            ## The belief's residual error at pos_lpf=0 is pure 0.5 wu quantization
            ## so this attacks the resolution floor, not the estimator.
            args.loc_subcell = int(os.environ.get('LIDAR_LOC_SUBCELL', '0'))

            ## --- LiDAR self-localization + goal-arrival termination ---
            args.lidar_arrival = int(os.environ.get('LIDAR_ARRIVAL', '0'))
            args.lidar_arrival_ends = int(os.environ.get('LIDAR_ARRIVAL_ENDS', '1'))
            args.lidar_arrival_goal_mode = os.environ.get(
                'LIDAR_ARRIVAL_GOAL_MODE', 'lidar').lower()
            args.lidar_arrival_persist = int(os.environ.get('LIDAR_ARRIVAL_PERSIST', '3'))

            ## --- grid navigator (rung L4 + check-and-move-on machinery) ---
            args.nav_mode = os.environ.get('LIDAR_NAV_MODE', '').lower()
            args.nav_goal_mode = os.environ.get('LIDAR_NAV_GOAL_MODE', 'lidar').lower()
            args.nav_step_wu = float(os.environ.get('LIDAR_NAV_STEP_WU', '0.13'))
            args.nav_yaw_smooth = int(os.environ.get('LIDAR_NAV_YAW_SMOOTH', '31'))
            args.nav_subgoal_cells = float(os.environ.get('LIDAR_NAV_SUBGOAL_CELLS', '10'))
            args.nav_min_conf = float(os.environ.get('LIDAR_NAV_MIN_CONF', '0.02'))
            args.nav_probe = int(os.environ.get('LIDAR_NAV_PROBE', '1'))
            args.nav_goal_hold = int(os.environ.get('LIDAR_NAV_GOAL_HOLD', '40'))
            args.nav_tf_min_frames = int(os.environ.get('LIDAR_NAV_TF_MIN', '120'))
            args.nav_tf_max_dyaw = float(os.environ.get('LIDAR_NAV_TF_MAX_DYAW', '60'))
            args.nav_pursuit = int(os.environ.get('LIDAR_NAV_PURSUIT', '1'))
            ## ball waypoint lookahead: ~4 frames (~0.5 wu) -- the ball tracks
            ## tightly; the ant needed 12-16 for its chunked executor.
            args.nav_lookahead = int(os.environ.get('LIDAR_NAV_LOOKAHEAD', '4'))
            args.nav_pursuit_win = int(os.environ.get('LIDAR_NAV_PURSUIT_WIN', '40'))
            args.nav_pursuit_back = int(os.environ.get('LIDAR_NAV_PURSUIT_BACK', '15'))
            args.nav_snap_mj = float(os.environ.get('LIDAR_NAV_SNAP_MJ', '3.0'))
            args.nav_corr_max = int(os.environ.get('LIDAR_NAV_CORR_MAX', '2'))
            args.nav_ribbon = os.environ.get('LIDAR_NAV_RIBBON', 'synth').lower()
            args.nav_a2_seg_frames = int(os.environ.get('LIDAR_NAV_A2_SEG', '12'))
            args.nav_a2_snap_wu = float(os.environ.get('LIDAR_NAV_A2_SNAP_WU', '2.0'))
            args.act_chunk_exec = int(os.environ.get('LIDAR_ACT_CHUNK_EXEC', '0'))
            args.nav_topk = int(os.environ.get('LIDAR_NAV_TOPK', '5'))
            args.nav_verify_hop = int(os.environ.get('LIDAR_VERIFY_HOP', '0'))
            args.nav_verify_tol = float(os.environ.get('LIDAR_VERIFY_TOL', '1.5'))
            args.nav_verify_hop_max = int(os.environ.get('LIDAR_VERIFY_MAX_HOPS', '6'))
            args.nav_verify_min_conf = float(os.environ.get('LIDAR_VERIFY_MIN_CONF', '0.0'))

            ## --- BALL executor (Arm A: decode + PID; see pid_controller.py) ---
            ## position source = the L1/L2 ('env') vs L3 ('belief') ladder knob
            args.loc_source = os.environ.get('LIDAR_LOC_SOURCE', 'env').lower()
            args.ball_kp = float(os.environ.get('LIDAR_BALL_KP', '5.0'))
            args.ball_kd = float(os.environ.get('LIDAR_BALL_KD', '0.0'))
            args.ball_ki = float(os.environ.get('LIDAR_BALL_KI', '0.0'))
            args.ball_ki_zone = float(os.environ.get('LIDAR_BALL_KI_ZONE', '2.0'))
            ## '' -> auto (0 for env-source, 0.7 for belief-source)
            args.ball_pos_lpf = os.environ.get('LIDAR_BALL_POS_LPF', '')
            args.ball_lookahead = int(os.environ.get('LIDAR_BALL_LOOKAHEAD', '4'))
            args.ball_decode_stride = int(os.environ.get('LIDAR_BALL_DECODE_STRIDE', '2'))
            ## plan scans -> xy. 'filter' = forward filter + per-step argmax
            ## (historical); 'viterbi' = max-product over paths, which cannot emit
            ## an unreachable step. Offline on 100 composed plans: cell-jumps
            ## p50 5 -> 0 with a LOWER scan-match residual (0.955 -> 0.806 wu/beam),
            ## so the teleports were a decode artifact, not model incoherence.
            ## Default 'filter' keeps an unset run bit-identical to the reference run.
            args.ball_decode_mode = os.environ.get(
                'LIDAR_BALL_DECODE_MODE', 'filter').lower()
            args.ball_decode_rmax = float(
                os.environ.get('LIDAR_BALL_DECODE_RMAX', '3.0'))
            ## per-segment aim poses. Offline this is the single largest effect
            ## measured on the composed plan (ev_probs G2b 44% -> 93%, end err
            ## 17.6 -> 0.5 wu, BFS progress 17/83 -> 50/50); default OFF so an
            ## unset run still reproduces the historical rollout numbers.
            args.ball_waypoints = int(os.environ.get('LIDAR_BALL_WAYPOINTS', '0'))
            ## pursuit: 'time' (historical) vs 'progress' (BALL.md 7.2). Default
            ## stays 'time' so an unset run reproduces the old rollout numbers.
            args.ball_pursuit = os.environ.get('LIDAR_BALL_PURSUIT', 'time').lower()
            args.ball_pursuit_win = int(os.environ.get('LIDAR_BALL_PURSUIT_WIN', '60'))
            ## 0 = index lookahead (historical); >0 = look this many wu ALONG
            ## the plan, which is pace-independent (progress arm only).
            args.ball_lookahead_wu = float(os.environ.get('LIDAR_BALL_LOOKAHEAD_WU', '0'))
            ## control steps with no cursor advance before the progress cursor is
            ## forced forward past an unreachable plan point. 0 disables.
            args.ball_stall_steps = int(os.environ.get('LIDAR_BALL_STALL_STEPS', '50'))
            ## line-of-sight pursuit: never aim at a plan point the ball cannot
            ## drive at in a straight line (benchmark post-mortem -- 7/20 episodes
            ## spent an entire 888-frame plan pinned against one wall). Default
            ## OFF so an unset run is bit-identical to 45675.
            args.ball_los = int(os.environ.get('LIDAR_BALL_LOS', '0'))
            args.ball_los_margin = float(os.environ.get('LIDAR_BALL_LOS_MARGIN', '0.6'))
            ## cursor runaway (arm I post-mortem): the stall escape above
            ## marched the cursor +1/step on 25% of all control steps while the
            ## LOS seek meant to rescue it fired on 0.9%, so the carrot ended up
            ## p50 6.3 wu (p90 25.5) from the ball. `cursor_leash_wu` stops the
            ## blind march past that distance and re-acquires backward instead;
            ## `max_carrot_wu` caps the carrot by straight-line distance rather
            ## than plan ARC. Both default 0 = OFF (bit-identical to the reference run).
            args.ball_cursor_leash_wu = float(os.environ.get('LIDAR_BALL_CURSOR_LEASH_WU', '0'))
            args.ball_max_carrot_wu = float(os.environ.get('LIDAR_BALL_MAX_CARROT_WU', '0'))
            args.ball_pursuit_back = int(os.environ.get('LIDAR_BALL_PURSUIT_BACK', '15'))
            ## frozen time (2026-08-01, ball_step_budget.py on the baseline arm). At
            ## OGBench's own 1000-step budget 51% of control steps move the ball
            ## ZERO wu, and removing that dead time takes the SAME plans from
            ## 45.2 to 74.2. Three knobs, all default 0 = OFF (bit-identical to
            ## the reference run):
            ##   MIN_CARROT_WU  refuse a carrot closer than this -- `lookahead_wu`
            ##     is ARC along a plan that loiters, so it can land 0.19 wu from
            ##     the ball, giving |u| ~ 0 and a cursor that cannot advance
            ##     because the ball does not move (17.7% of the budget).
            ##   FROZEN_STEPS   fire the stall escape on measured zero DISPLACEMENT
            ##     instead of on cursor non-advance; only ever earlier, never later
            ##     (the timer latency alone is 18.6% of the budget).
            ##   WALL_SLIDE     when frozen at a saturated command, deflect along a
            ##     free heading from the LIVE SCAN (84.7% of pinned steps have one
            ##     that still points goal-ward). Scan only: no map, no BFS.
            args.ball_min_carrot_wu = float(os.environ.get('LIDAR_BALL_MIN_CARROT_WU', '0'))
            args.ball_frozen_steps = int(os.environ.get('LIDAR_BALL_FROZEN_STEPS', '0'))
            args.ball_frozen_eps = float(os.environ.get('LIDAR_BALL_FROZEN_EPS', '0.02'))
            args.ball_wall_slide = int(os.environ.get('LIDAR_BALL_WALL_SLIDE', '0'))
            args.ball_slide_wu = float(os.environ.get('LIDAR_BALL_SLIDE_WU', '1.0'))
            ## deadlock (zigzag) detector -- S5: flag, dump, consult the team
            args.ball_deadlock_win = int(os.environ.get('LIDAR_BALL_DEADLOCK_WIN', '100'))
            args.ball_deadlock_disp = float(os.environ.get('LIDAR_BALL_DEADLOCK_DISP', '0.5'))
            args.ball_deadlock_umin = float(os.environ.get('LIDAR_BALL_DEADLOCK_UMIN', '0.3'))
            args.ball_log_npz = int(os.environ.get('LIDAR_BALL_LOG_NPZ', '0'))

            ## --- Arm B (optional learned inv-dyn executor): set the run dir ---
            args.inv_model_path = os.environ.get('LIDAR_INV_MODEL_PATH', '')
            args.inv_epoch = os.environ.get('LIDAR_INV_EPOCH', 'latest')

            utils.print_color(
                f'[ball lidar rollout] is_replan={args.is_replan} '
                f'loc_source={args.loc_source} '
                f'grid=(xy{args.loc_grid_xy_step},th{args.loc_grid_n_theta}) '
                f'pid=(kp{args.ball_kp},kd{args.ball_kd},ki{args.ball_ki}) '
                f'lookahead={args.ball_lookahead} waypoints={args.ball_waypoints} '
                f'nav={args.nav_mode or "-"} '
                f'arrival={args.lidar_arrival} '
                f"inv={'B:' + args.inv_model_path if args.inv_model_path else 'A:pid'}",
                c='c')

        ## Point Maze Giant
        elif 'giant' in args.dataset:
            # args.ev_n_comp = 9
            args.ev_n_comp = 8
            
            # args.ev_cp_infer_t_type = 'same_t' ## parallel
            args.ev_cp_infer_t_type = 'interleave' ## default
            # args.ev_cp_infer_t_type = 'gsc' ## gsc baseline
            # args.ev_cp_infer_t_type = 'same_t_p'
            # args.ev_cp_infer_t_type = 'ar_back' ## backward autoregressive

            # args.rd_resol = 1000
            # args.is_save_pkl = True ## save rollout stats
            
            # args.ep_st_idx = 40 ## starting eval problem idx, default is 0

            ## 
            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                max_n_repl=10,
                # max_n_repl=0, ## no repl abl
                thres=1,
                type='m_2',
                ada_dist_minus_n_wp=10,
                cond_2_extra=150, ##
                n_max_steps=1000, ## 
            )
            args.inv_epoch = int(8e5)

        elif 'large' in args.dataset:
            args.ev_n_comp = 5 ##
            args.ev_n_comp = 6 ## either is fine

            args.ev_cp_infer_t_type = 'interleave'

            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                # max_n_repl=10,
                max_n_repl=0, ## no replan
                thres=1,
                type='m_2',
                ada_dist_minus_n_wp=0,
                cond_2_extra=150, ##
                n_max_steps=1000, ## 
            )
            args.inv_epoch = int(8e5)
            

        elif 'medium' in args.dataset:
            args.ev_n_comp = 3 ## or 4

            args.ev_cp_infer_t_type = 'interleave' ## or 'same_t_p'

            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                # max_n_repl=10,
                max_n_repl=0, ## no replan ablation
                thres=1, ## 4
                type='m_2',
                ada_dist_minus_n_wp=0,
                cond_2_extra=150,
                n_max_steps=1000,
            )
            args.inv_epoch = int(8e5)

    else: 
        raise NotImplementedError
    
    ## Plan-sampling + selection (eval-only; env overrides mirror the LIDAR_* knobs).
    ## LIDAR_PICK_TYPE='goal' turns on the LiDAR-only goal-aware re-rank of the top_n
    ## (pick the candidate whose body arrives closest to the goal). Raise LIDAR_TOP_N /
    ## LIDAR_BSIZE so a good-corridor draw is actually generated AND in the re-rank pool.
    args.b_size_per_prob = int(os.environ.get('LIDAR_BSIZE', '40'))
    args.ev_top_n = int(os.environ.get('LIDAR_TOP_N', '5'))
    args.ev_pick_type = os.environ.get('LIDAR_PICK_TYPE', 'first')
    args.ev_goal_pick_tail = int(os.environ.get('LIDAR_GOAL_PICK_TAIL', '8'))
    args.tjb_blend_type = 'exp'
    args.tjb_exp_beta = 2


    ### Diffusion Sampling Hyper-param
    args.var_temp = 1.0 ## or 0.5
    ## LIDAR_COND_W overrides classifier-free guidance weight (default 2.0).
    ## Only the LiDAR sweep sets it; every other run keeps 2.0 (env var absent).
    args.cond_w = float(os.environ.get('LIDAR_COND_W', '2.0'))
    args.use_ddim = True
    args.ddim_eta = 1.0
    args.ddim_steps = 50
    
    args.repl_wp_cfg = repl_wp_cfg

    latest_e = utils.get_latest_epoch(loadpath)
    # n_e = round(latest_e // 1e5) + 1 # all
    # start_e = 5e5; # 2e5 end_e =
    # depoch_list = np.arange(start_e, int(n_e * 1e5), int(1e5), dtype=np.int32).tolist()

    ## LIDAR_DIFF_EPOCH pins the planner checkpoint step (e.g. an intermediate ckpt
    ## of a still-training run). Empty/absent -> latest (original behavior). The
    ## loaded step is recorded in the rollout json as `epoch_diffusion` -- always
    ## check it matches what you meant to evaluate.
    _de = os.environ.get('LIDAR_DIFF_EPOCH', '').strip()
    depoch_list = [int(float(_de))] if _de else [latest_e,]
    ## depoch_list = [800000,] # 1M
    
    if args.is_replan == 'ada_dist':
        args.env_n_max_steps = args.repl_ada_dist_cfg['n_max_steps']
    else:
        args.env_n_max_steps = None ## use ogb default ??


    ## no-replan (e.g. LiDAR) leaves env_n_max_steps=None; only used for the name
    _ems_k = (args.env_n_max_steps // 1000) if args.env_n_max_steps is not None else 0
    ## SLURM job id disambiguates concurrent submissions: the timestamp alone is
    ## only millisecond-resolution, and a 4-way sweep launched in one loop DID
    ## collide (2026-07-30, a benchmark run died on FileExistsError before doing any
    ## work). It also makes a run dir traceable back to its slurm log.
    _jid = os.environ.get('SLURM_JOB_ID', '')
    sub_dir = f'{datetime.now().strftime("%y%m%d-%H%M%S-%f")[:-3]}' + \
                        (f'-j{_jid}' if _jid else '') + \
                        f"-nm{int(args.plan_n_ep)}-ems{_ems_k}k" + \
                        f"-ncp{args.ev_n_comp}" + f"-{args.ev_cp_infer_t_type}"\
                        f"-evSd{','.join( [str(sd) for sd in args.pl_seeds] )}"
    
    # pdb.set_trace()
    ## f'-vt{args.var_temp}'
    if args.is_save_pkl:
        sub_dir += '-pkl'
    if hasattr(args, 'ep_st_idx'):
        sub_dir += f'-st{args.ep_st_idx}'
    if args.is_rd_agv:
        sub_dir += '-agv'

    args.savepath = osp.join(args.savepath, sub_dir)

    result_list = []
    for i in range(len(depoch_list)):
        args_train.diffusion_epoch = depoch_list[i]
        args.diffusion_epoch = depoch_list[i]
        tmp = main( copy.deepcopy(args_train),  copy.deepcopy(args) )
        
        result_list.append(tmp)
    

