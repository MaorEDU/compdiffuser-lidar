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
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_ball_planner_v1 import (
    OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1,
)




class Parser(utils.Parser):
    dataset: str = None
    config: str = None
    ## should not put any existing variable in config file here
    pl_seeds: str = '-1' # no seed
    plan_n_ep: int = -100 ## all if -100, auto parse to int

def main(args_train, args):
    
    #---------------------------------- setup ----------------------------------#

    ld_config = dict()

    ## LiDAR ball runs get the scan-space rollout; everything else is untouched.
    is_lidar = args_train.dataset_config.get('lidar_config', None) is not None
    planner_cls = (OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1 if is_lidar
                   else OgB_Stgl_Sml_MazeEnvPlanner_V1)
    ogmz_planner = planner_cls(args_train, args=args)
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


def setup_eval_args(args_train, args):
    """Fill in every evaluation hyper-parameter for this dataset/model.

    Split out of ``__main__`` so the OFFLINE gates (plan_decode_gate,
    hist_holdout_gate) construct their planner from the exact same settings the
    rollout uses. A gate configured differently from the run it gates measures
    something else -- which is how the previous fork ended up evaluating an mr24
    checkpoint with an mr12 config and voiding a whole experiment.

    Returns ``repl_wp_cfg`` (the at_given_t replan schedule; ``{}`` everywhere
    except the legacy antmaze branches).
    """
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


    ## ---------------------------------------
    ## ----------- Ant Maze Stitch -----------
    if 'antmaze' in args.dataset.lower() and 'stitch' in  args.dataset.lower():
        if 'giant' in args.dataset:

            ## Ant Maze Giant
            if dfu_ndim == 2:
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

    ## ---------------------------------------------------------------------
    ## BALL + LiDAR (comp_diffuser_ball_v2). Selected by the dataset config, not by
    ## the env name, so the (x, y) pointmaze branch below stays bit-exact.
    ##
    ## The two experimental knobs:
    ##   LIDAR_REPL_EVERY=N  -> replan every N control steps ('periodic'). 0 keeps the
    ##                          paper's adaptive-distance trigger, which is arm C.
    ##                          n_act_per_waypnt == 1, so N really is env steps.
    ##   LIDAR_BALL_HIST=1   -> condition each (re)plan on the last hist_len OBSERVED
    ##                          scans. 0 drops them: the no-history control, same weights.
    ## and the ladder knob:
    ##   LIDAR_LOC_SOURCE=env|belief -> PID closes on the true position (L1) or on the
    ##                          live histogram filter's estimate (L3, the deliverable).
    ##
    ## The step budget default is 1000 because that is what OGBench registers for
    ## pointmaze-giant and success LATCHES, so a longer budget monotonically buys score.
    ## Anything reportable runs at 1000; 2500 only as a clearly labelled ablation.
    elif args_train.dataset_config.get('lidar_config', None) is not None:
        assert 'pointmaze' in args.dataset.lower(), \
            f'the LiDAR stack in this repo is the BALL model only, got {args.dataset}'
        _envi = lambda k, d: int(os.environ.get(k, d))
        _envf = lambda k, d: float(os.environ.get(k, d))

        args.ev_n_comp = _envi('LIDAR_N_COMP', 8)
        args.ev_cp_infer_t_type = 'interleave'   ## the only mode history is wired into
        args.n_act_per_waypnt = 1
        args.rd_resol = _envi('LIDAR_RD_RESOL', 300)
        args.inv_epoch = None                    ## no learned executor in this repo

        repl_wp_cfg = {}
        _repl_every = _envi('LIDAR_REPL_EVERY', 0)
        _n_max_steps = _envi('LIDAR_N_MAX_STEPS', 1000)
        if _repl_every > 0:
            ## The replan BUDGET must cover the cadence, or the cadence silently stops
            ## partway through the episode and the arm is not the arm you asked for:
            ## at REPL_EVERY=50 over 1000 steps you need 19 replans, and a fixed
            ## max_n_repl=15 would truncate every episode at step 750. So size it from
            ## the budget by default (+2 slack for plan-exhaustion fires), and let
            ## LIDAR_MAX_N_REPL override deliberately.
            _auto_repl = -(-_n_max_steps // _repl_every) + 2
            args.is_replan = 'periodic'
            args.repl_ada_dist_cfg = dict(
                repl_every=_repl_every,
                max_n_repl=_envi('LIDAR_MAX_N_REPL', _auto_repl),
                n_max_steps=_n_max_steps,
            )
        else:
            ## arm C: the paper's own pointmaze-giant replan trigger, verbatim
            args.is_replan = 'ada_dist'
            args.repl_ada_dist_cfg = dict(
                max_n_repl=_envi('LIDAR_MAX_N_REPL', 10),
                thres=1,
                type='m_2',
                ada_dist_minus_n_wp=10,
                cond_2_extra=150,
                n_max_steps=_n_max_steps,
                ## INERT on this path. The LiDAR planner overrides _get_ada_dist
                ## with a world-unit distance off the decoded plan; this key only
                ## exists to satisfy the `len(used_idxs) in [2,4]` assert in the
                ## base setup. Do NOT read it as "the trigger uses beams 0 and 1".
                used_idxs=(0, 1),
            )

        ## --- ladder + experiment knobs ---
        args.loc_source = os.environ.get('LIDAR_LOC_SOURCE', 'env')
        args.ball_hist = _envi('LIDAR_BALL_HIST', 0)
        ## Short runs MUST be able to start off block 0. The 100 eval problems are
        ## 5 blocks of 20 and each block is ONE start-goal pair jittered ~1.2 wu, so
        ## a 4-episode run at the default start index is 4 samples of one route.
        ## A full N_EP=100 bench leaves this at 0 (the base loop asserts that).
        args.ep_st_idx = _envi('LIDAR_EP_ST_IDX', 0)
        ## --- perception ---
        args.loc_grid_xy_step = _envf('LIDAR_LOC_XY_STEP', 0.5)
        args.loc_grid_n_theta = _envi('LIDAR_LOC_NTHETA', 1)   ## the ball has no heading
        args.loc_grid_sigma = _envf('LIDAR_LOC_SIGMA', 1.0)
        args.loc_motion_sigma = _envf('LIDAR_LOC_MOTION_SIGMA', 1.0)
        ## Uniform mixing in the LIVE belief's predict step. `loc_err_p50` is bimodal
        ## -- 0.25 wu on 86% of episodes and ~45 wu on 14% -- and the multiplicative
        ## Bayes update has no floor, so a wrong hypothesis can never recover. 0.0 = off
        ## = bit-identical to every run before 2026-09-04. See `LiveLocalizer.update`.
        args.loc_eps_mix = _envf('LIDAR_LOC_EPS_MIX', 0.0)
        ## --- executor ---
        args.ball_kp = _envf('LIDAR_BALL_KP', 5.0)
        args.ball_pursuit_win = _envi('LIDAR_BALL_PURSUIT_WIN', 60)
        args.ball_lookahead_wu = _envf('LIDAR_BALL_LOOKAHEAD_WU', 2.0)
        args.ball_stall_steps = _envi('LIDAR_BALL_STALL_STEPS', 5)
        ## LOS shortcut window (0 = off, bit-identical to the pre-2026-09-03 executor).
        ## See `_advance_cursor`: the decoded plan is ~315 wu of arc against a 200 wu
        ## physical budget, and string-pulling it is what makes it drivable.
        args.ball_los_shortcut_win = _envi('LIDAR_BALL_LOS_SHORTCUT_WIN', 0)
        args.ball_los_shortcut_stride = _envi('LIDAR_BALL_LOS_SHORTCUT_STRIDE', 4)
        ## Plan string-pull at decode time -- the CORRECTED form of the shortcut
        ## (the runtime cursor version above is refuted; see `_pull_plan`).
        args.ball_pull_win = _envi('LIDAR_BALL_PULL_WIN', 0)
        args.ball_pull_stride = _envi('LIDAR_BALL_PULL_STRIDE', 4)
        args.ball_pull_resample_wu = _envf('LIDAR_BALL_PULL_RESAMPLE_WU', 0.4)
        ## the fork's two executor guards + adaptive composition depth (all 0 = off)
        args.ball_max_carrot_wu = _envf('LIDAR_BALL_MAX_CARROT_WU', 0.0)
        args.ball_cursor_leash_wu = _envf('LIDAR_BALL_CURSOR_LEASH_WU', 0.0)
        args.ball_pursuit_back = _envi('LIDAR_BALL_PURSUIT_BACK', 15)
        args.ball_ada_ncomp = _envi('LIDAR_BALL_ADA_NCOMP', 0)
        args.ball_ada_ncomp_cells = _envf('LIDAR_BALL_ADA_NCOMP_CELLS', 3.4)
        ## Goal capture (0 = off): within this many wu of the agent's OWN decoded goal,
        ## and with clear line of sight to it, drive straight at it instead of chasing
        ## plan indices. 12 wu = 3 cells = the measured near-miss shelf. Release radius
        ## defaults to 1.5x. See `_capture_target`.
        args.ball_capture_wu = _envf('LIDAR_BALL_CAPTURE_WU', 0.0)
        args.ball_capture_release_wu = _envf('LIDAR_BALL_CAPTURE_RELEASE_WU', 0.0)
        ## Goal re-entry (0 = off): within this many wu of the decoded goal, snap the
        ## cursor forward into the plan's own approach to it. Needed because the goal
        ## is visible from only ~19% of positions at 12 wu (6% on block 4), so capture
        ## alone almost never engages -- but the plan gets through the corner.
        args.ball_goal_reentry_wu = _envf('LIDAR_BALL_GOAL_REENTRY_WU', 0.0)
        ## Capture's own LOS margin (-1 = inherit ball_los_margin = bit-identical).
        ## 0.6 blocks 17% of sightlines at point-blank range and a third at 4-6 wu.
        args.ball_capture_margin = _envf('LIDAR_BALL_CAPTURE_MARGIN', -1.0)
        ## Release a capture after this many steps without getting closer (0 = off).
        args.ball_capture_stuck_steps = _envi('LIDAR_BALL_CAPTURE_STUCK_STEPS', 0)
        ## FIX 11, confidence-triggered replan (0 = off): replan when the cursor reaches
        ## the plan's first run of RUN decoded frames whose argmin margin is under THR.
        ## Requires LIDAR_DECODE=argmin, PULL_WIN=0 and REPL_EVERY>0. See `_conf_flag`.
        args.ball_conf_repl_thr = _envf('LIDAR_BALL_CONF_REPL_THR', 0.0)
        args.ball_conf_repl_run = _envi('LIDAR_BALL_CONF_REPL_RUN', 3)
        args.ball_conf_repl_min_gap = _envi('LIDAR_BALL_CONF_REPL_MIN_GAP', 20)
        args.ball_conf_repl_extra = _envi('LIDAR_BALL_CONF_REPL_EXTRA', 10)
        args.ball_blind_walk_steps = _envi('LIDAR_BALL_BLIND_WALK_STEPS', 50)
        args.ball_los_margin = _envf('LIDAR_BALL_LOS_MARGIN', 0.6)
        args.ball_decode_stride = _envi('LIDAR_BALL_DECODE_STRIDE', 2)
        ## --- plan decoder ---
        ## 'argmin'  per-frame nearest grid pose. THE DEFAULT since 2026-08-24.
        ## 'filter'  the forward histogram filter -- what every number before that
        ##           date used. Kept so the old artifacts stay reproducible; BALLV2
        ##           section 5 forbids comparing G2 across decoders, so the choice is
        ##           logged into every rollout json and every gate output.
        ## Measured on 40 real generated plans, all 5 blocks (plan_motion_census):
        ## endpoint within one cell -- filter@sigma1.0 48%, filter@sigma4.0 82%,
        ## argmin 90%. The filter's worst per-block p90 is 40 wu; argmin's is 0.6 wu
        ## outside block 1, whose 50% is single-scan aliasing that only max_range fixes.
        args.ball_decode = os.environ.get('LIDAR_DECODE', 'argmin')
        args.ball_decode_motion_sigma = _envf('LIDAR_DECODE_MOTION_SIGMA',
                                              args.loc_motion_sigma)
        ## rolling-median de-jitter for the argmin decode; 1 disables it
        args.ball_decode_median = _envi('LIDAR_DECODE_MEDIAN', 15)
        ## 'viterbi' beam-DP over the free-space graph. Added 2026-08-30 after the
        ## composer was measured and cleared (RETRAIN_2026-08-29 section 6): argmin is
        ## memoryless and the filter's motion model is a Gaussian on the lattice, so
        ## neither can express "stay in the corridor you are in" and neither passes
        ## G2a. This one makes an unreachable pose INADMISSIBLE rather than unlikely.
        ##   max_step  hard cap on the move between two DECODED (strided) frames, wu.
        ##             Ground-truth motion is ~0.20 wu/frame, so at stride 2 the real
        ##             step is ~0.4 wu and 2.0 already allows ~3.5x the p99.
        ##   beam      states kept per frame. This is NOT a speed knob: too small a
        ##             beam drops the legal predecessor of a surviving state, which
        ##             forces a restart, and a restart is a teleport -- restarts are
        ##             the ONLY source of jumps this decoder has. Measured on 80 r24
        ##             plans at max_step=2: beam 512 -> jump 0.0117 (49/80 over bar),
        ##             1024 -> 0.0077 (20/80), 2048 -> 0.0000 (0/80). Do not lower it
        ##             for speed; 2048 costs ~0.5 s/plan against ~40 s of sampling.
        args.ball_decode_max_step = _envf('LIDAR_DECODE_MAX_STEP', 2.0)
        args.ball_decode_beam = _envi('LIDAR_DECODE_BEAM', 2048)
        args.ball_decode_w_step = _envf('LIDAR_DECODE_W_STEP', 0.0)
        args.ball_decode_sigma_e = _envf('LIDAR_DECODE_SIGMA_E', 1.0)

        utils.print_color(
            f'[BALLv2 CONFIG] replan={args.is_replan} every={_repl_every} '
            f'max_n_repl={args.repl_ada_dist_cfg["max_n_repl"]} '
            f'n_max_steps={_n_max_steps} n_comp={args.ev_n_comp} '
            f'loc_source={args.loc_source} hist={args.ball_hist} '
            f'decode={args.ball_decode}/med{args.ball_decode_median}'
            + (f'/step{args.ball_decode_max_step:g}/beam{args.ball_decode_beam}'
               if args.ball_decode == 'viterbi' else ''), c='y')

    ## OGBench Point Maze Env
    elif 'pointmaze' in args.dataset.lower():
        
        ## Point Maze Giant
        if 'giant' in args.dataset:
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
    
    ## 
    args.b_size_per_prob = int(os.environ.get('LIDAR_BSIZE', 40))
    args.ev_top_n = int(os.environ.get('LIDAR_TOP_N', 5))
    args.ev_pick_type = os.environ.get('LIDAR_PICK_TYPE', 'first')
    args.tjb_blend_type = 'exp'
    args.tjb_exp_beta = 2


    ### Diffusion Sampling Hyper-param
    args.var_temp = 1.0 ## or 0.5
    args.cond_w = 2.0
    args.use_ddim = True
    args.ddim_eta = 1.0
    args.ddim_steps = 50
    
    args.repl_wp_cfg = repl_wp_cfg

    return repl_wp_cfg


if __name__ == '__main__':
    ## training args
    args_train = Parser().parse_args('diffusion')
    args = Parser().parse_args('plan')
    ## 1. get epoch to eval on, by default all
    loadpath = args.logbase, args.dataset, args_train.exp_name

    args.pl_seeds = utils.parse_seeds_str(args.pl_seeds) ## a list of int
    args.n_batch_acc_probs = 4 ##
    
    ### --- Hyper-parameters Setup ---
    repl_wp_cfg = setup_eval_args(args_train, args)


    latest_e = utils.get_latest_epoch(loadpath)
    # n_e = round(latest_e // 1e5) + 1 # all
    # start_e = 5e5; # 2e5 end_e = 
    # depoch_list = np.arange(start_e, int(n_e * 1e5), int(1e5), dtype=np.int32).tolist()
    
    depoch_list = [latest_e,]
    ## depoch_list = [800000,] # 1M
    
    if args.is_replan in ('ada_dist', 'periodic'):
        args.env_n_max_steps = args.repl_ada_dist_cfg['n_max_steps']
    else:
        args.env_n_max_steps = None ## use ogb default ??


    _ems = 'def' if args.env_n_max_steps is None else f'{args.env_n_max_steps // 1000}k'
    sub_dir = f'{datetime.now().strftime("%y%m%d-%H%M%S-%f")[:-3]}' + \
                        f"-nm{int(args.plan_n_ep)}-ems{_ems}" + \
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
    if args_train.dataset_config.get('lidar_config', None) is not None:
        ## the ARM is part of the run's identity: four arms differ only by two env
        ## vars, and a folder that does not say which one it was is unreadable later.
        sub_dir += (f"-{os.environ.get('LIDAR_LOC_SOURCE', 'env')}"
                    f"-hist{os.environ.get('LIDAR_BALL_HIST', '0')}"
                    f"-re{os.environ.get('LIDAR_REPL_EVERY', '0')}")

    args.savepath = osp.join(args.savepath, sub_dir)

    result_list = []
    for i in range(len(depoch_list)):
        args_train.diffusion_epoch = depoch_list[i]
        args.diffusion_epoch = depoch_list[i]
        tmp = main( copy.deepcopy(args_train),  copy.deepcopy(args) )
        
        result_list.append(tmp)
    

