"""
BALL (pointmaze) LiDAR inverse dynamics, ACTION-CHUNK ablation (m=8).
Train ONLY if the h12 single-action twin fails the G3 goal-sensitivity probe
(prediction: it will not fail; ball actions are ~closed-form in the goal).
-- Arm B of BALL.md Phase 4 (OPTIONAL:
the default ball executor is the analytic decode+PID pipeline, Arm A, which
needs no learned model at all).

Maps (state, goal scan) -> the 2-D pointmaze action:
    obs  = [x, y, vx, vy, lidar_0..lidar_29]   ('xyv_lidar', 34-D)
    goal = a future LiDAR scan at k ~ U[4, 8] steps ahead, DIFF-encoded
           (normed goal scan - normed current scan), so "which way to go" is
           directly readable and "you have arrived" is literally zero.
    act  = the next 8 actions, flattened (receding-horizon executor plays
           the first LIDAR_ACT_CHUNK_EXEC of them, default m/2=4).

Why these choices (vs. the ant's chunk16 saga): the ball action is a clipped
velocity command -- the optimal action is ~a closed-form function of the goal
direction, so I(goal; a_1 | state) is already large at k=1 and chunking should
NOT be needed. This config is the clean test of that prediction; the
`_chunk8` twin is the ablation if the goal-sensitivity probe (G3) fails here.

Velocities are per-episode finite differences (see
ogb_lidar_utils.finite_diff_velocities): OGBench pointmaze observations are
position-only. The rollout builds its live obs the same way (v_t = xy_t -
xy_{t-1}, v=0 at episode start) -- keep the two in sync.

ACCEPTANCE (gate G3, run BEFORE any closed-loop rollout):
    python -m diffuser.pl_eval.lidar_eval.lidar_invdyn_goal_sensitivity \
        --inv_model_path logs/pointmaze-giant-stitch-v0/diffusion/og_pntM_Gi_lidar30_xyv_g30d_invdyn_h16_chunk8 \
        --inv_epoch latest --device cpu --k 6 \
        --out logs/lidar_eval/ball_invdyn_goal_sensitivity_chunk8.json
    PASS = goal-shuffle action delta >= ~0.4 (normalized); expected to pass
    easily -- if it does not, drop Arm B and use the PID executor.
"""

import os.path as osp
import numpy as np

from diffuser.utils import watch

config_fn = osp.splitext(osp.basename(__file__))[0]

diffusion_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

plan_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ('conditional', 'cond'),
]


tot_horizon = 16                              # window >= max(goal_k_max, chunk) + 1
ACT_CHUNK = 8                                 # action-chunk ablation (v2b, ant-style)

## ---- LiDAR + ball-state observation settings ----
N_BEAMS = 30
STATE_DIM = 4                                 # x, y, vx, vy (finite-diff vel)
LIDAR_CONFIG = dict(
    n_beams=N_BEAMS,
    max_range=12.0,
    fov=2.0 * np.pi,
    step=0.1,
    obs_layout='xyv_lidar',                   # [x, y, vx, vy, lidar...]
    use_cache=True,                           # reuses the SAME cached scans
)

obs_dim = STATE_DIM + N_BEAMS                                  # 34
goal_sel_idxs = tuple(range(STATE_DIM, STATE_DIM + N_BEAMS))   # goal = LiDAR scan (30)


base = {
    'dataset': "pointmaze-giant-stitch-v0",

    'diffusion': {
        'config_fn': '',

        'tot_horizon': tot_horizon,
        'goal_sel_idxs': goal_sel_idxs,

        'model': 'ogb_task.og_inv_dyn.MLP_InvDyn_OgB_V3',
        'act_chunk': ACT_CHUNK,
        'inv_m_config': dict(
            hidden_dims=[512, 512, 512],
            final_fc_init_scale=1e-2,
            is_out_dist=False,
        ),
        'act_net_config': dict(
            act_f='gelu',
            use_dpout=False,
        ),

        ## goal at k ~ U[4, 8] steps ahead (matches the executor's waypoint
        ## lookahead ~3-5 frames), diff-encoded; target = the next action.
        'trainer_dict': dict(
            goal_horizon_mode='fixed',
            fixed_goal_k=6,
            goal_k_max=12,
            goal_encoding='diff',
            act_chunk=ACT_CHUNK,
        ),

        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset (LiDAR inverse-dynamics loader, xyv_lidar layout)
        'loader': 'datasets.ogb_dset.OgB_Lidar_InvDyn_SeqDataset_V1',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        ## make_indices needs max_path_length - horizon >= path_length - 1
        ## (the ant h64 lesson). Pointmaze stitch episode length is <= 141 by the
        ## same constraint on the upstream planner config (300 - 160); 300 here is
        ## therefore safely generous. Verified again at load time by the dataset's
        ## own asserts.
        'max_path_length': 300,
        'max_n_episodes': 20000,
        'dataset_config': dict(
            obs_select_dim=tuple(range(obs_dim)),  # ignored; lidar_config drives layout
            dset_type='ogb',
            lidar_config=LIDAR_CONFIG,
        ),

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training (expect convergence long before 2e6; G3-probe intermediate
        ## checkpoints from ~2e5 on)
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2_inv_v3',
        'n_train_steps': 2e6,

        'ema_decay': 0.995,
        'batch_size': 1024,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 1,

        'sample_freq': 4000,
        'save_freq': 4000,
        'n_saves': 5,

        'n_reference': 20,
        'n_samples': 8,

        'device': 'cuda',
    },

    'plan': {
        'config_fn': '',

        'batch_size': 1,
        'device': 'cuda',

        'horizon': tot_horizon,
        'n_diffusion_steps': 512,
        'normalizer': 'LimitsNormalizer',

        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}
