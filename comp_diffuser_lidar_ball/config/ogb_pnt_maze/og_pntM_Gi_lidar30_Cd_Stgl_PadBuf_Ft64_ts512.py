"""
CompDiffuser planner for ``pointmaze-giant-stitch-v0`` (the BALL model) trained
on **LiDAR** observations instead of the (x, y) position.

This is the LiDAR counterpart of
``config/ogb_pnt_maze/og_pntM_Gi_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py`` and the ball
twin of ``config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py``.

The only conceptual change vs. the o2d ball config is the observation space: a
30-beam, 360-degree, WORLD-FRAME LiDAR scan (the ball has no heading, so yaw==0
everywhere -- ``obs_to_xy_yaw`` auto-detects this from the 2-D pointmaze obs).
The model never receives the (x, y) coordinate: it denoises *sequences of LiDAR
scans*, conditioned on a start scan and a goal scan.

Key edits vs. the o2d ball config (same minimal diff as the ant lidar config):
  * dataset_config['lidar_config']  -> turns on the LiDAR transform.
  * ovlp_model_config['in_dim']     -> N_BEAMS (was 2 for x, y).
  * loader / trainer_cls            -> LiDAR-aware variants.
  * goal_band_k = 1                 -> single goal scan (BALL.md Phase 3: measure
                                       the decoy problem FIRST, only then decide
                                       on a K>1 band retrain).
The diffusion U-Net adapts automatically: its transition_dim is read from
dataset.observation_dim, which is now N_BEAMS.
"""

import os.path as osp
import numpy as np

from diffuser.utils import watch

#------------------------ base ------------------------#

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

## ---- LiDAR observation settings (world-frame: the ball has no yaw) ----
N_BEAMS = 30
LIDAR_CONFIG = dict(
    n_beams=N_BEAMS,
    max_range=12.0,          # world units (3 maze cells); beams clamp to this
    fov=2.0 * np.pi,         # full 360 degree scan
    step=0.1,                # ray-march resolution (world units)
    obs_layout='lidar',      # the planner observation is the LiDAR vector only
    use_cache=True,          # cache the LiDAR copy of the dataset to disk
)

## Single goal scan to start (K=1 == the original conditioning, no band).
## A K>1 goal band REQUIRES retraining and is gated on the Phase-7 decoy census
## (see BALL.md §13.3) -- do not raise this without that evidence.
GOAL_BAND_K = 1
GOAL_BAND_STEP = 4.0     # eval-only: band pose spacing (inert while K=1)

sm_horizon = 160
len_ovlap = 56
tot_horizon = sm_horizon
time_dim = 96

ovlp_o_dim = 256
ovlp_model_config = dict(
    c_traj_hzn=len_ovlap,
    in_dim=N_BEAMS,          # <-- was 2 (x, y); now the LiDAR dimension
    base_dim=32,
    dim_mults=(1, 2, 3, 4),
    time_dim=32,
    out_dim=ovlp_o_dim,
    tjti_enc_config=dict(t_seq_encoder_type='mlp',
                         cnn_out_dim=128,
                         final_mlp_dims=[1280, 512, ovlp_o_dim],
                         f_conv_ks=3,)
)


base = {
    'dataset': "pointmaze-giant-stitch-v0",

    'diffusion': {
        'config_fn': '',

        'sm_horizon': sm_horizon,
        'tot_horizon': tot_horizon,

        ## cnn model
        'model': 'models.cd_stgl_sml_dfu.stgl_sml_diffusion_v1.Unet1D_TjTi_Stgl_Cond_V1',
        'base_dim': 128,
        'dim_mults': (1, 2, 4, 8),
        'time_dim': time_dim,
        'network_config': dict(t_seq_encoder_type='mlp',
                               cat_t_w=True,
                               resblock_ksize=5,
                               st_ovlp_model_config=ovlp_model_config,
                               end_ovlp_model_config=ovlp_model_config,
                               ext_cond_dim=2 * ovlp_o_dim,
                               energy_mode=False,
                               time_mlp_config=3,
                               inpaint_token_dim=32,
                               inpaint_token_type='const',
                               ),

        ## sm dfu model
        'dfu_model': 'models.cd_stgl_sml_dfu.stgl_sml_diffusion_v1.Stgl_Sml_GauDiffusion_InvDyn_V1',
        'n_diffusion_steps': 512,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'diff_config': dict(
                            infer_deno_type='same',
                            obs_manual_loss_weights={},
                            w_loss_type='all',
                            is_direct_train=True,
                            len_ovlp_cd=len_ovlap,
                            tr_1side_drop_prob=0.20,
                            tr_inpat_prob=0.5,
                            tr_ovlp_prob=0.5,
                            tr_no_ovlp_none=False,
                            goal_band_k=GOAL_BAND_K,   ## K=1: single goal scan
                            ),

        ## LiDAR-aware trainer (renders LiDAR heatmaps instead of xy maze plots)
        'trainer_cls': 'ogb_task.ogb_maze_v1.OgB_Stgl_Sml_Lidar_Trainer_v1',
        'trainer_dict': dict(),

        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset (LiDAR loader + lidar_config switch)
        'loader': 'datasets.ogb_dset.OgB_Lidar_SeqDataset_V1',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 300,
        'dataset_config': dict(
            obs_select_dim=(0, 1),   # ignored when lidar_config is set (kept for API)
            dset_type='ogb',
            pad_option_2='buf',
            pad_type='first',
            extra_pad=64,
            ## --- LiDAR transform ---
            lidar_config=LIDAR_CONFIG,
            ## --- goal band width (K=1: bit-exact single-goal-scan conditioning) ---
            goal_band_k=GOAL_BAND_K,
        ),

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2_inv_v3',
        'n_train_steps': 2e6,

        'batch_size': 128,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 4000,
        'sample_freq': 8000,
        'n_saves': 5,

        'n_reference': 40,
        'n_samples': 10,

        'device': 'cuda',
    },

    'plan': {
        'config_fn': '',

        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': tot_horizon,
        'n_diffusion_steps': 512,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## goal band (inert at K=1; kept so the rollout knobs stay uniform)
        'goal_band_k': GOAL_BAND_K,
        'goal_band_step': GOAL_BAND_STEP,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}
