"""
CompDiffuser BALL planner for ``pointmaze-giant-stitch-v0`` -- **max_range=24** arm.

ONE variable changed vs ``og_pntM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py``:
``lidar_config['max_range']`` 12.0 -> 24.0 (3 -> 6 maze cells). Everything else
(30 beams, K=1 goal scan, arch, sched, steps) is identical. Fresh from-scratch
train -- do NOT resume any mr12 checkpoint into this config (the scan
distribution changes completely; caches are keyed by max_range so nothing
stale is reused: dataset cache ``lidar_*_30beams_r24_*.npz``, grid cache
``*_mr24.npz`` both auto-build).

Why (beam/range aliasing census, 2026-07-20, pure-geometry, all 4409 grid poses;
tool: ``diffuser/pl_eval/lidar_eval/ball_beam_aliasing_census.py``, results:
``logs/lidar_eval/ball_beam_aliasing_census.json``):
  * confusable pose == its nearest look-alike >=8 wu away is CLOSER in
    RMS-per-beam scan space than the median legal 0.5 wu lattice step
    (i.e. the L2 training loss literally prefers a 2-cell teleport to motion).
  * mr12: 63.6% confusable at 30 beams -- and 60 beams (60.1%) / 120 beams
    (61.3%) DO NOT HELP: the aliasing is locally-congruent geometry within the
    12 wu horizon, invisible at any angular resolution. n_beams is a dead lever.
  * mr24: 38% confusable at every beam count; the median pose flips to
    "legal step cheaper than teleport" (alias/tau p50 0.88 -> 1.16).
  * mr36 == mr48 == mr24: no sightline in giant exceeds ~24-36 wu, so
    mr24 already captures everything the sensor can ever see. This config is
    the sensing ceiling, not a point on a curve.
Residual 38% confusability means sensing alone may fix continuity (G2a /
first-jump depth) but not necessarily anchoring (G2b) -- gate on the G2 trend
(dump ~every 200K; see BALL_SESSIONS_LOG.md section 4 for the kill criteria).
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
    max_range=24.0,          # <-- THE one change (was 12.0). 6 maze cells;
                             #     census: sightline ceiling of the giant maze.
    fov=2.0 * np.pi,         # full 360 degree scan
    step=0.1,                # ray-march resolution (world units)
    obs_layout='lidar',      # the planner observation is the LiDAR vector only
    use_cache=True,          # cache keyed by n_beams AND max_range -> no staleness
)

## K=1 single goal scan: same as the original recipe. The K=8 goal-band
## fine-tune was killed 2026-07-20 (G2b 22%->10% at +320K FT steps, at/below
## the ~10% chance floor) -- the band gets satisfied by a local terminal
## teleport and never creates goal->route coupling. Bundling K=8 into this
## from-scratch run is an UNTESTED second variable: team call, default no.
GOAL_BAND_K = 1
GOAL_BAND_STEP = 4.0     # eval-only: band pose spacing (inert while K=1)

sm_horizon = 160
len_ovlap = 56
tot_horizon = sm_horizon
time_dim = 96

ovlp_o_dim = 256
ovlp_model_config = dict(
    c_traj_hzn=len_ovlap,
    in_dim=N_BEAMS,          # unchanged: still 30 beams, so in_dim stays 30
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
