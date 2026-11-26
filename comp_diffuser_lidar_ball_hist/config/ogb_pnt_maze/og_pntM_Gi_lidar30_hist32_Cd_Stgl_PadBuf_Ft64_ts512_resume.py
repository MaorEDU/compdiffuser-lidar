"""RESUME twin of og_pntM_Gi_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512.py.

the preemptible partition has a 72 h wall clock and preempts regardless, and a 2M-step run does
not fit; the previous fork's equivalent training was killed at ~1.45M. So the resume
path exists from day one rather than being written under pressure at 1.2M.

`load4resume` restores model + EMA + step; the optimizer moments restart, which is
harmless at this point in training. The run writes to its OWN directory (the config
filename is part of exp_name), so the original run's checkpoints are never overwritten.

Usage (cluster):
  RESUME_PATH=<run_dir>/state_<step>.pt \
  BALL_CONFIG=config/ogb_pnt_maze/og_pntM_Gi_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512_resume.py \
  sbatch train_ball_planner.sbatch

RESUME_PATH must point into the original run dir (train_ogb_stgl_sml.py asserts the
config family matches). Re-submit the same command whenever preemption kills it,
pointing RESUME_PATH at the newest state_*.pt.
"""
import os as _os
_RESUME_PATH = _os.environ.get('RESUME_PATH', '')
assert _RESUME_PATH, 'set RESUME_PATH=<run_dir>/state_<step>.pt to use the _resume config'

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
    obs_layout='lidar',      # the observation IS the scan -- no x, y, ever
    use_cache=True,          # cache the LiDAR copy of the dataset to disk
)

## ---- observation history ----
## 32 frames == 32 env steps (pointmaze runs n_act_per_waypnt=1) ~= 6.4 wu ~= 1.6 maze
## cells at the ball's top speed: "a few dozen steps backwards". It must be <= extra_pad
## below, because that left pad is what gives an episode-boundary window something to
## look back on.
HIST_LEN = 32
HIST_DROP_PROB = 0.25

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

## same encoder family as the two overlap encoders, over a HIST_LEN window. Unlike
## them the input is CLEAN (it is observed, not a latent), so it is never q_sampled
## and the encoder is always called at t = 0.
hist_o_dim = 256
hist_model_config = dict(
    c_traj_hzn=HIST_LEN,
    in_dim=N_BEAMS,
    base_dim=32,
    dim_mults=(1, 2, 3, 4),
    time_dim=32,
    out_dim=hist_o_dim,
    tjti_enc_config=dict(t_seq_encoder_type='mlp',
                         cnn_out_dim=128,
                         final_mlp_dims=[1280, 512, hist_o_dim],
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
                               ## the history branch (None => bit-identical to release)
                               hist_model_config=hist_model_config,
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
                            ## history dropout: required (segments 1..N-1 have none),
                            ## and it is what makes this checkpoint serve the control arm
                            tr_hist_drop_prob=HIST_DROP_PROB,
                            ),

        ## LiDAR-aware trainer (renders (time x beam) heatmaps instead of maze plots)
        'trainer_cls': 'ogb_task.ogb_maze_v1.OgB_Stgl_Sml_Lidar_Trainer_v1',
        'trainer_dict': dict(do_train_resume=True, path_resume=_RESUME_PATH),

        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset (LiDAR loader + history window)
        'loader': 'datasets.ogb_dset.OgB_Lidar_SeqDataset_V1',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 300,
        'dataset_config': dict(
            obs_select_dim=(0, 1),   # placeholder: ignored once lidar_config is set
            dset_type='ogb',
            pad_option_2='buf',
            pad_type='first',
            extra_pad=64,
            ## --- LiDAR transform ---
            lidar_config=LIDAR_CONFIG,
            ## --- observation history window (must be <= extra_pad) ---
            hist_len=HIST_LEN,
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
        'prefix': 'plans/ballv2',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}
