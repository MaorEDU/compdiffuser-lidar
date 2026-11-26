"""
comp_diffuser_ball_v2 -- the one training config.

CompDiffuser planner for ``pointmaze-giant-stitch-v0`` (the BALL model) trained on
**LiDAR** observations instead of the (x, y) position, plus an **observation-history**
conditioning branch.

Diff vs. the paper's ``og_pntM_Gi_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py`` -- and this list is
exhaustive, everything else is byte-for-byte the release:

  1. ``ovlp_model_config['in_dim']``       2 -> N_BEAMS         (the observation swap)
  2. ``loader``                            -> OgB_Lidar_SeqDataset_V1
  3. ``trainer_cls``                       -> OgB_Stgl_Sml_Lidar_Trainer_v1
  4. ``dataset_config['lidar_config']``    -> turns the LiDAR transform on
  5. ``network_config['hist_model_config']`` + ``dataset_config['hist_len']``
                                           -> the history branch
  6. ``diff_config['tr_hist_drop_prob']``  -> the history dropout

The diffusion U-Net adapts automatically: ``transition_dim`` is read from
``dataset.observation_dim``, which is now N_BEAMS.

ONE checkpoint serves BOTH arms of the experiment. `tr_hist_drop_prob` is not an
optional regulariser -- composed segments 1..N-1 never have a history and neither do
the first HIST_LEN steps of an episode -- and it is also what makes the trained model
runnable with the history dropped, which is the no-history control (`LIDAR_BALL_HIST=0`).
The control and the treatment therefore share weights, and the comparison is paired.

THIS FILE IS THE SEPARATE LABELLED SENSING ARM required by BALLV2.md section 7.

Section 7 declined `max_range` 12 -> 24 and `n_beams` 30 -> 32 as pre-registered
non-choices, to be revisited "only with evidence, and only as a separate labelled arm".
The evidence arrived (2026-08-23 .. 2026-08-25) and is recorded in
DECODER_2026-08-24.md and GATE_RESULTS_2026-08-23.md:

  * `goal_decode_census`, all 100 problems, model not involved: goal localizable within
    one cell **85.0% at max_range 12 -> 97.0% at 24**; err p90 22.34 -> 0.93 wu;
    single-scan aliasing 42.6% -> 23.5%.
  * The rollout probe's mis-aim tail is a HARD FLOOR no decoder touches: `head_err_vs_true`
    >= 4 wu sat at ~30% under the forward filter and at **29.8% under argmin**,
    while the same decoder swap moved the OFFLINE endpoint 48.5% -> 85.0%. Two decoders,
    one number unmoved -- that is sensing, not decoding.
  * `hist_len` 32 admits only **8.6%** fully-real training windows (giant-stitch episodes
    are 201 steps, `sm_horizon=160`, `extra_pad=64`, so a real history needs start >= 96).
    16 gives **23.8%**. G-hist failed on the 2M-step model because the history channel was
    learned as a frame-0 leak, which is a data-shape ceiling a retrain must lift.

DELIBERATELY NOT CHANGED: `n_beams` stays 30. Two sensing changes at once would make the
result uninterpretable, and the beam-alignment lever is worth ~2.4 pts against
max_range's ~19. It remains declined and available.

COMPARABILITY: every number produced before this file used max_range 12 and hist_len 32.
None of them carry across -- this is a different sensor. The 12/32 checkpoint stays the
baseline and its config is untouched; `config_fn` is part of `exp_name`, so this run
writes to its own logs dir and cannot overwrite it. The LiDAR dataset cache is keyed on
the full scanner config (`_cache_path` md5 + `r{max_range}` in the filename), so the
r12 cache is not silently reused -- expect a one-time ~85 s/1M-transition rebuild.
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
    max_range=24.0,          # world units (6 maze cells); giant's sightline ceiling
                             # 12.0 in the baseline arm -- see the docstring
    fov=2.0 * np.pi,         # full 360 degree scan
    step=0.1,                # ray-march resolution (world units)
    obs_layout='lidar',      # the observation IS the scan -- no x, y, ever
    use_cache=True,          # cache the LiDAR copy of the dataset to disk
)

## ---- observation history ----
## 16 frames == 16 env steps (pointmaze runs n_act_per_waypnt=1) ~= 3.2 wu ~= 0.8 maze
## cells at the ball's top speed. It must be <= extra_pad below, because that left pad is
## what gives an episode-boundary window something to look back on.
##
## 32 in the baseline arm. Lowered because the WINDOW SUPPLY, not the lookback length, was
## the binding constraint: at 32 only 8.6% of training windows carry a fully-real history
## and 63.7% are 32 identical copies of frame 0, so the model learned the channel as a
## frame-0 leak (G-hist FAIL, p=4e-188 on a degenerate feature). 16 -> 23.8% fully-real.
HIST_LEN = 16
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
        'trainer_dict': dict(),

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
