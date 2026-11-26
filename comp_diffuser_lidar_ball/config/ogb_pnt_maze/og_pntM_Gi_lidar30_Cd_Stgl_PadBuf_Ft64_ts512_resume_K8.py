"""K=8 goal-band FINE-TUNE twin of og_pntM_Gi_lidar30_..._ts512_resume.py.

Why (evidence gate from the base config satisfied, 2026-07-19):
  * The base config said: do not raise goal_band_k without the Phase-7 census.
    Census ran 2026-07-18 (26% of goal scans unimodal, top-6 contains truth
    99.5%) and the K=1 planner's G2 is chance-level and SATURATED:
      - G2a 0% jump-free / G2b 22-24% / end err ~35.5 wu at BOTH 1.2M and 1.6M
        (flat => not undertraining);
      - cond_w 2/4/8 and full-T sampling sweeps all flat (2026-07-19) => the
        learned conditioning channel carries ~no goal->route information;
      - start err 0.37 wu vs end err 35.55 wu: the single inpainted goal frame
        is satisfied by a terminal teleport (the policy docstring's own
        prediction), it never shapes the plan body.
  * Mechanism of the fix: an 8-frame consecutive inpainted goal band makes the
    teleport escape expensive -- the generated frame H-9 gets loss pressure to
    kinematically connect to the GIVEN frame H-8, and the band is a real
    approach funnel from the data window (dataset cond: last K frames).
  * ONE variable changed vs the 1.6M run: GOAL_BAND_K 1 -> 8 (stride stays 1
    everywhere: dataset banding, loss masking, and policy inference all default
    S=1 and agree; a strided band (S=4) is the NEXT increment and needs a small
    policy patch -- do not stack it into this run).

Usage (cluster; fine-tunes FROM the 1.6M ckpt into a NEW run dir
`..._ts512_resume_K8_*`; the run-family assert truncates at the first
'_resume' so RESUME_PATH inside the _resume_T512 dir passes):

  RESUME_PATH=logs/pointmaze-giant-stitch-v0/diffusion/og_pntM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512_resume_T512/state_1600000.pt \
  BALL_CONFIG=config/ogb_pnt_maze/og_pntM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512_resume_K8.py \
  sbatch train_lidar_ball_planner.sbatch

(If a state_2000000.pt turns up, point RESUME_PATH at it instead.)
load4resume restores model+ema+step; optimizer moments restart (harmless,
survived once already). n_train_steps=2.4e6 => up to 800K fine-tune steps from
1.6M; kill early once G2 passes. Gate the ckpts OFFLINE, no ladder GPU:
  EPOCH=latest PLANNER_LOGDIR=$(find logs/pointmaze-giant-stitch-v0/diffusion -maxdepth 3 -name model_config.pkl | xargs -n1 dirname | grep K8) \
  sbatch dump_ball_plans.sbatch
Success criterion (revised gate, session log 2026-07-19): G2b >= 0.8 with
first-jump p10 >= ~120 (teleports handled by replan cadence + P1 pursuit).
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
    obs_layout='lidar',      # the planner observation is the LiDAR vector only
    use_cache=True,          # cache the LiDAR copy of the dataset to disk
)

## THE one change vs the 1.6M run (evidence in the module docstring):
## 8-frame consecutive goal band, stride 1 (train/loss/inference all agree at S=1).
GOAL_BAND_K = 8
GOAL_BAND_STEP = 4.0     # eval-only: band pose spacing (unused at S=1 banding)

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
                            goal_band_k=GOAL_BAND_K,   ## K=8: goal-band fine-tune
                            ),

        ## LiDAR-aware trainer (renders LiDAR heatmaps instead of xy maze plots)
        'trainer_cls': 'ogb_task.ogb_maze_v1.OgB_Stgl_Sml_Lidar_Trainer_v1',
        'trainer_dict': dict(do_train_resume=True, path_resume=_RESUME_PATH),

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
            ## --- goal band width (K=8: last-8-frame approach band, stride 1) ---
            goal_band_k=GOAL_BAND_K,
        ),

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2_inv_v3',
        'n_train_steps': 2.4e6,   ## up to 800K fine-tune steps from the 1.6M resume point

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

        ## goal band (K=8; rollout/dump must pass the (1+K, n_probs, D) st_gl format)
        'goal_band_k': GOAL_BAND_K,
        'goal_band_step': GOAL_BAND_STEP,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}
