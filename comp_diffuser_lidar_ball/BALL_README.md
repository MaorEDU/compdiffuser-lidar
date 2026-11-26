# comp_diffuser_ball — True-LiDAR navigation on the PointMaze ("ball") model

Standalone, runnable implementation of True-LiDAR navigation for the PointMaze
ball: the diffusion planner sees only 30-beam LiDAR scans, never (x, y). The tree
started as a copy of an earlier LiDAR fork of CompDiffuser, with the ball port
applied in place.

Built 2026-07-12. The OGBench datasets are not bundled (see `./setup_data.sh` and
`conda_env/ogb_dset_download.ipynb`); the LiDAR scan caches and the localization
grids are in `data/ogb_maze/`.

---

## Final pipeline and result

The arm behind the headline number: **45.2 ± 0.8 %** success at OGBench's 1000-step
budget (500 episodes: 5 tasks × 20 problems × 5 seeds; 77.6 ± 1.1 % at 2500 steps).
The paper's CompDiffuser agent, which observes (x, y), scores 68 ± 3 %.

1. **Sensor**: 30-beam scan, FOV 2π, `max_range` 12 wu, yaw ≡ 0, raycast against the maze.
2. **Planner**: CompDiffuser in 30-D scan space; 8 segments × 160 frames with overlap 56
   make an 888-frame scan plan; start and goal scans are inpainted; DDIM-50, `cond_w` 2.0,
   T = 512.
3. **Decode**: a Viterbi localizer on a grid of 4409 poses (xy 0.5 wu, `n_theta` 1) turns
   the scan plan into (x, y) targets (max-product over paths).
4. **Pursuit**: BFS waypoints plus a cursor (argmin over a 61-index forward window),
   aiming 2 wu of plan arc ahead with a line-of-sight check.
5. **Executor**: a PID with Kp = 5 (the upstream P-law), Kd = Ki = 0.

A belief filter localizes the ball from its live scan at every step, and the same
likelihood machinery localizes the goal scan. No (x, y) reaches the planner; the decoder
and the localizer use the scans plus a reference grid built from the maze map.

- Planner checkpoint: the range-12 run
  `logs/pointmaze-giant-stitch-v0/diffusion/og_pntM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512_resume_T512/state_1600000.pt`
  (stored training step 1,992,000), kept as parts; rebuild it with `bash ../reassemble_checkpoints.sh`.
- Run settings: the `BASE:` line at the top of `plan_lidar_ball_bench100.sbatch`, with the
  planner loaded from that run.
- Caveat: steps 3 and 4 (the map-based decode and pursuit) are not part of the official
  CompDiffuser code, and the composed scan plans themselves are not jump-free.

## What is already verified (ran in this build, no GPU needed)

| Check | Result |
| --- | --- |
| **G1 geometry** (`ball_geometry_check.py --no_dataset`, all 3 mazes) | PASS — every free-cell ray matches the exact grid walk within march quantization (maze_unit=4/offset=4 hold for pointmaze) |
| **PID selftest** (`pid_controller.py`) | PASS — Kp=5 == upstream P-law; PI closes a steady-state offset; measured: on this first-order plant a D-term does NOT damp estimate noise, position EMA (`pos_lpf≈0.7`) cuts settled jitter ~2.4× (use it on the L3 belief tier) |
| **Ball reference grid** (giant, xy 0.5 wu, n_theta=1) | builds in seconds → 4409 poses, cached at `data/ogb_maze/lidar_grid/pointmaze-giant-stitch-v0_xy0.5_th1_mr12.npz` |
| **Decode round-trip** (865-frame cross-maze scan plan → grid decode) | median err 0.22 wu (5% of a cell), 94% same-cell, 0.37 s/plan — Arm-A targets are solid |
| **Likelihood census** (giant; 100 eval goals + 86 cell centers) | 26% unimodal, argmax==true 66%, **true goal in top-6: 99.5%** (ant was ~45%) — `logs/lidar_eval/ball_census_giant/census.json` |
| G1 dataset half / velocity reconstruction | needs the pointmaze npz — run on the cluster (below) |

The census number is the headline: the ball's check-and-move-on ceiling is
~99.5% vs the ant's ~45%. The wrong-region failure class that capped the ant
at 10% should nearly vanish once verify-and-hop is on.

## What changed vs. the ant fork (new / modified files)

**New**
```
config/ogb_pnt_maze/og_pntM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py   planner (giant)
config/ogb_pnt_maze/og_pntM_Me_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py   planner (medium smoke)
config/ogb_invdyn/og_inv_pnt/og_pntM_Gi_lidar30_xyv_g30d_invdyn_h12.py        Arm B (optional)
config/ogb_invdyn/og_inv_pnt/og_pntM_Gi_lidar30_xyv_g30d_invdyn_h16_chunk8.py Arm B ablation
diffuser/datasets/lidar/pid_controller.py                    PID2D + selftest
diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_lidar_pnt_planner_v1.py   ball rollout
diffuser/pl_eval/lidar_eval/ball_geometry_check.py           gate G1
diffuser/pl_eval/lidar_eval/ball_likelihood_census.py        Phase 7 (peaks/decoys/top-k)
diffuser/pl_eval/lidar_eval/ball_calibration.py              Phase 6.3 (failure vs uncertainty)
diffuser/pl_eval/lidar_eval/ball_traj_consistency.py         Phase 8 (lidar vs xy routes)
train_lidar_ball_planner.sbatch / train_lidar_ball_invdyn.sbatch
plan_lidar_ball_bench100.sbatch                              LADDER=L1|L2|L3|L4 presets
setup_data.sh
```

**Modified (guarded — ant/antmaze paths bit-exact)**
```
diffuser/datasets/ogb_dset/ogb_lidar_utils.py     +'xyv_lidar' layout, finite_diff_velocities
diffuser/datasets/ogb_dset/ogb_utils.py           terminals threaded into the lidar transform
diffuser/datasets/ogb_dset/ogb_lidar_dataset_v1.py  invdyn loader accepts 'xyv_lidar'
diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_planner_v1.py  +'pid_lidar' controller branch (hook)
diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py        ball subclass selection + ball-lidar arg branch
```

## Architecture in one paragraph

The diffusion planner sees **only 30-beam world-frame scans** (the ball has no
yaw — `has_quat` auto-detects from the 2-D obs). At rollout, plans are decoded
once per replan to (x, y) targets with the n_theta=1 grid localizer
(sequence-consistent filter), and a **PID law tracks them** (Arm A; Kp=5 ==
upstream's proven P-controller — no learned executor, hence no ant-style
goal-blindness). `LIDAR_LOC_SOURCE=env|belief` switches the PID's position
between the true position (L1/L2) and the live histogram-filter estimate (L3 —
the full True-LiDAR loop). Goal localization, plan re-rank, arrival detection,
verify-and-hop and the BFS grid navigator (L4) are inherited from the ant
stack unchanged; the route synthesizer is wrapped to render yaw=0 scans.
An optional learned inverse dynamics (Arm B, `xyv_lidar` = [x,y,vx,vy,scan])
is wired behind `LIDAR_INV_MODEL_PATH`, gated by the G3 sensitivity probe.
A zigzag/deadlock detector flags oscillation, dumps an artifact pack
(`LIDAR_BALL_LOG_NPZ=1`), and the policy is to pause and
inspect it — it never fakes success.

## Run order (gates in bold)

```bash
# 0. install (identical to upstream: conda_env/, then ogbench_cpdfu_release)
./setup_data.sh                       # dirs + optional cache links

# 1. G1 — full dataset check (cluster, after downloading the dataset)
python -m diffuser.pl_eval.lidar_eval.ball_geometry_check \
    --env pointmaze-giant-stitch-v0 --out logs/lidar_eval/ball_g1.json

# 1b. census for medium/large (giant already in logs/lidar_eval/ball_census_giant)
python -m diffuser.pl_eval.lidar_eval.ball_likelihood_census \
    --env pointmaze-medium-stitch-v0 --probs data/ogb_maze/ev_probs/ogb_pointM_Me_ev_prob_numEp20_eSdSt0.hdf5 \
    --out logs/lidar_eval/ball_census_medium

# 2. Phase 0 — reproduce the upstream (x,y) baseline (paper: 100/100/~68)
#    with the released o2d checkpoints + config/ogb_pnt_maze/og_pntM_Gi_o2d_*.py
sh diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.sh 0 20 0

# 3. train the LiDAR ball planner (medium smoke first, then giant)
BALL_CONFIG=config/ogb_pnt_maze/og_pntM_Me_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py sbatch train_lidar_ball_planner.sbatch
sbatch train_lidar_ball_planner.sbatch          # giant, 2M steps

# 4. **G2** at each checkpoint — offline decode BEFORE any rollout GPU time
#    (sample plans, decode with the grid, ask: do endpoints land in the goal cell?)
#    tooling: lidar_planner_recon.py pattern + ball_traj_consistency.py decode

# 5. eval ladder (each rung isolates one error source; 20 probs x 3 seeds)
LADDER=L1 sbatch plan_lidar_ball_bench100.sbatch
LADDER=L2 sbatch plan_lidar_ball_bench100.sbatch
LADDER=L3 sbatch plan_lidar_ball_bench100.sbatch
LADDER=L4 sbatch plan_lidar_ball_bench100.sbatch
# check-and-move-on on top of L3/L4:  LIDAR_VERIFY_HOP=1 LADDER=L3 sbatch ...

# 6. analyses
python -m diffuser.pl_eval.lidar_eval.ball_calibration \
    --jsons 'logs/pointmaze-giant-stitch-v0/plans/**/00_rollout*.json' \
    --census logs/lidar_eval/ball_census_giant/census.json
# phase 8 needs a plans npz -- generate it first (GPU): EPOCH=... sbatch dump_ball_plans.sbatch
# (that job also runs G2 + the consistency step; manual equivalent:)
python -m diffuser.pl_eval.lidar_eval.ball_traj_consistency --env pointmaze-giant-stitch-v0 \
    --plans_lidar logs/lidar_eval/ball_plans_lidar_1200000.npz

# 7. optional Arm B (learned inv-dyn) — G3 probe BEFORE rollouts
sbatch train_lidar_ball_invdyn.sbatch
python -m diffuser.pl_eval.lidar_eval.lidar_invdyn_goal_sensitivity --inv_model_path \
    logs/pointmaze-giant-stitch-v0/diffusion/og_pntM_Gi_lidar30_xyv_g30d_invdyn_h12 --inv_epoch latest --k 6
# then: LIDAR_INV_MODEL_PATH=<that dir> LADDER=L1 sbatch plan_lidar_ball_bench100.sbatch
```

## Key knobs (full list in plan_ogb_stgl_sml.py, ball branch)

`LIDAR_LOC_SOURCE` env|belief · `LIDAR_BALL_KP/KD/KI/KI_ZONE/POS_LPF` ·
`LIDAR_BALL_LOOKAHEAD` (4) · `LIDAR_BALL_DECODE_STRIDE` (2) ·
`LIDAR_LOC_XYSTEP` (0.5) / `LIDAR_LOC_NTHETA` (1) · `LIDAR_REPLAN` (1) /
`LIDAR_N_MAX_STEPS` (1000) · `LIDAR_ARRIVAL` / `LIDAR_VERIFY_HOP` ·
`LIDAR_NAV_MODE` (''|bfs|subgoal|tf_exec) · `LIDAR_BALL_DEADLOCK_WIN/DISP/UMIN`
· `LIDAR_BALL_LOG_NPZ` · `LIDAR_INV_MODEL_PATH` (Arm B).

## Known gaps / notes
* Pointmaze stitch episode length is asserted at load (G1 §3); configs assume
  ≤141 frames (the upstream o2d config's own constraint).
* `n_act_per_waypnt=1` + plan spacing ~0.13 wu/frame vs ball max speed 0.2
  wu/step: the ball can always keep up; raise `LIDAR_BALL_LOOKAHEAD` if it
  shears corners.
