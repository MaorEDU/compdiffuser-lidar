# comp_diffuser_ball_v2 — clean LiDAR L3 + receding-horizon replanning with scan history

**Env:** `pointmaze-giant-stitch-v0` · **Base:** a clean fork of `comp_diffuser_release`
(the CompDiffuser paper release) · **Status:** trained and benchmarked; best arm 13 / 12 / 9 % (3 seeds × 100 problems).

The agent navigates giant PointMaze from 30 range beams alone. No `(x, y)` reaches the
planner at any point, in any form. On that base we run **one** experiment:

> **Receding-horizon planning with observation history** — replan every *X* control
> steps instead of on the paper's adaptive-distance trigger, and condition each replan
> on the last few dozen *observed* scans instead of only the current one.

These are one mechanism, not two. Today the planner sees a single start scan inpainted
at frame 0 — "the current abstraction". History is only meaningful if the planner is
re-invoked while the agent has a trajectory behind it; replanning is only worth its GPU
cost if each replan carries information the previous one lacked. Cadence is how the
experiment is parameterised; history is what makes the cadence matter.

## Why a new folder

`comp_diffuser_ball` accreted ~20 executor/decoder/planner modifications over six
weeks. Each defaults to *off*, but the file deciding the ball's next action is 1228
lines on top of a 1403-line ant planner, and half the project's post-mortems are about
one modification interacting with another. This repo is the same idea rebuilt from the
paper release with only the vanilla LiDAR stack, so that what the experiment changes is
the only thing that differs.

---

## 1. One training run, not two

The history branch trains with **25% history dropout**. That is not an optional
regulariser — composed segments 1…N−1 never have a history, and neither do the first 32
steps of an episode. It is also what makes the trained model runnable with the history
dropped, which *is* the no-history control. One checkpoint, four arms, paired
comparison, no seed or training-length variance between control and treatment.

| Arm | History | Replan | What it is |
| --- | --- | --- | --- |
| **C** | dropped | `ada_dist` (paper) | the vanilla L3 baseline |
| **R** | dropped | every X ∈ {50,100,200} | cadence alone |
| **H** | last 32 scans | `ada_dist` | history alone |
| **RH** | last 32 scans | every X | the experiment |

R and H are the off-diagonal cells. They cost eval time only, and without them an RH
win is uninterpretable.

---

## 2. The control loop

```
scan_t = scanner.scan(x, y, 0.0)                    # 30 beams, 2π, max_range 12, yaw ≡ 0
plan   = policy.gen_cond_stgl(start scan, goal scan, [history])   # 8 × 160, ovlp 56 → 888
xy     = GridLocalizer.localize_sequence(plan[::2]) # FORWARD FILTER, interpolated back
target = progress cursor → 2 wu of plan arc → LOS pull-back
u      = clip(5·(target − p̂), −1, 1)                # PID2D, kp=5, kd=ki=0, pos_lpf=0
```

**Ladder** (`LIDAR_LOC_SOURCE`):

| Rung | Position the PID closes on | Isolates |
| --- | --- | --- |
| L1 | `env.get_xy()` | planner + decode quality |
| **L3 — the deliverable** | `LiveLocalizer` MAP estimate from observed scans | + self-localization |

**There is no L2 here, deliberately.** The previous fork's L2 ("goal localized from its
scan") differed from L1 only through machinery this repo does not have — an arrival
detector, a BFS navigator, a plan re-ranker — so in this control loop nothing consumes a
decoded goal cell and an "L2" would be the *same rollout* as L1. (Consistently, the fork
measured them flat: 55.0 vs 55.0, p = 1.0, 55/100 episodes bit-identical.) Goal-scan
localizability is real but offline, so it is **logged** per episode
(`ball_ep_summaries[i].goal_decode`: decoded goal cell, error, rank of the true cell)
rather than dressed up as a rung.

### What is deliberately absent

Viterbi plan decode · BFS per-segment waypoints · goal-band K>1 · `goal`/`grid` plan
re-ranking · arrival detection + verify-and-hop · the BFS grid navigator · cursor leash
/ max-carrot / min-carrot · wall-slide · frozen-step escapes · sub-cell belief
interpolation · position low-pass · the learned inverse-dynamics arm and its v2
chunk/diff/act-CFG machinery.

Two of these are removed *structurally* rather than flagged off, so they cannot come
back by an unset environment variable: `GridLocalizer.localize_viterbi` and
`LiveLocalizer._subcell_xy` are deleted from the source.

> **Amended 2026-09-04 — two of these are now knobs, both defaulting to OFF.**
> `LIDAR_BALL_CAPTURE_WU` re-introduces a form of *arrival* — but only as **steering**,
> never as self-reporting: within R wu of the agent's **own decoded goal** and with clear
> line of sight, the PID aims at that instead of at a plan index. Success is still decided
> by the env, so it is not the fork's verify-and-hop rule. It exists because ~64 of 300
> episodes come within 3 cells of the goal, 3–6 convert, and the ball is measurably
> *driving past* — `cursor_max` p50 215 of 888, `final_goal_dist` 42 vs `min` 36.
> Like `LIDAR_BALL_ADA_NCOMP`, it consumes a decoded goal cell, so **"there is no L2 here"
> does not hold for a capture run** and must be stated wherever such a number is reported.
>
> Still structurally absent and not coming back: `GridLocalizer.localize_viterbi` and
> `LiveLocalizer._subcell_xy` are deleted from the source, and the connectivity-constrained
> decode is out of scope by decision (2026-09-04) — including any greedy variant, not only
> the beam DP.

### What is kept from the lessons learned

Progress pursuit (bounded forward-window argmin, not a time cursor) · LOS carrot
pull-back · the 50-step stall escape · `pos_lpf = 0` · **belief seeding before plan 0**.
The last one was a bug in the fork worth +15 points: the belief refreshed inside a step
that ran *after* the first plan, so plan 0 of 100% of L3 episodes went out blind. Here
it happens in `_ep_reset_hook`, which the base loop calls between `env.reset()` and plan
0, so the ordering is structural.

---

## 3. Where the code lives

| Piece | File |
| --- | --- |
| sensor, grid, localizer, PID | `diffuser/datasets/lidar/` |
| one definition of the history window, shared by train and rollout | `diffuser/datasets/lidar/hist_window.py` |
| LiDAR transform + dataset (returns the history slice) | `diffuser/datasets/ogb_dset/ogb_lidar_{utils,dataset_v1}.py` |
| the history encoder in the denoiser | `diffuser/models/cd_stgl_sml_dfu/stgl_sml_temporal_cond_v1.py` |
| history threaded through train + sampling | `stgl_sml_diffusion_v1.py`, `stgl_sml_policy_v1.py` |
| `periodic` replan mode + the four hooks | `diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_planner_v1.py` |
| the whole executor | `diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_lidar_ball_planner_v1.py` |
| gates + report | `diffuser/pl_eval/lidar_eval/` |
| the one training config | `config/ogb_pnt_maze/og_pntM_Gi_lidar30_hist32_*.py` |

### The history branch, concretely

The denoiser's global conditioning vector was
`cat([time, st_ovlp_feat, end_ovlp_feat, st_token, end_token])`, where the two
`*_ovlp_feat` embed the **noisy** overlaps with neighbouring composed segments. A third
`Traj_Time_Encoder` embeds the last 32 **observed** scans, and its output plus a
has-history token are concatenated on (`tot_cond_dim` 672 → 960).

Two differences from the overlap encoders, both deliberate:
* the history is **clean** — it is observed, not a latent — so it is never `q_sample`d
  and the encoder is always called at `t = 0`;
* it is optional. `hist_model_config=None` leaves the block inert and the network is
  then **bit-identical to the release** (asserted by `test_hist_parity.py`), and
  `hist_is_drop` zeroes it per sample, which is what the control arm rides on.

History conditions **segment 0 only**; segments 1…N−1 are sampled with it dropped,
which is the regime the dropout trains for. Frame-0 inpainting of the current scan is
kept — history is an additional channel, not a replacement — so the decoded plan still
starts at the agent and needs no index offset.

Training and rollout build the window through the same `left_pad_history`, so they
cannot drift apart: the last 32 frames, oldest first, with the oldest available frame
repeated at the front when there are not enough. At an episode's start that means 32
copies of the start scan, which is exactly the `pad_type='first'` distribution the
training windows use.

---

## 4. Running it

```bash
# smoke first, always (medium, 2e5 steps)
BALL_CONFIG=config/ogb_pnt_maze/og_pntM_Me_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512.py \
  sbatch train_ball_planner.sbatch

# the one giant training (2M steps)
sbatch train_ball_planner.sbatch

# resume after preemption (a preemptible partition WILL kill a 2M-step run)
RESUME_PATH=logs/pointmaze-giant-stitch-v0/diffusion/<run>/state_<step>.pt \
BALL_CONFIG=config/ogb_pnt_maze/og_pntM_Gi_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512_resume.py \
  sbatch train_ball_planner.sbatch

# gates, on every checkpoint, before any rollout
python -m diffuser.pl_eval.lidar_eval.plan_decode_gate  --config <cfg> --n_prob 20 --n_samp 8
python -m diffuser.pl_eval.lidar_eval.hist_holdout_gate --config <cfg> --n_win 256

# the 2x2
N_EP=100 SEEDS=0,1,2 LADDER=L3 HIST=0 REPL_EVERY=0   sbatch plan_ball_bench100.sbatch  # C
for X in 50 100 200; do
N_EP=100 SEEDS=0,1,2 LADDER=L3 HIST=0 REPL_EVERY=$X  sbatch plan_ball_bench100.sbatch  # R
done
N_EP=100 SEEDS=0,1,2 LADDER=L3 HIST=1 REPL_EVERY=0   sbatch plan_ball_bench100.sbatch  # H
N_EP=100 SEEDS=0,1,2 LADDER=L3 HIST=1 REPL_EVERY=<best> sbatch plan_ball_bench100.sbatch  # RH

python -m diffuser.pl_eval.lidar_eval.rollout_report --runs "logs/**/plans/**/*.json"
```

### Executor knobs (all default OFF = bit-identical to the shipped loop)

| knob | what it does | status |
| --- | --- | --- |
| `LIDAR_BALL_CAPTURE_WU` | goal capture: within R wu of the **decoded** goal, in line of sight, aim at it | **the live arm**, run at 12 |
| `LIDAR_BALL_ADA_NCOMP` | clip `n_comp` to the distance still to cover | positive but never isolated |
| `LIDAR_LOC_EPS_MIX` | uniform floor in the belief's predict step; kills the 14% latch | **null** on score (latch 11.7%→4.3%, succ 9/300 vs RE 11/300, p=0.63) |
| `LIDAR_BALL_CONF_REPL_THR` | FIX 11: replan when the cursor reaches the plan's first run of low argmin-margin frames (`_RUN`, `_MIN_GAP`, `_EXTRA` alongside) | implemented 2026-10-18, **not benched**; run at 0.3 on top of RE |
| `LIDAR_BALL_PULL_WIN` | plan-level string-pull at decode time | **never benched**; gated on `pull_plan_audit` |
| `LIDAR_BALL_MAX_CARROT_WU` | cap the carrot by straight-line distance | **REFUTED**, leave 0 |
| `LIDAR_BALL_CURSOR_LEASH_WU` | leash the blind cursor march | **null**, never fires, leave 0 |
| `LIDAR_BALL_LOS_SHORTCUT_WIN` | runtime cursor shortcut | **REFUTED**, leave 0 |

```bash
# the CPU gate that must pass before PULL_WIN is ever set
python -m diffuser.pl_eval.lidar_eval.pull_plan_audit \
    --npz logs/lidar_eval/r24_hist16/plans_dump_hist0.npz --max_range 24
```

`REPL_EVERY=N` means literally every N control steps: pointmaze runs
`n_act_per_waypnt = 1`, so one waypoint is one env step. The replan **budget** is sized
from the cadence automatically (`⌈n_max_steps / N⌉ + 2`) — a fixed `max_n_repl=15` would
have truncated the `REPL_EVERY=50` arm at step 750, i.e. run a different experiment for
the last quarter of every episode. That bug was caught by `test_ballv2_core.py` and the
test now guards it.

---

## 5. Gates (pre-registered, in order)

> **Outcome on the first 2M-step checkpoint (2026-08-23): G-hist and G2 both FAIL.** The planner
> was then retrained with range-24 scans and a 16-scan history (`og_pntM_Gi_lidar30_r24_hist16`),
> and that model is the one benchmarked.


| Gate | Test | Fail → |
| --- | --- | --- |
| **G0** | the paper's `o2d` giant planner reproduces ≈68 @1000 in this repo | fix the harness before trusting anything |
| **G1** | scanner geometry on pointmaze: dataset xy all in free cells, hand-placed poses give exact ray distances | fix maze constants; nothing downstream is valid |
| **G-parity** | `hist_model_config=None` reproduces the release denoiser exactly | the port introduced a silent change |
| **G2** | offline plan decode: ≥95% free-cell frames, ~0 jumps, ≥80% of endpoints within a cell of the goal | planner work, **no rollouts** |
| **G-hist** | held-out paired test: history lowers x0 error, p<0.01, concentrated in the early frames | cancel the H and RH rollouts |
| **G-mech** | replans/episode rises toward `1000/X`; frozen share moves | that arm did not run — discard its score |

G-parity passes today. G2 is computed with the **filter** decode, matching the arm it
gates; never compare G2 numbers across decoders. Record that the fork's composed `G2b`
numbers were measuring the decoder's sequence filter rather than goal anchoring.

---

## 6. House rules (inherited; each cost the previous project weeks)

- **`N_EP=100`, ≥3 seeds, read per block.** `N_EP=20` is ev_probs *block 0* — one
  start–goal pair jittered and repeated 20 times. Three weeks of the fork's success
  rates were 20 samples of one route. Seed-to-seed spread reached 10 points; report
  mean ± sd or a paired per-episode test, never a single run.
- **1000 steps.** OGBench registers `pointmaze-giant` at `max_episode_steps=1000` and
  success **latches**, so budget monotonically buys score. 2500 is an extended-budget
  ablation and must be labelled as one; it must never be compared to a paper number.
  `rollout_report.py` flags both of these automatically.
- **Read the mechanism gate before the score.** Pre-register what a change should move
  mechanically. The fork has a documented case where the mechanism fired exactly as
  designed and the score did not follow (p = 0.83) — without the gate that becomes an
  argument about ±2 points.
- **Decompose the instrument before convicting the model.** Twice in the fork the
  load-bearing number had a filter inside it; one CPU measurement would have saved
  eleven days each time.
- **Falsify on CPU before spending GPU.** A 2-second geometry census killed a 60-beam
  retrain. Counter-example to remember: an offline pre-screen is blind to belief↔control
  feedback (sub-cell interpolation predicted 0.37 wu of error and delivered 6.22).
- **Never write "beats CompDiffuser".** This pipeline uses a localizer grid built from
  the true occupancy map, and `los_clear` raycasts against it too. The paper's method
  has neither. We have strictly *more* task knowledge, delivered through a sensor. The
  honest comparison is stated alongside any external number.

## 7. Pre-registered non-choices

Two sensing levers are free while retraining anyway, and both are **declined**:

- `n_beams` 30 → 32. 30 beams straddle 90°/270°; alignment is worth ~2.4 pts of
  aliasing at equal or fewer beams.
- `max_range` 12 → 24. Aliasing 63.6% → 38.3%; 24 wu is giant's sightline ceiling.

Either would break comparability with every number the previous fork produced, and this
project is about replan cadence and history, not sensing. Revisit only with evidence,
and only as a separate labelled arm.
