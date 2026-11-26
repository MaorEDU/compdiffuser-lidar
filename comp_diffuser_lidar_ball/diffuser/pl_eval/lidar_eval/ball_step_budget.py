#!/usr/bin/env python3
"""Where does the STEP BUDGET go? Per-step accounting from rollout timelines.

Pure numpy over on-disk `00_rollout.json` + `ball_ep<N>_timeline.npz` + the cached
LiDAR grid. No GPU, no MuJoCo, no torch -> runs on the login node (BALL_SESSIONS_LOG
§1: npz + LiDAR grid is login-safe; instantiating a dataset config is NOT, because
`MazeEnv.__init__` renders).

    python diffuser/pl_eval/lidar_eval/ball_step_budget.py \
        --jobs <job-ids>

WHY THIS EXISTS (2026-08-01). OGBench registers pointmaze-giant at
`max_episode_steps=1000`; this project always ran 2500, so the deliverable at the
benchmark's own horizon is 45.2, not 77.6. At 1000 steps the binding constraint is
TIME, and time is not being spent on locomotion: **~51% of the first 1000 control
steps produce zero ball displacement**, while a moving ball always moves at the
physical maximum (step sizes are quantised at 0 / 0.200 / 0.283 wu -- there is no
"slow", only "stopped"). Deleting the frozen steps takes the SAME plans from 45.2
to 74.2 @1000. This tool splits that frozen time into the two mechanisms so each
can be costed before any GPU spend:

  A-idle    frozen with the carrot ON the ball (`|target - pos| < --near_wu`).
            The cursor is `argmin` over a forward window, so it advances only when
            the BALL moves; when the carrot collapses onto the cursor point the
            command is ~0, the ball cannot move, and the cursor cannot advance.
            Nothing breaks that fixed point except the stall timer.
  B-pinned  frozen with the carrot far away (p50 27 wu) and the PID saturated --
            the escape marched the monotone cursor out of reach and the ball is
            being driven into a wall. Only a replan resets the cursor.
  timer     the frozen steps a shorter `--stall_steps` would have skipped, i.e.
            the first min(run, stall_steps) of every frozen run. A subset overlay
            on A/B, reported separately, never double-counted in the totals.

STANDING TRAPS this tool exists to avoid:
  * `plan_frac` / `cursor_max` / `n_stall_escapes` are CURSOR statistics. The
    physical quantity is displacement of `true_xy`, which is what is counted here.
  * Success LATCHES and OGBench's own metric is "reached the goal at any step"
    (`locomaze/maze.py`: `info['success']=1.0` when `|xy-goal| <= _goal_tol`,
    `terminate_at_goal=True`), so arrival TIME is the whole story, not final
    distance. Station-keeping after arrival buys nothing.
  * N_EP=20 is ONE task (ev_probs block 0). Everything here reads per block.
  * The counterfactuals are UPPER BOUNDS: they delete frozen time from a trajectory
    that is otherwise assumed unchanged. They size a fix, they do not predict one.
"""
import os, sys, glob, json, argparse, importlib.util
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))


def _load(mod_name, rel_path):
    """Import our numpy-only lib modules without dragging in the torch package.

    Same helper as `ball_decode_control.py`: `import diffuser.datasets...` pulls
    `diffuser/utils/rendering.py` -> `d4rl`, which is not installed here and is
    not needed for anything this tool does.
    """
    try:
        return __import__(mod_name, fromlist=['*'])
    except Exception:  # noqa: BLE001
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, rel_path))
        spec = importlib.util.spec_from_file_location(os.path.basename(rel_path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


LG = _load('diffuser.datasets.lidar.lidar_grid', '../../datasets/lidar/lidar_grid.py')
GC = _load('diffuser.pl_eval.lidar_eval.lidar_grid_common', 'lidar_grid_common.py')

ROOT_DEFAULT = ('logs/pointmaze-giant-stitch-v0/plans/'
                'release_og_pntM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512_H160_T512_'
                'LimitsNormalizer_b1_condFalse/0')
BUDGETS = (500, 1000, 1500, 2000, 2500)


def rundir(root, job):
    m = sorted(glob.glob(os.path.join(root, f'*j{job}-*')))
    assert len(m) == 1, f'job {job}: expected 1 run dir, got {m}'
    return m[0]


def load_grid(env, max_range, xy_step, n_theta):
    """The cached grid, for its `scanner` only (maze_map + world_to_cell + scan).

    `max_range` is NOT load-bearing for the wall geometry used here (LOS and the
    occupancy test read `maze_map`), but the cache path is keyed by it, so the
    resolved path is printed as the audit line §1 demands.
    """
    return LG.load_or_build_grid(env, dict(max_range=max_range), xy_step=xy_step,
                                 n_theta=n_theta, verbose=True)


def frozen_runs(mask):
    """[(start, length), ...] for each contiguous True run in `mask`."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def analyse_ep(z, a):
    """Per-step budget accounting for one episode, up to arrival (or timeout)."""
    xy, pos, tgt = z['true_xy'], z['pos_used'], z['target']
    g, cur, u = z['true_goal_xy'], z['cursor'], z['u_norm']

    gd = np.linalg.norm(xy - g, axis=1)
    hit = np.flatnonzero(gd <= a.goal_tol)
    t_arr = int(hit[0]) if len(hit) else -1
    end = t_arr if t_arr > 0 else len(xy) - 1

    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)[:end]
    frozen = step < a.eps
    ## the controller closes on `pos_used` (the belief), so the carrot geometry
    ## must be measured against the position the executor actually saw.
    td = np.linalg.norm(tgt[:end] - pos[:end], axis=1)
    near = td < a.near_wu

    A = frozen & near
    B = frozen & ~near
    timer = np.zeros_like(frozen)
    for s, L in frozen_runs(frozen):
        timer[s:s + min(L, a.stall_steps)] = True

    ## `_b` = restricted to the first `--budget` steps, which is the pool the
    ## headline number is actually spending. The unsuffixed counts run to
    ## arrival/timeout and are what the counterfactuals subtract from.
    w = slice(0, a.budget)
    return dict(
        t_arr=t_arr, end=end, n_steps=len(xy),
        moving=int((~frozen).sum()), a_idle=int(A.sum()), b_pinned=int(B.sum()),
        timer=int(timer.sum()), n_frozen_runs=len(frozen_runs(frozen)),
        moving_b=int((~frozen[w]).sum()), a_idle_b=int(A[w].sum()),
        b_pinned_b=int(B[w].sum()), timer_b=int(timer[w].sum()),
        arc=float(step.sum()),
        speed_moving=float(step[~frozen].mean()) if (~frozen).any() else 0.0,
        td_b_p50=float(np.median(td[B])) if B.any() else 0.0,
        u_a_p50=float(np.median(u[:end][A])) if A.any() else 0.0,
        cursor_max=int(cur[:end].max()) if end else 0,
        n_plans=int(len(z['plans_step'])),
    ), (A, B, frozen)


def diagnose(z, a, sel, scanner, rng, cap):
    """The two questions Phase 1 owes: A's trigger and B's recoverability.

    Sampled (the LOS predicate is O(len(window)) raycasts per step), and only over
    steps in `sel`, using the executor's own `los_clear` -- not a re-derivation.
    """
    pos, tgt, cur = z['pos_used'], z['target'], z['cursor']
    pxy, ptr, pst = z['plans_xy'], z['plans_ptr'], z['plans_step']
    idx = np.flatnonzero(sel)
    if not len(idx):
        return {}
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    out = dict(n=0, blind_ahead=0, behind_in_reach=0, behind_any_in_reach=0,
               slide_free=0)
    for k in idx:
        p = max(int(np.searchsorted(pst, k, side='right')) - 1, 0)
        plan = pxy[ptr[p]:ptr[p + 1]]
        c = int(cur[k])
        if c + 1 >= len(plan):
            continue
        out['n'] += 1

        ## --- A: is there anything VISIBLE in the next `lookahead_wu` of arc? ---
        seg = np.linalg.norm(np.diff(plan[c:], axis=0), axis=1)
        adv = int(np.searchsorted(np.cumsum(seg), a.lookahead_wu) + 1)
        win = plan[c + 1:min(c + 1 + adv, len(plan))]
        if len(win):
            ok = GC.los_clear(scanner, pos[k], win, margin=a.los_margin)
            out['blind_ahead'] += int(not ok.any())

        ## --- B: could a BACKWARD re-acquire have found a point in reach? ---
        ## Two variants, because they license different fixes: the shipped
        ## `_ball_cursor_reacquire` only looks back `--back` indices, whereas
        ## searching the WHOLE plan is a different (cheap) change. If the two
        ## differ a lot the window is the defect; if both are low the cursor is
        ## genuinely orphaned and only a replan or a slide can help.
        d_all = np.linalg.norm(plan[:c + 1] - pos[k], axis=1)
        out['behind_in_reach'] += int((d_all[max(0, c - a.back):] <= a.leash_wu).any())
        out['behind_any_in_reach'] += int((d_all <= a.leash_wu).any())

        ## --- B: does the live SCAN offer a free direction that is goal-ward? ---
        ## The scan-only wall slide is only buildable if, while pinned, some beam
        ## heading is both open (>= `slide_wu`) and positively projected on the
        ## direction to the carrot. No map is consulted here beyond the raycast
        ## the agent's own sensor performs.
        r = np.asarray(scanner.scan(float(pos[k][0]), float(pos[k][1]), 0.0),
                       dtype=np.float64).reshape(-1)
        th = np.asarray(scanner.beam_offsets, dtype=np.float64)  # yaw = 0
        d_c = tgt[k] - pos[k]
        nrm = float(np.linalg.norm(d_c))
        if nrm > 1e-9:
            d_c = d_c / nrm
            proj = np.cos(th) * d_c[0] + np.sin(th) * d_c[1]
            out['slide_free'] += int(((r >= a.slide_wu) & (proj > 0)).any())
    return out


def report(name, runs, a, scanner):
    rng = np.random.default_rng(0)
    rows, diagA, diagB, rates = [], [], [], []
    print(f'\n===== {name} =====')
    for job, d in runs:
        j = json.load(open(os.path.join(d, '00_rollout.json')))
        suc = np.asarray(j['ep_is_suc'], dtype=bool)
        rates.append(float(j['ep_srate']))
        dm = (j['ball_decode_mode'] if 'ball_decode_mode' in j
              else 'filter(key absent: run predates the knob)')
        print(f'  j{job}: srate={j["ep_srate"]:5.1f}  decode_mode={dm}  '
              f'max_steps={j.get("max_episode_steps")}  '
              f'stall_steps={j.get("ball_stall_steps")}  los={j.get("ball_los")}  '
              f'leash={j.get("ball_cursor_leash_wu")}  '
              f'max_carrot={j.get("ball_max_carrot_wu")}  '
              f'lookahead_wu={j.get("ball_lookahead_wu")}')
        for ep in range(len(suc)):
            p = os.path.join(d, f'ball_ep{ep}_timeline.npz')
            if not os.path.exists(p):
                raise SystemExit(
                    f'{p} missing -- the run was made without LIDAR_BALL_LOG_NPZ=1, '
                    f'so no per-step timeline exists and nothing here is computable.')
            z = np.load(p, allow_pickle=True)
            r, (A, B, _) = analyse_ep(z, a)
            r.update(ep=ep, blk=ep // 20, job=job, suc=bool(suc[ep]))
            ## GATE: the reconstruction must agree with what the env recorded,
            ## or every number below is void.
            assert (r['t_arr'] >= 0) == bool(suc[ep]), (
                f'j{job} ep{ep}: reconstructed arrival {r["t_arr"]} contradicts '
                f'recorded ep_is_suc={bool(suc[ep])}')
            rows.append(r)
            if a.diag_cap:
                diagA.append(diagnose(z, a, A, scanner, rng, a.diag_cap))
                diagB.append(diagnose(z, a, B, scanner, rng, a.diag_cap))
    rt = np.asarray(rates)
    print(f'  ---> recorded srate {rt.mean():.1f} +- '
          f'{rt.std(ddof=1) if len(rt) > 1 else 0:.1f} over {len(rt)} seed(s) '
          f'[reconstruction agrees on {len(rows)}/{len(rows)} episodes]')
    return rows, diagA, diagB


def budget_table(rows, a):
    t = np.asarray([r['t_arr'] for r in rows], dtype=float)
    A = np.asarray([r['a_idle'] for r in rows], dtype=float)
    B = np.asarray([r['b_pinned'] for r in rows], dtype=float)
    TM = np.asarray([r['timer'] for r in rows], dtype=float)
    blk = np.asarray([r['blk'] for r in rows])

    def rate(tt, bud):
        return 100.0 * np.mean((tt >= 0) & (tt < bud))

    def cf(sub):
        return np.where(t >= 0, t - sub, -1.0)

    print('\n--- budget curve (reconstructed) ---')
    print('  ' + '  '.join(f'{b}:{rate(t, b):5.1f}' for b in BUDGETS))

    print(f'\n--- where the first {a.budget} steps go ---')
    mvb = np.asarray([r['moving_b'] for r in rows], dtype=float)
    Ab = np.asarray([r['a_idle_b'] for r in rows], dtype=float)
    Bb = np.asarray([r['b_pinned_b'] for r in rows], dtype=float)
    TMb = np.asarray([r['timer_b'] for r in rows], dtype=float)
    pool = float((mvb + Ab + Bb).sum())
    for nm, v in (('moving', mvb), ('A-idle (carrot on the ball)', Ab),
                  ('B-pinned (carrot away, PID saturated)', Bb),
                  ('  [overlay: stall-timer latency]', TMb)):
        print(f'  {nm:40s} {v.sum():9.0f} steps  {100 * v.sum() / pool:5.1f}%  '
              f'median/ep {np.median(v):6.0f}')
    print(f'  {"POOL (steps lived inside the budget)":40s} {pool:9.0f}')

    print(f'\n--- counterfactual @{a.budget} (UPPER BOUNDS) ---')
    print(f'  {"baseline":38s} {rate(t, a.budget):5.1f}')
    for nm, sub in (('- stall-timer latency', TM), ('- A-idle', A),
                    ('- B-pinned', B), ('- A and B (ceiling)', A + B)):
        print(f'  {nm:38s} {rate(cf(sub), a.budget):5.1f}')

    print(f'\n--- per block @{a.budget} ---')
    print('  blk   base  -A    -B    -A-B   @2500')
    for b in range(int(blk.max()) + 1):
        m = blk == b
        print(f'   {b}   {rate(t[m], a.budget):5.1f} {rate(cf(A)[m], a.budget):5.1f} '
              f'{rate(cf(B)[m], a.budget):5.1f} {rate(cf(A + B)[m], a.budget):5.1f}  '
              f'{rate(t[m], 2500):5.1f}')

    print('\n--- profile by outcome class ---')
    cls = [('fast (arrived <= budget)', (t >= 0) & (t < a.budget)),
           ('slow (arrived later)', (t >= a.budget)),
           ('never arrived', t < 0)]
    for nm, m in cls:
        if not m.any():
            continue
        f = lambda k: np.median([rows[i][k] for i in np.flatnonzero(m)])
        n_fr = np.median([(rows[i]['a_idle'] + rows[i]['b_pinned'])
                          / max(rows[i]['end'], 1) for i in np.flatnonzero(m)])
        print(f'  {nm:26s} n={int(m.sum()):4d}  t_med={f("t_arr"):7.0f}  '
              f'frozen={100 * n_fr:4.1f}%  speed_moving={f("speed_moving"):.3f}  '
              f'n_plans={f("n_plans"):.0f}  frozen_runs={f("n_frozen_runs"):.0f}')


def diag_table(diagA, diagB, a):
    def agg(ds, keys):
        n = sum(d.get('n', 0) for d in ds)
        return n, {k: sum(d.get(k, 0) for d in ds) for k in keys}

    nA, dA = agg(diagA, ('blind_ahead',))
    nB, dB = agg(diagB, ('behind_in_reach', 'behind_any_in_reach', 'slide_free'))
    print(f'\n--- mechanism diagnostics (sampled, <= {a.diag_cap} steps/episode) ---')
    if nA:
        print(f'  A-idle steps sampled: {nA}')
        print(f'    nothing LOS-visible in the next {a.lookahead_wu} wu of plan arc '
              f'(margin {a.los_margin}): {100 * dA["blind_ahead"] / nA:5.1f}%')
        print(f'    => the rest ({100 * (1 - dA["blind_ahead"] / nA):.1f}%) had a '
              f'reachable point ahead and idled anyway: pursuit geometry, not visibility')
    if nB:
        print(f'  B-pinned steps sampled: {nB}')
        print(f'    a plan point within {a.leash_wu} wu of the ball, searching back '
              f'{a.back} indices: {100 * dB["behind_in_reach"] / nB:5.1f}%  '
              f'(the SHIPPED _ball_cursor_reacquire window)')
        print(f'    ... searching the WHOLE plan behind the cursor:            '
              f'{100 * dB["behind_any_in_reach"] / nB:5.1f}%')
        print(f'    live scan offers a free ({a.slide_wu} wu) goal-ward heading: '
              f'{100 * dB["slide_free"] / nB:5.1f}%  (scan-only wall slide is buildable)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True, help='comma-separated slurm job ids')
    ap.add_argument('--root', default=ROOT_DEFAULT)
    ap.add_argument('--env', default='pointmaze-giant-stitch-v0')
    ap.add_argument('--budget', type=int, default=1000,
                    help="OGBench's own max_episode_steps for pointmaze-giant")
    ap.add_argument('--goal_tol', type=float, default=1.0,
                    help='locomaze _goal_tol for a point agent')
    ap.add_argument('--eps', type=float, default=0.02,
                    help='|dxy| below this is "frozen" (a moving ball does 0.2/0.283)')
    ap.add_argument('--near_wu', type=float, default=1.0,
                    help='carrot within this of the agent => A-idle, else B-pinned')
    ap.add_argument('--stall_steps', type=int, default=50,
                    help='the executor\'s LIDAR_BALL_STALL_STEPS, for the timer overlay')
    ap.add_argument('--lookahead_wu', type=float, default=2.0)
    ap.add_argument('--los_margin', type=float, default=0.6)
    ap.add_argument('--leash_wu', type=float, default=3.0)
    ap.add_argument('--back', type=int, default=15, help='backward re-acquire window')
    ap.add_argument('--slide_wu', type=float, default=1.0)
    ap.add_argument('--diag_cap', type=int, default=40,
                    help='max sampled steps per episode per family (0 = skip diagnostics)')
    ap.add_argument('--max_range', type=float, default=12.0)
    ap.add_argument('--xy_step', type=float, default=0.5)
    ap.add_argument('--n_theta', type=int, default=1)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    scanner = load_grid(a.env, a.max_range, a.xy_step, a.n_theta).scanner if a.diag_cap \
        else None
    jobs = [s.strip() for s in a.jobs.split(',') if s.strip()]
    rows, dA, dB = report('ARM', [(x, rundir(a.root, x)) for x in jobs], a, scanner)
    budget_table(rows, a)
    if a.diag_cap:
        diag_table(dA, dB, a)
    if a.out:
        json.dump(rows, open(a.out, 'w'))
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
