#!/usr/bin/env python3
"""Failure decomposition + paired arm comparison from rollout timelines.

Pure numpy over on-disk `00_rollout.json` + `ball_ep<N>_timeline.npz`. No GPU, no
MuJoCo, no torch -> runs on the login node (see BALL_SESSIONS_LOG §1: anything that
is npz + LiDAR grid is login-safe; instantiating a dataset config is NOT).

    # decompose one arm (job ids, comma separated)
    python diffuser/pl_eval/lidar_eval/ball_failure_decomp.py --jobs <job-ids>

    # paired comparison of a new arm against the baseline, per episode
    python diffuser/pl_eval/lidar_eval/ball_failure_decomp.py \
        --jobs <new ids> --baseline <job-ids>

WHY THE BUCKETS ARE DEFINED THIS WAY (BALL_SESSIONS_LOG latest+18):
  planner  - no plan the episode sampled ever routes within `--tol` of the goal.
             A continuity fix cannot touch these (82/215 of the 57.0% baseline).
  executor - some plan does route to the goal, but the ball never got within 3 wu.
             78% of failures detach persistently from the plan; 1% of successes do.
  lastmile - the ball got inside 3 wu and never inside `--tol`.

STANDING TRAPS this tool exists to avoid:
  * `plan_frac` / `cursor_max` are CURSOR statistics -- they say nothing about where
    the BALL is. Read `cursor_dist`/`ball_min_gd`, which is what this reports.
  * `ep_min_goal_dist` rewards wandering; `net/arc` is circular.
  * N_EP=20 is ONE task (ev_probs block 0). Always read per block (`ep // 20`).
  * Seed-to-seed spread is large: report mean +- sd over >=3 seeds or a PAIRED test.
"""
import os, sys, glob, json, argparse
import numpy as np
from collections import Counter

ROOT_DEFAULT = ('logs/pointmaze-giant-stitch-v0/plans/'
                'release_og_pntM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512_H160_T512_'
                'LimitsNormalizer_b1_condFalse/0')
DETACH_WU = 3.0
JUMP_WU = 2.0


def rundir(root, job):
    m = sorted(glob.glob(os.path.join(root, f'*j{job}-*')))
    assert len(m) == 1, f'job {job}: expected 1 run dir, got {m}'
    return m[0]


def analyse_run(d, tol, detach_wu=DETACH_WU, jump_wu=JUMP_WU):
    j = json.load(open(os.path.join(d, '00_rollout.json')))
    suc = np.asarray(j['ep_is_suc'], dtype=bool)
    rows = []
    for ep in range(len(suc)):
        p_npz = os.path.join(d, f'ball_ep{ep}_timeline.npz')
        if not os.path.exists(p_npz):
            raise SystemExit(
                f'{p_npz} missing -- the run was made without LIDAR_BALL_LOG_NPZ=1, '
                f'so no per-step timeline exists and nothing here is computable.')
        z = np.load(p_npz, allow_pickle=True)
        xy, cur, cd = z['true_xy'], z['cursor'], z['cursor_dist']
        pxy, ptr, pst = z['plans_xy'], z['plans_ptr'], z['plans_step']
        g = z['true_goal_xy']
        n = len(xy)

        plan_gd, jumps_ct = [], []
        for p in range(len(pst)):
            q = pxy[ptr[p]:ptr[p + 1]]
            plan_gd.append(float(np.linalg.norm(q - g, axis=1).min()))
            st = np.linalg.norm(np.diff(q, axis=0), axis=1)
            jumps_ct.append(int((st > jump_wu).sum()))

        # persistent detachment: first step after which the ball stays off the plan
        det, above = -1, cd > detach_wu
        if above.any():
            for t in np.flatnonzero(above):
                if above[t:].mean() > 0.9:
                    det = int(t)
                    break

        ball_gd = np.linalg.norm(xy - g, axis=1)
        best_gd = float(min(plan_gd))
        ok = bool(suc[ep])
        bmin = float(ball_gd.min())
        bucket = ('success' if ok else
                  'lastmile' if bmin <= detach_wu else
                  'planner' if best_gd > tol else 'executor')
        rows.append(dict(ep=ep, blk=ep // 20, suc=ok, bucket=bucket, det=det,
                         n_steps=n, best_plan_gd=best_gd, ball_min_gd=bmin,
                         ball_final_gd=float(ball_gd[-1]),
                         jumps_plan0=jumps_ct[0], n_plans=len(pst),
                         cursor_dist_p50=float(np.median(cd))))
    return j, rows


def mcnemar(a, b):
    """Exact two-sided McNemar over paired boolean outcomes."""
    from math import comb
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
    return n10, n01, min(1.0, p)


def report(name, runs, tol):
    allrows, rates = [], []
    print(f'\n===== {name} =====')
    for job, d in runs:
        j, rows = analyse_run(d, tol)
        for r in rows:
            r['job'] = job
        allrows += rows
        rates.append(j['ep_srate'])
        ## §3 mistake 13: never let a missing json key masquerade as a recorded
        ## value. Runs predating the knob genuinely have no field, so say so
        ## rather than silently printing the default.
        dm = (j['ball_decode_mode'] if 'ball_decode_mode' in j
              else 'filter(key absent: run predates the knob)')
        gates = (f"decode_mode={dm} "
                 f"wp_calls={j.get('ball_wp_calls')} "
                 f"wp_scan_seeds={j.get('ball_wp_scan_seeds')} "
                 f"wp_fallbacks={j.get('ball_wp_fallbacks')}")
        ds = j.get('ball_decode_stats') or []
        fj = np.median([x['frac_cell_jumps'] for x in ds]) if ds else float('nan')
        print(f'  j{job}: srate={j["ep_srate"]:5.1f}  frac_cell_jumps_p50={fj:.4f}  {gates}')
    r = np.asarray(rates, dtype=float)
    print(f'  ---> {name} = {r.mean():.1f} +- {r.std(ddof=1) if len(r) > 1 else 0:.1f} '
          f'over {len(r)} seed(s)')

    n = len(allrows)
    F = [x for x in allrows if not x['suc']]
    print(f'  N={n}  success={n - len(F)} ({100 * (n - len(F)) / n:.1f}%)  failures={len(F)}')
    c = Counter(x['bucket'] for x in F)
    for k, v in c.most_common():
        print(f'    {k:9s} {v:4d}  {100 * v / len(F):5.1f}% of failures')
    det = [x for x in F if x['det'] >= 0]
    dsuc = [x for x in allrows if x['suc'] and x['det'] >= 0]
    print(f'    persistently detached: {len(det)}/{len(F)} of failures '
          f'({100 * len(det) / max(len(F), 1):.0f}%) vs '
          f'{len(dsuc)}/{n - len(F)} of successes '
          f'({100 * len(dsuc) / max(n - len(F), 1):.0f}%)')
    print('    per block  suc/ n | planner | executor')
    for b in range(5):
        G = [x for x in allrows if x['blk'] == b]
        if not G:
            continue
        print(f'      blk{b}: {sum(x["suc"] for x in G):3d}/{len(G):3d} | '
              f'{sum(x["bucket"] == "planner" for x in G):3d} | '
              f'{sum(x["bucket"] == "executor" for x in G):3d}')
    return allrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True, help='comma-separated slurm job ids')
    ap.add_argument('--baseline', default='', help='job ids to pair against')
    ap.add_argument('--root', default=ROOT_DEFAULT)
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    jobs = [s.strip() for s in a.jobs.split(',') if s.strip()]
    runs = [(x, rundir(a.root, x)) for x in jobs]
    new = report('ARM', runs, a.tol)

    if a.baseline:
        bj = [s.strip() for s in a.baseline.split(',') if s.strip()]
        base = report('BASELINE', [(x, rundir(a.root, x)) for x in bj], a.tol)
        assert len(base) == len(new), (
            f'paired test needs equal episode counts: {len(new)} vs {len(base)}')
        A = np.asarray([x['suc'] for x in new], bool)
        B = np.asarray([x['suc'] for x in base], bool)
        n10, n01, p = mcnemar(A, B)
        print(f'\n===== PAIRED (n={len(A)}) =====')
        print(f'  arm {A.sum()} vs baseline {B.sum()}   '
              f'discordant {n10} arm-only / {n01} baseline-only   McNemar p={p:.3g}')
        for b in range(5):
            m = np.asarray([x['blk'] == b for x in new])
            if not m.any():
                continue
            n10b, n01b, pb = mcnemar(A[m], B[m])
            print(f'    blk{b}: {A[m].sum():3d} vs {B[m].sum():3d}  '
                  f'({n10b} / {n01b})  p={pb:.3g}')

    if a.out:
        json.dump(new, open(a.out, 'w'))
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
