"""Read the 2026-09-04 capture / re-entry / ada_ncomp arms -- MECHANISM BEFORE SCORE.

Prints, in the order the house rules require them to be read:

  1. the MECHANISM GATES, per arm:
       ball_reentry_hops > 0                      -- the arm actually ran
       carrot_dist_p50 <= the control's 2.96 wu   -- if it ROSE, this became FIX 5
                                                     (cursor running away from the ball,
                                                     progress a phantom) and the arm is
                                                     VOID whatever it scored
       drift = final_goal_dist - min_goal_dist    -- +4.9 wu in the control; capture is
                                                     supposed to drive this to ~0
       capture_in_range vs capture_blocked        -- separates "capture never had a
                                                     chance" (LOS census: the goal is
                                                     visible from only ~19% of positions
                                                     at 8-12 wu) from "the guard refused"
  2. the deliverables' own mechanism (replans/ep, history on and its length)
  3. the score, per block as FRACTION CLOSED -- the five blocks are 62.8 / 63.3 / 59.3 /
     37.8 / 9.2 wu, and block 4's raw distance has fooled this project once already
  4. the near-miss ladder, as the mechanism record
  5. paired tests against the control

Usage:
    python -m diffuser.pl_eval.lidar_eval.read_capture_bench
"""
import argparse
import glob
import json
import os

import numpy as np

GAP = np.array([62.8, 63.3, 59.3, 37.8, 9.2])      # |start - goal| per ev_probs block
## The carrot bar is taken from the CONTROL ARM IN THIS TABLE, not hardcoded: 2.96 was
## the MEAN over episodes and the per-episode MEDIAN is 1.95, and comparing a median
## against a mean is how a gate quietly stops gating. Falls back to the measured 1.95
## when the control arm is absent. srt200/srt888 sat at 6.25/6.50, void on either.
CTRL_CARROT_FALLBACK = 1.95                         # the benchmark runs, median of per-ep p50
CTRL_DRIFT = 4.90                                   # the benchmark runs


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='logs/**/plans/**/00_rollout.json')
    p.add_argument('--min_nep', type=int, default=100,
                   help='ignore short runs: N_EP=20 is ev_probs BLOCK 0')
    return p.parse_args()


def _arm(d):
    def g(k, dv=0):
        v = d.get(k, dv)
        return dv if v is None else v
    cap, re_, nc = g('ball_capture_wu'), g('ball_goal_reentry_wu'), g('ball_ada_ncomp')
    car, leash = g('ball_max_carrot_wu'), g('ball_cursor_leash_wu')
    srt, pull, eps = g('ball_los_shortcut_win'), g('ball_pull_win'), g('ball_loc_eps_mix')
    bits = []
    if re_: bits.append('RE')
    if cap and not re_: bits.append('CAPonly')
    if nc: bits.append('NC')
    if car: bits.append(f'car{car:g}')
    if leash: bits.append(f'leash{leash:g}')
    if srt: bits.append(f'srt{srt:g}')
    if pull: bits.append(f'pull{pull:g}')
    if eps: bits.append(f'eps{eps:g}')
    cm = d.get('ball_capture_margin')
    if re_ and cm is not None and abs(float(cm) - 0.6) > 1e-9:
        bits.append(f'm{float(cm):g}')
    ## THE ARM IS NOT JUST THE NEW KNOBS. `hist`, `repl_every` and `stall_steps` define
    ## which 2x2 cell this is, and pooling across them is exactly the arm-mixing this
    ## report exists to prevent -- the first version of this function silently merged
    ## the ada_dist C/H arms into "CONTROL" and produced a 15-run control whose replan
    ## steps were [1..10] and whose hist was both True and False at once.
    cell = (f"h{int(bool(d.get('ball_hist_on')))}"
            f"/re{int(g('repl_every_x') or (d.get('repl_ada_dist_cfg') or {}).get('repl_every') or 0)}"
            f"/st{int(g('ball_stall_steps', -1))}")
    return (('+'.join(bits) if bits else 'base') + ' ' + cell)


def _fisher(a, n1, b, n2):
    """Two-sided Fisher exact on [[a, n1-a], [b, n2-b]]; scipy if present."""
    try:
        from scipy import stats
        return float(stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1])
    except Exception:
        return float('nan')


def main():
    A = _parse()
    runs = {}
    for f in sorted(glob.glob(A.runs, recursive=True)):
        d = json.load(open(f))
        if d.get('num_ep', 0) < A.min_nep:
            continue
        if not d.get('ball_ep_summaries'):
            continue
        runs.setdefault(_arm(d), []).append((os.path.getmtime(f), d))
    if not runs:
        raise SystemExit('no runs matched')

    CTRL = 'base h1/re100/st5'          # the benchmark runs, the RH arm with every knob off
    order = sorted(runs, key=lambda k: (k != CTRL, k))

    def _carrot(v):
        c = np.array([e.get('carrot_dist_p50', np.nan)
                      for _, d in v for e in d['ball_ep_summaries']], dtype=float)
        return float(np.nanmedian(c)) if np.isfinite(c).any() else float('nan')

    ctrl_carrot = _carrot(runs[CTRL]) if CTRL in runs else CTRL_CARROT_FALLBACK
    if not np.isfinite(ctrl_carrot):
        ctrl_carrot = CTRL_CARROT_FALLBACK

    def summ(v):
        return [e for _, d in v for e in d['ball_ep_summaries']]

    def col(v, k, default=np.nan):
        return np.array([e.get(k, default) for e in summ(v)], dtype=float)

    print('=' * 100)
    print('1. MECHANISM GATES  (an arm whose mechanism did not move did not run)')
    print('=' * 100)
    print(f"{'arm':>18} {'runs':>4} {'reentryHops':>12} {'carrotP50':>10} {'drift':>7} "
          f"{'capEp%':>7} {'inRange':>8} {'blocked':>8} {'nearMin':>8}  GATE")
    for k in order:
        v = runs[k]
        hops = sum(int(d.get('ball_reentry_hops') or 0) for _, d in v)
        carrot = _carrot(v)
        mg = np.concatenate([d['ep_min_goal_dist'] for _, d in v])
        fg = np.concatenate([d['ep_final_goal_dist'] for _, d in v])
        drift = float(np.median(fg - mg))
        capep = 100.0 * sum(int(d.get('ball_capture_episodes') or 0)
                            for _, d in v) / (100 * len(v))
        inr = float(np.nansum(col(v, 'capture_in_range_steps', 0)))
        blk = float(np.nansum(col(v, 'capture_blocked_steps', 0)))
        nm = col(v, 'capture_near_min', np.inf)
        nm = float(np.median(nm[np.isfinite(nm)])) if np.isfinite(nm).any() else np.nan
        bad = []
        if k != CTRL and 'RE' in k and hops == 0:
            bad.append('reentry NEVER FIRED')
        if k != CTRL and np.isfinite(carrot) and carrot > ctrl_carrot + 0.5:
            bad.append(f'carrot ROSE past {ctrl_carrot:.2f} -> FIX 5, arm VOID')
        verdict = 'ok' if not bad else '** ' + '; '.join(bad)
        print(f"{k:>18} {len(v):>4} {hops:>12} {carrot:>10.2f} {drift:>7.2f} "
              f"{capep:>7.1f} {inr:>8.0f} {blk:>8.0f} {nm:>8.2f}  {verdict}")
    print(f"\n  control reference: carrot_dist_p50 {ctrl_carrot:.2f} (+0.5 wu tolerance), drift +{CTRL_DRIFT} wu")
    print("  blocked == inRange everywhere means capture had NO OPPORTUNITY (the LOS")
    print("  census: goal visible from ~19% of positions at 8-12 wu, 6% on block 4),")
    print("  not that the guard is mis-tuned. Only re-entry is acting in that case.")

    print('\n' + '=' * 100)
    print('2. THE DELIVERABLES still fire')
    print('=' * 100)
    for k in order:
        v = runs[k]
        rp = np.mean([np.mean(d['ep_cnt_repls']) for _, d in v])
        steps = sorted({s for _, d in v for r in d['ep_repl_steps'] for s in r})
        allst = np.concatenate([np.array(r) for _, d in v
                                for r in d['ep_repl_steps'] if len(r)])
        h = {(bool(d['ball_hist_on']), int(d['ball_hist_len'])) for _, d in v}
        print(f"{k:>18}  repl/ep {rp:4.1f}  after200 {np.mean(allst > 200):.2f}  "
              f"hist {h}  steps {steps[:10]}")

    print('\n' + '=' * 100)
    print('3. SCORE  (per block as FRACTION CLOSED -- never raw distance)')
    print('=' * 100)
    print(f"{'arm':>18} {'succ':>9} {'frozen':>7} {'locErr':>7} " +
          ' '.join(f'b{i}%' for i in range(5)))
    res = {}
    for k in order:
        v = runs[k]
        suc = np.concatenate([np.array(d['ep_is_suc'], float) for _, d in v])
        mg = np.concatenate([d['ep_min_goal_dist'] for _, d in v])
        res[k] = (suc, mg)
        per = []
        for b in range(5):
            m = np.concatenate([np.array(d['ep_min_goal_dist'])[b * 20:(b + 1) * 20]
                                for _, d in v])
            per.append(100 * (1 - np.median(m) / GAP[b]))
        print(f"{k:>18} {suc.sum():4.0f}/{len(suc):<4d} "
              f"{np.nanmean(col(v,'frozen_frac')):>7.3f} "
              f"{np.nanmedian(col(v,'loc_err_p50')):>7.2f} " +
              ' '.join(f'{p:5.1f}' for p in per))

    print('\n' + '=' * 100)
    print('4. NEAR-MISS LADDER  (mechanism record; the p50 hides all of this)')
    print('=' * 100)
    ths = [1, 2, 3, 4, 6, 8, 12, 16, 20]
    print(f"{'arm':>18} " + ' '.join(f'<{t}wu' for t in ths))
    for k in order:
        mg = res[k][1]
        print(f"{k:>18} " + ' '.join(f'{int((mg < t).sum()):4d}' for t in ths))

    if CTRL in res:
        print('\n' + '=' * 100)
        print('5. vs the CONTROL in this table')
        print('=' * 100)
        cs, cm = res[CTRL]
        for k in order:
            if k == CTRL:
                continue
            s, m = res[k]
            p1 = _fisher(int(s.sum()), len(s), int(cs.sum()), len(cs))
            p2 = _fisher(int((m < 8).sum()), len(m), int((cm < 8).sum()), len(cm))
            print(f"{k:>18}  succ {s.sum():.0f}/{len(s)} vs {cs.sum():.0f}/{len(cs)} "
                  f"p={p1:.3f}   <8wu {int((m<8).sum())} vs {int((cm<8).sum())} p={p2:.3f}")
    else:
        print(f'\n  No {CTRL!r} arm in this table -- compare against the benchmark runs (3/300, '
              'carrot_dist_p50 2.96, drift +4.9) by hand, and say so when reporting.')


if __name__ == '__main__':
    main()
