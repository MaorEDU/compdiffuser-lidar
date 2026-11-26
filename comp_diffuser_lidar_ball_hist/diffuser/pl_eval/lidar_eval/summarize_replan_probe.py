"""Summarize the short instrumented rollout probe (`rollout_probe.sbatch`).

Reads every `00_rollout.json` written with ``LIDAR_BALL_PROBE=1`` and answers the
three closed-loop questions the offline gates cannot:

  Q1  does the history tensor REACH the model? (`reached_model`)
  Q2  how degenerate is the deque per replan, and does it drift out of the training
      distribution as real frames accumulate? (`real_frac` vs replan step)
  Q3  is the forward filter's decode error a TAIL artefact -- i.e. concentrated in
      the region that replanning discards rather than the head the executor
      pursues? (`{head,mid,tail}_agree_1cell`)

The arm is parsed from the run directory name, which carries `-hist{0,1}-re{N}-st{K}`.

Usage:
    python -m diffuser.pl_eval.lidar_eval.summarize_replan_probe
    python -m diffuser.pl_eval.lidar_eval.summarize_replan_probe --root logs --since 3600
"""
import argparse, glob, json, os, re, sys, time
sys.path.append('./')

import numpy as np

ARM_RE = re.compile(r'-hist(?P<hist>\d+)-re(?P<re>\d+)')
ST_RE = re.compile(r'-st(?P<st>\d+)')


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='logs')
    p.add_argument('--since', type=float, default=0,
                   help='only runs modified in the last N seconds (0 = all)')
    p.add_argument('--out', default='logs/lidar_eval/replan_probe_summary.json')
    return p.parse_args()


def _load(root, since):
    runs = []
    for f in glob.glob(os.path.join(root, '*', 'plans', '*', '**', '00_rollout.json'),
                       recursive=True):
        if since and (time.time() - os.path.getmtime(f)) > since:
            continue
        try:
            j = json.load(open(f))
        except Exception:                                   # noqa: BLE001
            continue
        if 'ball_replan_probe' not in j:
            continue
        d = os.path.dirname(f)
        m, ms = ARM_RE.search(d), ST_RE.search(d)
        if not m:
            continue
        hist, rep = int(m.group('hist')), int(m.group('re'))
        runs.append(dict(
            path=f,
            arm={(0, 0): 'C', (0, 1): 'R', (1, 0): 'H', (1, 1): 'RH'}[(hist, int(rep > 0))],
            hist=hist, repl_every=rep,
            block=int(ms.group('st')) // 20 if ms else -1,
            probe=j['ball_replan_probe'],
            eps=j.get('ball_ep_summaries', []),
            repl_steps=j.get('ep_repl_steps', []),
            hist_on=bool(j.get('ball_hist_on', False)),
        ))
    return runs


def main():
    a = _parse()
    runs = _load(a.root, a.since)
    if not runs:
        print(f'no probe rollouts found under {a.root!r} '
              f'(need LIDAR_BALL_PROBE=1). nothing to summarize.')
        return 1
    print(f'loaded {len(runs)} probe runs, '
          f'{sum(len(r["probe"]) for r in runs)} replans total\n')

    out = {}
    for arm in ('C', 'R', 'H', 'RH'):
        rs = [r for r in runs if r['arm'] == arm]
        if not rs:
            continue
        pr = [p for r in rs for p in r['probe']]
        if not pr:
            continue
        ok = [p for p in pr if 'probe_error' not in p]
        hist_on = any(r['hist_on'] for r in rs)
        print(f"=== arm {arm}  (hist={rs[0]['hist']} repl_every={rs[0]['repl_every']}) "
              f"blocks={sorted({r['block'] for r in rs})}  replans={len(pr)} ===")
        if pr and 'probe_error' in pr[0]:
            print(f"   !! probe errors: {sum('probe_error' in p for p in pr)}/{len(pr)}"
                  f"  e.g. {pr[0]['probe_error']}")

        # -- Q1 -------------------------------------------------------------
        reached = [bool(p.get('reached_model', False)) for p in pr]
        shapes = {tuple(p['hist_shape']) for p in pr if 'hist_shape' in p}
        print(f"  Q1 hist reaches model : {100*np.mean(reached):5.1f}% of replans"
              f"   hist_on={hist_on}   shapes={sorted(shapes) if shapes else 'none'}")

        # -- Q2 -------------------------------------------------------------
        rf = np.array([p['real_frac'] for p in pr if 'real_frac' in p])
        st = np.array([p['step'] for p in pr if 'real_frac' in p])
        if len(rf):
            early, late = st < 32, st >= 32
            print(f"  Q2 deque real_frac    : all p50={np.percentile(rf,50):.2f}"
                  + (f"   step<32 p50={np.percentile(rf[early],50):.2f} (n={early.sum()})"
                     if early.any() else "   step<32 n=0")
                  + (f"   step>=32 p50={np.percentile(rf[late],50):.2f} (n={late.sum()})"
                     if late.any() else "   step>=32 n=0"))
            print(f"     (training: 63.7% of windows are real_frac=0.0, "
                  f"only 8.2% are real_frac~1.0)")

        # -- Q3 -------------------------------------------------------------
        row = []
        for b in ('head', 'mid', 'tail'):
            v = np.array([p[f'{b}_agree_1cell'] for p in ok if f'{b}_agree_1cell' in p])
            g = np.array([p[f'{b}_argmin_gap_p50'] for p in ok if f'{b}_argmin_gap_p50' in p])
            row.append((b, np.mean(v) if len(v) else np.nan,
                        np.percentile(g, 50) if len(g) else np.nan))
        print("  Q3 filter vs argmin   : " + "   ".join(
            f"{b} agree={100*v:5.1f}% gap_p50={g:5.2f}wu" for b, v, g in row))

        # -- operational ----------------------------------------------------
        he = np.array([p['head_err_vs_true'] for p in ok if 'head_err_vs_true' in p])
        ee = np.array([p['end_err_vs_goal'] for p in ok if 'end_err_vs_goal' in p])
        if len(he):
            print(f"  plan[0] vs true pos   : p50={np.percentile(he,50):5.2f}wu  "
                  f"p90={np.percentile(he,90):5.2f}wu")
        if len(ee):
            print(f"  plan[-1] vs true goal : p50={np.percentile(ee,50):5.2f}wu  "
                  f"p90={np.percentile(ee,90):5.2f}wu")

        eps = [e for r in rs for e in r['eps']]
        if eps:
            print(f"  mechanism             : frozen_frac p50="
                  f"{np.percentile([e['frozen_frac'] for e in eps],50):.3f}"
                  f"   loc_err_p50 p50={np.percentile([e['loc_err_p50'] for e in eps],50):.2f}wu"
                  f"   cursor_max p50={np.percentile([e['cursor_max'] for e in eps],50):.0f}")
        nrep = [len(s) for r in rs for s in r['repl_steps']]
        if nrep:
            print(f"  replans per episode   : p50={np.percentile(nrep,50):.0f} "
                  f"min={min(nrep)} max={max(nrep)}")
        print()

        ## PER BLOCK. The arm-level mean hides route-to-route spread, and on this
        ## task that spread is the whole story: arm C's head-bin agreement was 95.4%
        ## on block 0 and 40.9% on block 1, two routes of near-identical length.
        ## Never report an arm number without this table under it.
        print(f"     {'blk':>4} {'nrep':>5} {'head':>7} {'mid':>7} {'tail':>7} "
              f"{'plan[0]':>9} {'plan[-1]':>9} {'frozen':>7} {'curs_max':>9}")
        for blk in sorted({r['block'] for r in rs}):
            br = [r for r in rs if r['block'] == blk]
            bp = [p for r in br for p in r['probe'] if 'probe_error' not in p]
            be = [e for r in br for e in r['eps']]
            if not bp:
                continue
            gg = lambda k, src=bp: [p[k] for p in src if k in p]
            m = lambda k: (100 * np.mean(gg(k))) if gg(k) else float('nan')
            q = lambda v: np.percentile(v, 50) if len(v) else float('nan')
            print(f"     {blk:>4} {len(bp):>5} {m('head_agree_1cell'):>6.1f}% "
                  f"{m('mid_agree_1cell'):>6.1f}% {m('tail_agree_1cell'):>6.1f}% "
                  f"{q(gg('head_err_vs_true')):>7.2f}wu {q(gg('end_err_vs_goal')):>7.2f}wu "
                  f"{q([e['frozen_frac'] for e in be]):>7.3f} "
                  f"{q([e['cursor_max'] for e in be]):>9.0f}")
        print()

        out[arm] = dict(
            n_replans=len(pr), reached_model=float(np.mean(reached)),
            real_frac_p50=float(np.percentile(rf, 50)) if len(rf) else None,
            agree={b: (None if np.isnan(v) else float(v)) for b, v, _ in row},
            head_err_p50=float(np.percentile(he, 50)) if len(he) else None,
            end_err_p50=float(np.percentile(ee, 50)) if len(ee) else None,
        )

    print("READ THIS BEFORE THE NUMBERS: these are 200-step runs. Success rate is")
    print("meaningless at that budget on a 63 wu traverse and must not be quoted.")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        json.dump(out, open(a.out, 'w'), indent=2)
        print(f'\nwrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
