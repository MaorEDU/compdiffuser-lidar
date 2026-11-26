"""Read the 2x2 experiment off the rollout jsons: arms, blocks, mechanism, statistics.

    python -m diffuser.pl_eval.lidar_eval.rollout_report --runs "logs/**/plans/**/*.json"

Three things it enforces, each because the previous fork lost weeks to its absence:

1. **Protocol.** Every run is checked for N_EP=100 and a 1000-step budget, and flagged
   loudly otherwise. ``N_EP=20`` is ev_probs BLOCK 0 -- one start-goal pair jittered
   and repeated 20 times -- and success LATCHES, so a longer budget monotonically buys
   score. A table that silently mixes budgets is worse than no table.

2. **Mechanism before score.** For every arm it prints replans/episode against the
   cadence it asked for, and the frozen-step share. An arm whose mechanism did not
   move did not run, whatever it scored -- and the fork has a documented case of a fix
   whose mechanism fired perfectly while the score did not follow (p = 0.83).

3. **Paired tests.** Arms share problems and seeds, so episodes pair up. Reports
   McNemar on the paired wins rather than the difference of two means, plus the
   per-block split, because the five blocks are not one task (their success rates
   ranged 12 to 90 in the fork's measurements).

Arms are identified from the json fields the rollout writes (`ball_hist_on`,
`is_replan`, `repl_ada_dist_cfg.repl_every`, `ball_pos_source`), never from the
folder name.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from math import erfc, sqrt

import numpy as np

N_BLOCKS = 5          # ev_probs is 5 tasks x 20 problems
EXPECTED_N_EP = 100
EXPECTED_STEPS = 1000


def _parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True,
                   help='rollout json paths or globs')
    p.add_argument('--control', default='',
                   help='arm label to test against (default: the hist0/re0 arm)')
    p.add_argument('--out', default='')
    return p.parse_args()


def _load(paths):
    files = []
    for pat in paths:
        files.extend(sorted(glob.glob(pat, recursive=True)))
    runs = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        if 'ep_is_suc' not in d:
            continue
        d['_path'] = f
        runs.append(d)
    return runs


def _arm_label(d):
    hist = int(bool(d.get('ball_hist_on', False)))
    every = int((d.get('repl_ada_dist_cfg') or {}).get('repl_every', 0) or 0)
    src = d.get('ball_pos_source', '?')
    ladder = {'env': 'L1', 'belief': 'L3'}.get(src, src)
    return f'{ladder}/hist{hist}/re{every}'


def _mcnemar(a, b):
    """Two-sided McNemar on paired binary outcomes. Returns (n_a_only, n_b_only, p)."""
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    z = (abs(n01 - n10) - 1) / sqrt(n)          # continuity-corrected
    return n01, n10, float(erfc(abs(z) / sqrt(2)))


def main():
    a = _parse()
    runs = _load(a.runs)
    if not runs:
        raise SystemExit(f'no rollout jsons matched {a.runs}')

    arms = defaultdict(list)
    for d in runs:
        arms[_arm_label(d)].append(d)

    print(f'\nfound {len(runs)} run(s) across {len(arms)} arm(s)\n')

    ## ---- protocol check, before anything is tabulated ----
    problems = []
    for d in runs:
        n_ep = len(d['ep_is_suc'])
        steps = (d.get('repl_ada_dist_cfg') or {}).get('n_max_steps')
        if n_ep != EXPECTED_N_EP:
            problems.append(f'{os.path.basename(d["_path"])}: N_EP={n_ep} '
                            f'(expected {EXPECTED_N_EP}; 20 is BLOCK 0 = one task)')
        if steps != EXPECTED_STEPS:
            problems.append(f'{os.path.basename(d["_path"])}: n_max_steps={steps} '
                            f'(expected {EXPECTED_STEPS}; success latches, so budget '
                            f'buys score)')
    if problems:
        print('*** PROTOCOL WARNINGS -- these runs are not comparable as-is ***')
        for m in problems:
            print('   ! ' + m)
        print()

    ## ---- per-arm summary ----
    summary = {}
    print(f'{"arm":22s} {"seeds":>5s} {"succ%":>13s} {"repl/ep":>8s} '
          f'{"frozen%":>8s} {"locErr50":>9s}')
    print('-' * 72)
    for label in sorted(arms):
        ds = arms[label]
        rates, repls, frozen, locerr = [], [], [], []
        for d in ds:
            suc = np.asarray(d['ep_is_suc'], dtype=bool)
            rates.append(100.0 * suc.mean())
            repls.append(float(np.mean(d.get('ep_cnt_repls', [0]))))
            eps = d.get('ball_ep_summaries', []) or []
            if eps:
                frozen.append(100.0 * float(np.mean([e.get('frozen_frac', 0.0) for e in eps])))
                locerr.append(float(np.mean([e.get('loc_err_p50', 0.0) for e in eps])))
        m, s = float(np.mean(rates)), float(np.std(rates))
        summary[label] = dict(
            n_seeds=len(ds), succ_mean=m, succ_sd=s,
            repl_per_ep=float(np.mean(repls)) if repls else 0.0,
            frozen_pct=float(np.mean(frozen)) if frozen else float('nan'),
            loc_err_p50=float(np.mean(locerr)) if locerr else float('nan'),
            paths=[d['_path'] for d in ds])
        print(f'{label:22s} {len(ds):5d} {m:7.1f} +- {s:4.1f} '
              f'{summary[label]["repl_per_ep"]:8.1f} '
              f'{summary[label]["frozen_pct"]:8.1f} '
              f'{summary[label]["loc_err_p50"]:9.3f}')

    ## ---- mechanism gate ----
    print('\n--- mechanism gate (read this BEFORE the score) ---')
    for label in sorted(arms):
        every = int(label.split('/re')[-1])
        got = summary[label]['repl_per_ep']
        if every > 0:
            want = EXPECTED_STEPS / every
            ok = got >= 0.6 * want
            print(f'  {label:22s} replans/ep {got:5.1f} vs ~{want:.0f} expected '
                  f'-> {"ok" if ok else "**DID NOT RUN AT THIS CADENCE**"}')
        else:
            print(f'  {label:22s} replans/ep {got:5.1f} (ada_dist trigger, no cadence)')

    ## ---- per-block split ----
    print('\n--- per block (the five tasks are not one task) ---')
    hdr = ' '.join(f'blk{b:<4d}' for b in range(N_BLOCKS))
    print(f'{"arm":22s} {hdr}')
    for label in sorted(arms):
        cells = []
        for b in range(N_BLOCKS):
            vals = []
            for d in arms[label]:
                suc = np.asarray(d['ep_is_suc'], dtype=bool)
                per = len(suc) // N_BLOCKS
                if per:
                    vals.append(100.0 * suc[b * per:(b + 1) * per].mean())
            cells.append(f'{np.mean(vals):5.1f}' if vals else '  n/a')
        summary[label]['per_block'] = [float(c) if c.strip() != 'n/a' else None
                                       for c in cells]
        print(f'{label:22s} ' + '  '.join(cells))

    ## ---- paired tests against the control ----
    ctrl = a.control or next((l for l in sorted(arms) if '/hist0/re0' in l), None)
    if ctrl and ctrl in arms:
        print(f'\n--- paired McNemar vs control [{ctrl}] ---')
        ctrl_by_seed = {i: np.asarray(d['ep_is_suc'], dtype=bool)
                        for i, d in enumerate(arms[ctrl])}
        for label in sorted(arms):
            if label == ctrl:
                continue
            n01 = n10 = 0
            ps = []
            for i, d in enumerate(arms[label]):
                c = ctrl_by_seed.get(i)
                t = np.asarray(d['ep_is_suc'], dtype=bool)
                if c is None or len(c) != len(t):
                    continue
                x, y, p = _mcnemar(t, c)
                n01 += x; n10 += y; ps.append(p)
            if not ps:
                print(f'  {label:22s} no comparable seed pairing')
                continue
            _, _, p_all = _mcnemar(
                np.r_[np.ones(n01, bool), np.zeros(n10, bool)],
                np.r_[np.zeros(n01, bool), np.ones(n10, bool)])
            d_pts = summary[label]['succ_mean'] - summary[ctrl]['succ_mean']
            print(f'  {label:22s} {d_pts:+6.1f} pts   arm-only wins {n01:4d}  '
                  f'control-only {n10:4d}   p={p_all:.3g}')
        summary['_control'] = ctrl

    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'\nwrote {a.out}')
    print()


if __name__ == '__main__':
    main()
