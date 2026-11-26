#!/usr/bin/env python3
"""BALL failure-vs-uncertainty calibration (BALL.md Phase 6.3 / S2).

Question answered: *does the distribution of successes and failures align with
the measured uncertainty?* Episodes are bucketed by pre-registered uncertainty
scores and the per-bucket success rate is reported (reliability table) plus a
Brier score for the induced predictor.

Inputs: one or more rollout jsons produced by the ball rollout
(00_rollout*.json; the ball planner writes `ball_ep_summaries` with
per-episode belief confidence / localization error, and `ep_is_suc`).

Scores used (whichever are present):
  * belief_conf_mean  -- mean MAP posterior mass (self-localization certainty)
  * loc_err_mean      -- mean |belief - true| (diagnostic; not lidar-available)
  * mass_true         -- goal-scan posterior mass at the true cell, joined from
                         a census.json (--census) by problem index, if given.

Usage
-----
  python -m diffuser.pl_eval.lidar_eval.ball_calibration \
      --jsons 'logs/pointmaze-giant-stitch-v0/plans/**/00_rollout*.json' \
      --census logs/lidar_eval/ball_census_giant/census.json \
      --n_buckets 4 --out logs/lidar_eval/ball_calibration.json
"""
import os, sys, json, glob, argparse
import numpy as np


def buckets_table(score, suc, n_buckets):
    """Equal-count buckets by score -> per-bucket success rate."""
    score = np.asarray(score, dtype=np.float64)
    suc = np.asarray(suc, dtype=np.float64)
    order = np.argsort(score)
    rows = []
    for chunk in np.array_split(order, n_buckets):
        if len(chunk) == 0:
            continue
        rows.append(dict(
            n=int(len(chunk)),
            score_lo=float(score[chunk].min()), score_hi=float(score[chunk].max()),
            score_mean=float(score[chunk].mean()),
            success_rate=float(suc[chunk].mean())))
    ## Brier of the per-bucket-rate predictor (in-sample; a coarse calibration stat)
    pred = np.empty_like(suc)
    for row, chunk in zip(rows, np.array_split(order, n_buckets)):
        pred[chunk] = row['success_rate']
    brier = float(np.mean((pred - suc) ** 2))
    ## monotonicity: does success rise with the score? (Spearman-lite: bucket trend)
    rates = [r['success_rate'] for r in rows]
    mono = float(np.mean(np.sign(np.diff(rates)))) if len(rates) > 1 else 0.0
    return rows, brier, mono


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsons', required=True,
                    help="glob for rollout jsons, e.g. 'logs/**/00_rollout*.json'")
    ap.add_argument('--census', default='', help='census.json to join mass_true by prob id')
    ap.add_argument('--n_buckets', type=int, default=4)
    ap.add_argument('--out', default='logs/lidar_eval/ball_calibration.json')
    args = ap.parse_args()

    files = sorted(glob.glob(args.jsons, recursive=True))
    assert files, f'no jsons match {args.jsons!r}'
    eps = []
    for fp in files:
        d = json.load(open(fp))
        suc = d.get('ep_is_suc', [])
        summ = {s['ep']: s for s in d.get('ball_ep_summaries', [])}
        st_idx = int(d.get('ep_st_idx', 0) or 0)
        for k, s in enumerate(suc):
            row = dict(json=os.path.basename(os.path.dirname(fp)), ep=k,
                       prob_id=st_idx + k, suc=bool(s))
            row.update({kk: vv for kk, vv in summ.get(k, {}).items()
                        if kk in ('loc_err_mean', 'loc_err_max', 'belief_conf_mean',
                                  'deadlock_step', 'n_steps', 'target_dist_mean')})
            eps.append(row)
    print(f'[calib] {len(eps)} episodes from {len(files)} jsons '
          f'(success rate {np.mean([e["suc"] for e in eps])*100:.1f}%)')

    if args.census:
        rows = json.load(open(args.census))['rows']
        mass = {r['prob_id']: r['mass_true'] for r in rows if r['src'] == 'prob'}
        rank = {r['prob_id']: r['true_rank'] for r in rows if r['src'] == 'prob'}
        for e in eps:
            if e['prob_id'] in mass:
                e['mass_true'] = mass[e['prob_id']]
                e['goal_true_rank'] = rank[e['prob_id']]

    out = dict(n_episodes=len(eps),
               success_rate=float(np.mean([e['suc'] for e in eps])),
               deadlock_rate=float(np.mean([e.get('deadlock_step', -1) >= 0 for e in eps])),
               tables={})
    for key in ('belief_conf_mean', 'mass_true', 'loc_err_mean'):
        have = [e for e in eps if key in e and e[key] is not None]
        if len(have) >= 2 * args.n_buckets:
            rows, brier, mono = buckets_table([e[key] for e in have],
                                              [e['suc'] for e in have], args.n_buckets)
            out['tables'][key] = dict(buckets=rows, brier=brier, bucket_monotonicity=mono)
            print(f'[calib] {key}: ' + ' | '.join(
                f"[{r['score_lo']:.3f},{r['score_hi']:.3f}]->{r['success_rate']*100:.0f}%"
                for r in rows) + f'  (brier {brier:.3f}, mono {mono:+.2f})')
        else:
            print(f'[calib] {key}: not enough episodes with this field ({len(have)})')
    ## the project's alignment verdict: uncertainty-aligned failure = success rises with
    ## confidence/mass and falls with loc error. Misaligned = flat/inverted ->
    ## the failures are algorithmic, not perceptual.
    out['episodes'] = eps
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    print(f'[calib] wrote {args.out}')


if __name__ == '__main__':
    main()
