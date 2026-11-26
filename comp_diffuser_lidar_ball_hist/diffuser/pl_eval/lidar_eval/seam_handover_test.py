"""Can `jump_frac` be collapsed by changing the COMPOSER, with no retrain? -- CPU only.

G2a's `jump_frac <= 0.01` is the one gate bench100 is blocked on. `g2a_floor.py` showed
the bar is reachable (walkable paths score 0.0009 through the shipped decoder) and
`jump_census.py` showed 88.3% of the jump mass sits in the 44.1% of frames that are
BLENDED, peaking exactly at the `exp_beta=2` w=0.5 crossover. The r24/hist16 retrain then
moved `jump_frac` 0.028 -> 0.025 while moving anchoring 0.850 -> 0.970, which says the
defect is not reachable from the checkpoint at all.

That leaves the composer. `blend_2_np_trajs_23d` takes an elementwise weighted mean of
two segment trajectories. For the paper's `o2d` branch those are (x, y) and a weighted
mean of two positions is a position. Here they are 30-beam RANGE VECTORS, and a weighted
mean of two range vectors is **a scan from no pose** -- maximally so at w=0.5, which is
where the jumps peak. This file asks whether that is actually the cause, by rebuilding
the same plans three ways and scoring G2a on each:

  blend    the shipped elementwise scan-space blend. CONTROL -- must reproduce the
           plan the gate actually scored, and is asserted to.
  hard     the same weight curve, thresholded: each overlap frame taken whole from its
           owning segment. Removes the off-manifold scans, but hands over abruptly, so
           it trades a distributed artefact for up to `n_comp - 1` sharp switches.
  xyblend  decode each segment to (x, y) FIRST, then blend the positions with the same
           weights. This is the paper's own branch semantics restored: it keeps the
           smooth crossover AND never averages two range vectors. Costs n_comp decodes
           per plan instead of one.

Every arm ends as a (tot_hzn, 2) plan and is scored with `plan_decode_gate`'s exact
expressions, through the SHIPPED `_decode_plan_xy` (called unbound on `g2a_floor`'s shim,
so stride / median / interp / 3-tap / endpoint-pin are the rollout's, not a copy).

Reads the npz written by `dump_plans.py`, which must carry `segs` -- the pre-blend
per-segment trajectories. A blended plan cannot be inverted back into its segments, so a
dump without `segs` cannot answer this and is refused rather than approximated.

Usage:
    python -m diffuser.pl_eval.lidar_eval.seam_handover_test \
        --npz logs/lidar_eval/r24_hist16/plans_dump_hist0.npz
"""
import argparse
import json
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.append('./')

import numpy as np

from diffuser.datasets.lidar import (
    make_scanner_for_env, load_or_build_grid, GridLocalizer)
from diffuser.pl_eval.lidar_eval.decode_bench import assemble, seam_spans, blend_weights
from diffuser.pl_eval.lidar_eval.g2a_floor import _DecodeShim, g2a_metrics
from diffuser.pl_eval.lidar_eval.plan_decode_gate import G2A_FREE_FRAC, G2A_JUMP_FRAC

ENV = 'pointmaze-giant-stitch-v0'
ARMS = ('blend', 'hard', 'xyblend')


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', default='logs/lidar_eval/r24_hist16/plans_dump_hist0.npz')
    p.add_argument('--stride', type=int, default=2, help='ball_decode_stride')
    p.add_argument('--median', type=int, default=15, help='LIDAR_DECODE_MEDIAN')
    p.add_argument('--motion_sigma', type=float, default=1.0)
    p.add_argument('--loc_sigma', type=float, default=1.0, help='loc_grid_sigma')
    p.add_argument('--decode', default='argmin', choices=['argmin', 'filter'])
    p.add_argument('--out', default=None,
                   help='default: <npz dir>/seam_handover.json')
    return p.parse_args()


def _seam_mask(n_comp, hzn, len_ovlp, tot):
    m = np.zeros(tot, dtype=bool)
    for lo, hi in seam_spans(n_comp, hzn, len_ovlp):
        m[lo:hi] = True
    return m


def _switch_frames(n_comp, hzn, len_ovlp, w):
    """The frames where `hard` flips from segment i to segment i+1.

    `assemble` takes segment i wherever w >= 0.5, so the flip lands on the first
    offset where that stops being true. Reported separately because a hard handover
    is *expected* to concentrate whatever it leaves at exactly these frames.
    """
    off = int(np.argmax(w < 0.5)) if (w < 0.5).any() else len(w)
    return [lo + off for lo, _ in seam_spans(n_comp, hzn, len_ovlp)]


def main():
    a = _parse()
    d = np.load(a.npz, allow_pickle=True)
    if 'segs' not in d.files:
        raise SystemExit(
            f'{a.npz} has no `segs` -- it predates the pre-blend segment dump. The '
            f'blend is lossy and cannot be inverted, so the seam counterfactual is '
            f'unanswerable on this file. Re-run dump_plans.py (it now saves `segs`).')

    segs_all = d['segs']                       # (n_plans, n_comp, hzn, D)
    plans = d['plans']                         # (n_plans, tot_hzn, D) as shipped
    n_plans, n_comp, hzn, D = segs_all.shape
    len_ovlp = int(d['len_ovlp'])
    blend_type = str(d['blend_type'])
    exp_beta = float(d['exp_beta'])
    tot = n_comp * hzn - (n_comp - 1) * len_ovlp
    max_range = float(d['max_range'])
    n_beams = int(d['n_beams'])

    ## prefer the planner's own config, recorded by dump_plans; the fallback keeps
    ## older dumps readable. `make_scanner_for_env` ignores keys it does not know.
    if 'lidar_config' in d.files:
        lidar_config = json.loads(str(d['lidar_config']))
        lidar_config['n_beams'] = n_beams
        lidar_config['max_range'] = max_range
    else:
        lidar_config = dict(n_beams=n_beams, max_range=max_range, fov=2.0 * np.pi,
                            fov_centered=False, step=0.1)
    scanner = make_scanner_for_env(ENV, lidar_config)
    grid = load_or_build_grid(ENV, lidar_config, xy_step=0.5, n_theta=1, verbose=False)
    loc = GridLocalizer(grid, sigma=a.loc_sigma)
    cell = float(scanner.maze_unit)

    print(f'[seam] {a.npz}')
    print(f'[seam] {n_plans} plans  n_comp={n_comp} hzn={hzn} len_ovlp={len_ovlp} '
          f'tot={tot} D={D}  blend={blend_type} beta={exp_beta}')
    print(f'[seam] epoch={int(d["epoch"])} hist={int(d["hist"])} '
          f'max_range={max_range:g} n_beams={n_beams} cell={cell:g} wu')
    print(f'[seam] decoder={a.decode} stride={a.stride} median={a.median} '
          f'loc_sigma={a.loc_sigma}', flush=True)

    w = blend_weights(len_ovlp, blend_type, exp_beta)
    seam = _seam_mask(n_comp, hzn, len_ovlp, tot)
    switches = _switch_frames(n_comp, hzn, len_ovlp, w)
    print(f'[seam] seam frames {seam.sum()}/{tot} = {100*seam.mean():.1f}%   '
          f'hard-handover switch frames at {switches}', flush=True)

    def decode(scans):
        shim = _DecodeShim(scanner, loc, a.decode, a.stride, a.median, a.motion_sigma)
        return shim.decode(scans)

    rows = {k: [] for k in ARMS}
    ctrl_res, seg_gap = [], []
    off_sw = int(np.argmax(w < 0.5)) if (w < 0.5).any() else len_ovlp - 1

    for i_p in range(n_plans):
        sg = segs_all[i_p].astype(np.float32)

        for arm in ARMS:
            if arm == 'xyblend':
                ## decode each segment to (x, y) first, then blend POSITIONS
                seg_xy = np.stack([decode(sg[i_c]) for i_c in range(n_comp)])
                ## Free diagnostic off the same decodes: how far apart do two
                ## consecutive segments actually place the overlap they SHARE? This
                ## is the dose the composer is being asked to absorb, in world units.
                ## It bounds what a hard handover can do -- a hard switch leaves a
                ## step of exactly this size at the crossover.
                gap = [np.linalg.norm(seg_xy[i][hzn - len_ovlp:] - seg_xy[i + 1][:len_ovlp],
                                      axis=1) for i in range(n_comp - 1)]
                gap = np.concatenate(gap)
                seg_gap.append(dict(i_p=i_p, p50=float(np.median(gap)),
                                    p90=float(np.percentile(gap, 90)),
                                    at_switch=float(np.median([
                                        np.linalg.norm(seg_xy[i][hzn - len_ovlp + off_sw]
                                                       - seg_xy[i + 1][off_sw])
                                        for i in range(n_comp - 1)]))))
                xy = assemble(seg_xy.astype(np.float32), n_comp, hzn, len_ovlp,
                              blend_type, exp_beta, mode='blend').astype(np.float64)
            else:
                scans = assemble(sg, n_comp, hzn, len_ovlp,
                                 blend_type, exp_beta, mode=arm)
                if arm == 'blend':
                    ## CONTROL: this must be the plan the gate actually scored
                    ctrl_res.append(float(np.max(np.abs(
                        scans.astype(np.float64) - plans[i_p].astype(np.float64)))))
                xy = decode(scans)

            f, j, step = g2a_metrics(xy, scanner, cell)
            jmp = step > cell                      # jump STARTS, index t -> t..t+1
            in_seam = int(jmp[seam[:len(jmp)]].sum())
            at_switch = int(sum(bool(jmp[t]) for t in switches if t < len(jmp)))
            rows[arm].append(dict(
                i_p=i_p, free_frac=f, jump_frac=j,
                n_jump=int(jmp.sum()), jump_in_seam=in_seam,
                jump_at_switch=at_switch,
                arc=float(step.sum()),
                step_p99=float(np.percentile(step, 99)),
                end_xy=xy[-1].tolist()))

        if (i_p + 1) % 5 == 0 or i_p == 0:
            msg = '  '.join(f'{k} jump={rows[k][-1]["jump_frac"]:.4f}' for k in ARMS)
            print(f'[seam] plan {i_p:3d}  {msg}', flush=True)

    ## ------------------------------------------------------------- control check
    ctrl = np.asarray(ctrl_res)
    print(f'\n[seam] control: assemble(mode="blend") vs the shipped plan -- '
          f'max residual {ctrl.max():.3e}')
    assert ctrl.max() < 1e-3, (
        f'the `blend` arm does not reproduce the dumped plan (max {ctrl.max():.3e}); '
        f'the composer geometry here disagrees with the shipped one, so neither '
        f'counterfactual is interpretable.')

    ## ------------------------------------------------------------------ aggregate
    def agg(arm):
        r = rows[arm]
        jf = np.array([x['jump_frac'] for x in r])
        ff = np.array([x['free_frac'] for x in r])
        return dict(
            arm=arm, n=len(r),
            jump_frac=float(jf.mean()), jump_frac_p50=float(np.median(jf)),
            jump_frac_p90=float(np.percentile(jf, 90)), jump_frac_max=float(jf.max()),
            free_frac=float(ff.mean()), free_frac_min=float(ff.min()),
            n_over_bar=int((jf > G2A_JUMP_FRAC).sum()),
            n_jump=float(np.mean([x['n_jump'] for x in r])),
            jump_in_seam=float(np.mean([x['jump_in_seam'] for x in r])),
            jump_at_switch=float(np.mean([x['jump_at_switch'] for x in r])),
            arc=float(np.mean([x['arc'] for x in r])),
            step_p99=float(np.mean([x['step_p99'] for x in r])),
            g2a_pass=bool(ff.mean() >= G2A_FREE_FRAC and jf.mean() <= G2A_JUMP_FRAC),
        )

    summ = {k: agg(k) for k in ARMS}

    gap_p50 = float(np.mean([g['p50'] for g in seg_gap]))
    gap_p90 = float(np.mean([g['p90'] for g in seg_gap]))
    gap_sw = float(np.mean([g['at_switch'] for g in seg_gap]))
    gaps = dict(overlap_gap_p50=gap_p50, overlap_gap_p90=gap_p90,
                overlap_gap_at_switch=gap_sw, cell=cell)
    print(f'\n[seam] segment disagreement over the shared overlap, decoded to xy: '
          f'p50 {gap_p50:.2f} wu  p90 {gap_p90:.2f} wu  '
          f'at the switch frame {gap_sw:.2f} wu  (1 cell = {cell:g} wu)')
    print(f'[seam] a hard switch leaves a step of ~{gap_sw:.2f} wu at each of the '
          f'{n_comp - 1} crossovers; the jump threshold is {cell:g} wu')

    ## paired against the control, per plan -- the plans are the same 40 either way
    base = np.array([x['jump_frac'] for x in rows['blend']])
    paired = {}
    for arm in ARMS:
        if arm == 'blend':
            continue
        v = np.array([x['jump_frac'] for x in rows[arm]])
        dlt = v - base
        n_better = int((dlt < 0).sum())
        n_worse = int((dlt > 0).sum())
        ## exact two-sided sign test, no scipy dependency
        from math import comb
        n_eff = n_better + n_worse
        k = min(n_better, n_worse)
        p = min(1.0, 2.0 * sum(comb(n_eff, i) for i in range(k + 1)) / 2 ** n_eff) \
            if n_eff else 1.0
        paired[arm] = dict(
            vs='blend', n=len(v), median_delta=float(np.median(dlt)),
            rel_change=float(v.mean() / base.mean() - 1.0) if base.mean() else 0.0,
            n_better=n_better, n_worse=n_worse, sign_test_p=float(p))

    ## ---------------------------------------------------------------------- report
    print('\n' + '=' * 78)
    print(f'{"arm":>9} {"jump_frac":>10} {"p90":>8} {"max":>8} {"free":>8} '
          f'{"over bar":>9} {"jumps":>7} {"in seam":>8} {"at switch":>10}')
    for k in ARMS:
        s = summ[k]
        print(f'{k:>9} {s["jump_frac"]:10.4f} {s["jump_frac_p90"]:8.4f} '
              f'{s["jump_frac_max"]:8.4f} {s["free_frac"]:8.4f} '
              f'{s["n_over_bar"]:>4}/{s["n"]:<4} {s["n_jump"]:7.1f} '
              f'{s["jump_in_seam"]:8.1f} {s["jump_at_switch"]:10.2f}')
    print(f'\nG2a bars: free >= {G2A_FREE_FRAC}, jump <= {G2A_JUMP_FRAC}')
    for arm, pr in paired.items():
        print(f'  {arm:>8} vs blend: {100*pr["rel_change"]:+.1f}%  '
              f'median delta {pr["median_delta"]:+.4f}  '
              f'{pr["n_better"]} better / {pr["n_worse"]} worse  '
              f'sign p={pr["sign_test_p"]:.3g}')
    winners = [k for k in ARMS if summ[k]['g2a_pass']]
    print(f'\n  G2a would PASS under: {winners if winners else "-- none --"}')
    print('=' * 78)

    out = a.out or os.path.join(os.path.dirname(a.npz) or '.', 'seam_handover.json')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(dict(
            npz=a.npz, env=ENV, epoch=int(d['epoch']), hist=int(d['hist']),
            n_plans=n_plans, n_comp=n_comp, hzn=hzn, len_ovlp=len_ovlp, tot_hzn=tot,
            blend_type=blend_type, exp_beta=exp_beta,
            max_range=max_range, n_beams=n_beams, cell=cell,
            decode=a.decode, stride=a.stride, median=a.median,
            loc_sigma=a.loc_sigma,
            g2a_free_bar=G2A_FREE_FRAC, g2a_jump_bar=G2A_JUMP_FRAC,
            control_max_residual=float(ctrl.max()),
            seam_frame_frac=float(seam.mean()), switch_frames=switches,
            summary=summ, paired=paired, segment_gap=gaps,
            per_plan={k: rows[k] for k in ARMS},
        ), f, indent=2)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
