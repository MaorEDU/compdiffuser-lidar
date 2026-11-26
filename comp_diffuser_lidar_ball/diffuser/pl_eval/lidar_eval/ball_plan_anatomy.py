#!/usr/bin/env python3
"""Anatomy of a dumped plan: separate GOAL ANCHORING from ROUTE COHERENCE.

Why this tool exists (2026-07-29, after the bridge test passed)
---------------------------------------------------------------
G2b is defined as "the DECODED endpoint lands in the goal cell (+-1)", and the
decode is ``GridLocalizer.localize_sequence(..., motion_sigma_cells=1.0)`` -- a
*filtered* estimate that propagates a belief along the whole route. Meanwhile the
composed sampler HARD-INPAINTS the goal scan into the last frame of the last
segment (``stgl_sml_diffusion_v1.comp_pred_p_loop_n``, the
``apply_conditioning(x_p_list[-1], {hzn-1: goal})`` after the loop).

Those two facts cannot both be innocent: if the final frame *is* the goal scan,
a healthy decode must place it at the goal, yet composed dumps report end err
~35.5 wu. So on composed plans **G2b is entangled with G2a**: when the route
teleports (median 6 multi-cell jumps), the filter cannot trust the final
measurement and drags the endpoint to wherever the (broken) route led. G2b has
never been a pure anchoring measure on composed plans.

This script decouples them, with no GPU and no resampling, on dumps already on
disk:

  * ``goal_resid``   -- is the inpainted goal scan actually PRESENT at frame H-1?
  * ``map_end_err``  -- localize the final frame ALONE (``localize_single``, no
                        motion prior). This is anchoring with coherence removed.
  * ``filt_end_err`` -- the G2 definition, for reference. The gap between the two
                        is exactly the contamination.
  * ``map_resid``    -- per-frame distance to the nearest map-realizable scan
                        (``grid.free_scans``). "Is this scan even a possible
                        observation?" -- the quantity an in-loop map projection
                        would act on.
  * MAP-vs-filtered disagreement, belief entropy, and jump positions vs the
    composition seams, all as per-frame profiles.
  * ``bfs_gdist``    -- BFS *maze* distance to the goal per frame, from the
                        MAP-alone decode. The progress profile (2026-07-30 item
                        7) was originally read off ``goal_dist_p50``, which is
                        STRAIGHT-LINE and understates maze progress around every
                        corner; the "0.019 wu/frame early" figure was therefore a
                        lower bound, not a measurement. This is the honest one,
                        and it is reported as a fraction of the problem's own
                        initial BFS distance so plans of different lengths pool.

Usage (CPU, seconds-to-minutes)
-------------------------------
  python -m diffuser.pl_eval.lidar_eval.ball_plan_anatomy \
      --plans logs/lidar_eval/ball_plans_lidar_1600000.npz \
      --probs data/ogb_maze/ev_probs/ogb_pointM_Gi_ev_prob_numEp20_eSdSt0.hdf5 \
      --out logs/lidar_eval/ball_anatomy_16M.json

  # positive control: the o2d composed dump (D=2, identity decode)
  python -m diffuser.pl_eval.lidar_eval.ball_plan_anatomy \
      --plans logs/lidar_eval/ball_plans_o2d.npz --probs <same> \
      --out logs/lidar_eval/ball_anatomy_o2d.json

  # bridge dumps are self-describing (carry starts/goals/gt_xy): no --probs needed
  python -m diffuser.pl_eval.lidar_eval.ball_plan_anatomy \
      --plans logs/lidar_eval/ball_bridge_lidar.npz \
      --out logs/lidar_eval/ball_anatomy_bridge.json
"""
import os, sys, json, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
from diffuser.pl_eval.lidar_eval.lidar_grid_common import (
    LG, LL, resolve_lidar_config, json_safe_lidar_config, print_color,
    maze_bfs_wu)


def load_dump(path):
    d = np.load(path, allow_pickle=True)
    plans = np.asarray(d['plans'], dtype=np.float32)
    meta = str(d['meta']) if 'meta' in d.files else None
    extra = {k: np.asarray(d[k]) for k in ('starts', 'goals', 'gt_xy') if k in d.files}
    return plans, meta, extra


def seam_frames(H, horizon, n_comp):
    """Frame indices where a new composed segment starts (stride = horizon-ovlp)."""
    if n_comp is None or n_comp < 2:
        return [], 0
    stride = int(round((H - horizon) / (n_comp - 1)))
    return [k * stride for k in range(1, n_comp)], stride


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plans', required=True)
    ap.add_argument('--probs', default='',
                    help='ev_probs hdf5; ignored when the dump carries starts/goals')
    ap.add_argument('--env', default='pointmaze-giant-stitch-v0')
    ap.add_argument('--n_probs', type=int, default=0, help='0 = all in the dump')
    ap.add_argument('--n_dec', type=int, default=8, help='resamples decoded per problem')
    ap.add_argument('--stride', type=int, default=2, help='frame stride (G2 uses 2)')
    ap.add_argument('--xy_step', type=float, default=0.5)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--motion_sigma', type=float, default=1.0)
    ap.add_argument('--horizon', type=int, default=160, help='per-segment horizon')
    ap.add_argument('--arrive_offsets', type=int, nargs='+', default=[0, 8, 16, 32, 64],
                    help='measure the endpoint this many frames BEFORE the inpainted '
                         'goal frame; offset 0 == the inpainted frame itself')
    ap.add_argument('--cache_root', default='data/ogb_maze/lidar_grid')
    ## geometry overrides; default = resolved from the dump meta / its run dir
    ap.add_argument('--planner_logdir', default='')
    ap.add_argument('--n_beams', type=int, default=0)
    ap.add_argument('--max_range', type=float, default=0.0)
    ap.add_argument('--out', default='logs/lidar_eval/ball_anatomy.json')
    ap.add_argument('--png', default='', help='default = <out>.png')
    args = ap.parse_args()

    plans, meta, extra = load_dump(args.plans)
    mj = json.loads(meta) if meta else {}
    lidar_cfg = resolve_lidar_config(logdir=args.planner_logdir, meta=meta,
                                     n_beams=args.n_beams, max_range=args.max_range)
    P, N, H, D = plans.shape
    n_comp = int(mj.get('n_comp', 0)) or None
    seams, stride_seam = seam_frames(H, args.horizon, n_comp)
    print_color(f'[anatomy] {args.plans}: plans={plans.shape} n_comp={n_comp} '
                f'seams at {seams[:8]} (stride {stride_seam})', c='c')

    ## ---- problems ----
    if 'starts' in extra:
        starts = np.asarray(extra['starts'], np.float64)
        goals = np.asarray(extra['goals'], np.float64)
        print_color(f'[anatomy] problems read from the dump itself ({len(starts)})', c='g')
    else:
        assert args.probs, '--probs required: this dump does not carry starts/goals'
        from diffuser.utils.cp_utils.cp_serial import load_stgl_lh_ev_probs_hdf5
        pr = load_stgl_lh_ev_probs_hdf5(args.probs)
        starts = np.asarray(pr['start_state'], np.float64)[:P, :2]
        goals = np.asarray(pr['goal_pos'], np.float64)[:P, :2]
    P = min(P, args.n_probs) if args.n_probs else P
    n_dec = min(args.n_dec, N)

    grid = LG.load_or_build_grid(args.env, lidar_cfg, xy_step=args.xy_step, n_theta=1,
                                cache_root=args.cache_root)
    scanner, loc = grid.scanner, LL.GridLocalizer(grid, sigma=args.sigma)
    unit = scanner.maze_unit

    ## ---- the conditioning scans the sampler was given ----
    if D == 2:
        st_ref, gl_ref = starts.astype(np.float32), goals.astype(np.float32)
    else:
        from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env
        sc = make_scanner_for_env(args.env, lidar_cfg)
        st_ref = np.stack([sc.scan(x, y, 0.0) for x, y in starts]).astype(np.float32)
        gl_ref = np.stack([sc.scan(x, y, 0.0) for x, y in goals]).astype(np.float32)

    idx = np.unique(np.concatenate([np.arange(0, H, args.stride), [H - 1]]))
    T = len(idx)
    rows = []
    prof_resid = np.zeros((0, T)); prof_disag = np.zeros((0, T)); prof_ent = np.zeros((0, T))
    prof_gdist = np.zeros((0, T)); prof_bfs = np.zeros((0, T))

    ## BFS distance-to-goal field, one per distinct goal cell (cached: the giant
    ## maze BFS is ~ms but there are P*n_dec plans).
    _bfs_cache = {}

    def bfs_field(g):
        ci, cj = scanner.world_to_cell(np.array([g[0]]), np.array([g[1]]))
        k = (int(np.ravel(ci)[0]), int(np.ravel(cj)[0]))
        if k not in _bfs_cache:
            _bfs_cache[k] = maze_bfs_wu(scanner, k)
        return _bfs_cache[k]

    for p in range(P):
        for n in range(n_dec):
            pl = plans[p, n]
            if D == 2:
                map_xy = np.asarray(pl[idx], np.float64)
                filt_xy = map_xy.copy()
                map_resid = np.zeros(T); ent = np.zeros(T)
            else:
                ## independent per-frame MAP: no motion prior, no cross-frame info
                dmat = LL.scan_distance(pl[idx], grid.free_scans, chunk=512)   # (T, M)
                k = np.argmin(dmat, axis=1)
                map_xy = np.asarray(grid.free_xy[k], np.float64)
                map_resid = dmat[np.arange(T), k].astype(np.float64)
                ## the G2 definition: sequence-filtered decode
                r = loc.localize_sequence(pl[idx].astype(np.float32),
                                          motion_sigma_cells=args.motion_sigma,
                                          theta_blur=0)
                filt_xy = np.asarray(r['map_xy'], np.float64)
                ent = np.asarray(r['entropy'], np.float64)

            ## anchoring, two ways
            g = goals[p]
            map_end = float(np.linalg.norm(map_xy[-1] - g))
            filt_end = float(np.linalg.norm(filt_xy[-1] - g))

            ## ---- ARRIVAL: the endpoint EXCLUDING the inpainted goal frame ----
            ## Frame H-1 is hard-inpainted with the goal scan, so localizing it
            ## recovers the goal BY CONSTRUCTION -- the same round-trip artifact
            ## that makes start_err ~0.4 wu meaningless as a health check. The
            ## honest question is whether the plan ARRIVES on its own: how close
            ## is it a few frames BEFORE the inpainted end?
            arrive = {}
            for off in args.arrive_offsets:
                j = int(np.searchsorted(idx, max(H - 1 - off, 0)))
                j = min(j, len(idx) - 1)
                arrive[f'map_arrive_{off}'] = float(np.linalg.norm(map_xy[j] - g))
                arrive[f'filt_arrive_{off}'] = float(np.linalg.norm(filt_xy[j] - g))
            cg = np.array(scanner.world_to_cell(g[0:1], g[1:2])).ravel()
            ci_m = np.array(scanner.world_to_cell(map_xy[-1:, 0], map_xy[-1:, 1])).ravel()
            ci_f = np.array(scanner.world_to_cell(filt_xy[-1:, 0], filt_xy[-1:, 1])).ravel()

            ## is the inpainted conditioning actually present in the dumped array?
            nb = max(D, 1)
            goal_resid = float(np.sqrt(((pl[-1] - gl_ref[p]) ** 2).mean()))
            start_resid = float(np.sqrt(((pl[0] - st_ref[p]) ** 2).mean()))

            ## coherence, both decodes
            jf = np.where(np.linalg.norm(np.diff(filt_xy, axis=0), axis=1) > unit)[0]
            jm = np.where(np.linalg.norm(np.diff(map_xy, axis=0), axis=1) > unit)[0]
            disag = np.linalg.norm(map_xy - filt_xy, axis=1)

            rows.append(dict(
                problem=int(p), resample=int(n),
                goal_resid_rms=goal_resid, start_resid_rms=start_resid,
                map_end_err_wu=map_end, filt_end_err_wu=filt_end,
                map_end_in_goal_cell=bool(np.abs(ci_m - cg).max() <= 1),
                filt_end_in_goal_cell=bool(np.abs(ci_f - cg).max() <= 1),
                map_resid_p50=float(np.median(map_resid)),
                map_resid_p99=float(np.quantile(map_resid, 0.99)),
                disagree_p50=float(np.median(disag)), disagree_max=float(disag.max()),
                n_jumps_filt=int(len(jf)), n_jumps_map=int(len(jm)),
                jump_frames_filt=[int(idx[i + 1]) for i in jf],
                first_jump_filt=int(idx[jf[0] + 1]) if len(jf) else -1,
                **arrive))
            ## PROGRESS, in maze units, from the MAP-alone decode.
            ## Straight-line distance understates progress around every corner,
            ## so the 2026-07-30 "0.019 wu/frame early" figure was a lower bound.
            ## Normalised by the plan's own frame-0 BFS distance => 1.0 at the
            ## start, 0.0 at the goal, poolable across problems of any length.
            fld = bfs_field(g)
            bi, bj = scanner.world_to_cell(map_xy[:, 0], map_xy[:, 1])
            bd = fld[np.asarray(bi, int), np.asarray(bj, int)]
            d0 = bd[0]
            ## a decode that lands in a walled/unreachable cell gives inf; carry
            ## it forward as the last finite value rather than poisoning the median
            bd = np.asarray(bd, np.float64)
            if not np.isfinite(bd).all():
                good = np.isfinite(bd)
                bd = np.interp(np.arange(T), np.arange(T)[good], bd[good]) \
                    if good.any() else np.full(T, np.nan)
                d0 = bd[0]
            prof_bfs = np.vstack([prof_bfs,
                                  (bd / d0 if np.isfinite(d0) and d0 > 0
                                   else np.full(T, np.nan))[None]])
            rows[-1]['bfs_d0_wu'] = float(d0)
            rows[-1]['bfs_end_wu'] = float(bd[-1])

            prof_gdist = np.vstack([prof_gdist,
                                    np.linalg.norm(filt_xy - g, axis=1)[None]])
            prof_resid = np.vstack([prof_resid, map_resid[None]])
            prof_disag = np.vstack([prof_disag, disag[None]])
            prof_ent = np.vstack([prof_ent, ent[None]])

    def med(k):
        return float(np.median([r[k] for r in rows]))

    def frac(k):
        return float(np.mean([r[k] for r in rows]))

    agg = dict(
        n_plans=len(rows), n_problems=int(P), n_dec=int(n_dec), H=int(H), D=int(D),
        n_comp=n_comp, seams=seams,
        goal_resid_rms_med=med('goal_resid_rms'), start_resid_rms_med=med('start_resid_rms'),
        map_end_err_med=med('map_end_err_wu'), filt_end_err_med=med('filt_end_err_wu'),
        g2b_map=frac('map_end_in_goal_cell'), g2b_filt=frac('filt_end_in_goal_cell'),
        map_resid_p50_med=med('map_resid_p50'), map_resid_p99_med=med('map_resid_p99'),
        disagree_p50_med=med('disagree_p50'),
        n_jumps_filt_med=med('n_jumps_filt'), n_jumps_map_med=med('n_jumps_map'),
        arrive={f'off_{o}': dict(map_med=med(f'map_arrive_{o}'),
                                 filt_med=med(f'filt_arrive_{o}'))
                for o in args.arrive_offsets})

    ## ---- readout ----
    print_color(f"[anatomy] inpaint check: goal-frame residual {agg['goal_resid_rms_med']:.3f} "
                f"wu/beam (RMS), start-frame {agg['start_resid_rms_med']:.3f}. "
                f"~0 => the conditioning IS present in the dumped plan.",
                c=('g' if agg['goal_resid_rms_med'] < 0.05 else 'r'))
    print_color(f"[anatomy] ANCHORING, coherence removed: final frame localized ALONE -> "
                f"end err {agg['map_end_err_med']:.2f} wu, in goal cell "
                f"{agg['g2b_map'] * 100:.0f}%", c='c')
    print_color(f"[anatomy] ANCHORING, G2 definition (sequence-filtered)  -> "
                f"end err {agg['filt_end_err_med']:.2f} wu, in goal cell "
                f"{agg['g2b_filt'] * 100:.0f}%", c='c')
    gap = agg['g2b_map'] - agg['g2b_filt']
    if agg['g2b_map'] > 0.8 and gap > 0.3:
        print_color(f"[anatomy] VERDICT: the endpoint DOES carry the goal ({agg['g2b_map']*100:.0f}%); "
                    f"the filtered decode loses it ({gap*100:+.0f} pts). The composed failure is "
                    f"ROUTE COHERENCE, and G2b-on-composed-plans has been reporting a coherence "
                    f"defect as an anchoring defect.", c='y')
    elif agg['g2b_map'] < 0.4:
        print_color(f"[anatomy] VERDICT: even localized alone the final frame misses the goal "
                    f"({agg['g2b_map']*100:.0f}%) -- anchoring is genuinely broken, independent "
                    f"of the filter.", c='y')
    else:
        print_color(f"[anatomy] VERDICT: mixed -- anchoring {agg['g2b_map']*100:.0f}% alone vs "
                    f"{agg['g2b_filt']*100:.0f}% filtered. Read the profiles.", c='y')
    ## ---- the metric that neither G2b form provides ----
    a = agg['arrive']
    tab = ' | '.join(f"H-1-{o}: {a[f'off_{o}']['map_med']:.1f}/{a[f'off_{o}']['filt_med']:.1f}"
                     for o in args.arrive_offsets)
    print_color(f'[anatomy] ARRIVAL (dist to goal, wu, MAP-alone/filtered) at frames '
                f'{tab}', c='c')
    off_max = max(args.arrive_offsets)
    near = a[f'off_{off_max}']['map_med']
    print_color(
        f'[anatomy] offset 0 is the INPAINTED goal frame -- localizing it recovers the goal '
        f'by construction (same round-trip artifact as start_err ~0.4 wu), so it is NOT '
        f'evidence of routing. The honest anchoring question is the larger offsets: '
        f'{off_max} frames before the end the plan sits {near:.1f} wu from the goal.',
        c=('g' if near < 5 else 'r'))

    print_color(f"[anatomy] map-manifold residual (nearest realizable scan): p50 "
                f"{agg['map_resid_p50_med']:.3f}, p99 {agg['map_resid_p99_med']:.3f} wu/beam. "
                f"Tracks depth, but is NOT causal: the 2026-07-30 switch_map control cut this "
                f"5.6x (to below the n_comp=1 baseline) and route coherence did not move at "
                f"all. Do not build in-loop map projection on the strength of this number.",
                c='c')
    print_color(f"[anatomy] jumps: {agg['n_jumps_filt_med']:.0f} median (filtered) vs "
                f"{agg['n_jumps_map_med']:.0f} (independent MAP); MAP-vs-filter disagreement "
                f"p50 {agg['disagree_p50_med']:.2f} wu", c='c')

    ## ---- WHERE the incoherence lives (2026-07-30 item 6) ----
    ## Regions are defined by CONDITIONING, not by seams: segment 0 is
    ## start-inpainted, the last segment is goal-inpainted, and the interior was
    ## historically stgl_cond={} (--waypoints gives it an aim). Density per 100
    ## frames, so the three regions are comparable despite different spans.
    hz = args.horizon
    if H > 2 * hz:
        spans = dict(seg0=(0, hz), interior=(hz, H - hz), last=(H - hz, H))
        dens = {}
        for name, (a, b) in spans.items():
            c = sum(int(((np.asarray(r['jump_frames_filt'], int) >= a) &
                         (np.asarray(r['jump_frames_filt'], int) < b)).sum())
                    for r in rows)
            dens[name] = 100.0 * c / (len(rows) * (b - a))
        agg['jump_density_per100'] = dens
        ratio = dens['interior'] / dens['seg0'] if dens['seg0'] > 0 else float('inf')
        print_color(
            f"[anatomy] jump density per 100 frames by CONDITIONING region: "
            f"seg0 (start-inpainted) {dens['seg0']:.2f} | interior {dens['interior']:.2f} | "
            f"last (goal-inpainted) {dens['last']:.2f}  -> interior/seg0 = {ratio:.2f}x. "
            f"Baseline without waypoints was ~0.31 / 0.92 / 0.91 (3.0x).",
            c=('g' if ratio < 1.3 else 'y'))

    ## ---- PROGRESS, in maze units (the honest version of item 7) ----
    if len(prof_bfs) and np.isfinite(prof_bfs).any():
        bp = np.nanmedian(prof_bfs, axis=0)
        qs = [int(round(q * (T - 1))) for q in (0, .25, .5, .75, 1.0)]
        agg['bfs_progress_quartiles'] = [float(bp[q]) for q in qs]
        ## linearity: a plan that advances at a constant rate traces frac = 1-t.
        ## Report how much of the maze distance is covered in each half.
        half = int(round(0.5 * (T - 1)))
        first, second = 1.0 - bp[half], bp[half] - bp[-1]
        agg['bfs_frac_first_half'], agg['bfs_frac_second_half'] = float(first), float(second)
        print_color(
            f"[anatomy] PROGRESS (BFS maze distance to goal, MAP-alone, as a fraction "
            f"of the frame-0 distance) at 0/25/50/75/100% of the plan: "
            + ' -> '.join(f'{bp[q]:.2f}' for q in qs)
            + f".  First half covers {first * 100:.0f}% of the maze distance, second half "
              f"{second * 100:.0f}% (50/50 = steady pace; front-loaded second half = the "
              f"'coherent but aimless early' signature).",
            c=('g' if 0.3 <= first <= 0.7 else 'y'))

    ## seam alignment of jumps (regression check; rejected 3x already)
    if seams and stride_seam:
        allj = [f for r in rows for f in r['jump_frames_filt']]
        if allj:
            m = np.asarray(allj) % stride_seam
            win = 12
            obs = 100.0 * np.mean((m <= win) | (m >= stride_seam - win))
            null = 100.0 * (2 * win + 1) / stride_seam
            agg['seam_pct'], agg['seam_null_pct'] = float(obs), float(null)
            print_color(f'[anatomy] {len(allj)} jumps; {obs:.0f}% within +-{win} of a seam '
                        f'(mod {stride_seam}) vs {null:.0f}% uniform null -> {obs - null:+.0f} pts',
                        c='c')

    ## ---- per-frame profiles ----
    prof = dict(frame=idx.tolist(),
                map_resid_p50=np.median(prof_resid, axis=0).tolist(),
                disagree_p50=np.median(prof_disag, axis=0).tolist(),
                entropy_p50=np.median(prof_ent, axis=0).tolist(),
                goal_dist_p50=np.median(prof_gdist, axis=0).tolist(),
                bfs_gdist_frac_p50=(np.nanmedian(prof_bfs, axis=0).tolist()
                                    if len(prof_bfs) else []))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    json.dump(dict(plans=args.plans, meta=mj, lidar_config=json_safe_lidar_config(lidar_cfg),
                   agg=agg, profile=prof, rows=rows), open(args.out, 'w'), indent=1)
    print_color(f'[anatomy] wrote {args.out}', c='c')

    ## ---- figure ----
    png = args.png or (os.path.splitext(args.out)[0] + '.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys = ['bfs_gdist_frac_p50', 'goal_dist_p50', 'map_resid_p50',
                'disagree_p50', 'entropy_p50']
        labs = ['BFS maze distance\nto goal (frac of start)',
                'straight-line distance\nto goal (wu, filtered)',
                'nearest map scan\n(wu/beam)',
                'MAP vs filtered\ndisagreement (wu)',
                'belief entropy\n(nats)']
        keep = [(k, l) for k, l in zip(keys, labs) if prof.get(k)]
        fig, axes = plt.subplots(len(keep), 1, figsize=(11, 2.4 * len(keep)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (key, lab) in zip(axes, keep):
            ax.plot(prof['frame'], prof[key], lw=1.3, color='tab:blue')
            if key == 'bfs_gdist_frac_p50':
                ## steady-pace reference: a plan advancing at a constant rate
                ax.plot([prof['frame'][0], prof['frame'][-1]], [1.0, 0.0],
                        lw=1, ls='--', color='tab:red', label='steady pace')
                ax.legend(fontsize=8, loc='upper right')
            for s in seams:
                ax.axvline(s, color='0.7', ls=':', lw=1, zorder=0)
            ax.set_ylabel(lab, fontsize=9)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel('frame index along the composed plan (dotted = segment seams)')
        axes[0].set_title(f'plan anatomy: {os.path.basename(args.plans)} '
                          f'(n_comp={n_comp}, {len(rows)} plans)\n'
                          f"anchoring alone {agg['g2b_map']*100:.0f}% vs filtered "
                          f"{agg['g2b_filt']*100:.0f}%", fontsize=10)
        fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
        print_color(f'[anatomy] wrote {png}', c='c')
    except Exception as e:  # noqa: BLE001
        print_color(f'[anatomy] figure skipped: {e}', c='y')


if __name__ == '__main__':
    main()
