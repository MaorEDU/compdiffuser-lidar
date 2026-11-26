"""Is G2b measuring the PLAN, or the DECODER?

G2b on the giant checkpoint: 25.6% of plan endpoints within a cell of the goal,
median endpoint error 35.45 wu. But the goal-decode census showed the goal SCAN
localizes to within a cell 85% of the time, so the target is visible. Either the
planner is not aiming, or the thing that turns a scan plan into (x, y) is losing it.

G2's endpoint comes from `_decode_plan_xy`, the forward FILTER over all 888 frames.
The gate's own docstring warns a filter's per-step argmax hops between posterior
modes -- a filter that loses track at frame 600 reports a wrong endpoint even when
the plan's last scan sits exactly on the goal. So measure the endpoint three ways on
the SAME generated plans, from the cheapest interpretation to the most expensive:

  (c) SCAN SPACE, no decoding at all -- || plan[-1] - goal_scan ||, against the
      calibration floor of what the goal scan scores versus the grid. If this is at
      the floor, the model put the right scan at the last frame and everything
      downstream is decode.
  (b) SINGLE-FRAME decode -- localize_single(plan[-1]), no sequence filter.
  (a) FILTER decode -- planner._decode_plan_xy(plan)[-1], exactly what G2 reports.

(c) small + (b) near the goal + (a) far  =>  the model aims; the FILTER is the bug.
all three far                            =>  G2b is real; the planner does not anchor.

Also reports where filter and per-frame decodes diverge along the plan, which is
where a mode hop would show up.

Usage:
    python -m diffuser.pl_eval.lidar_eval.endpoint_decompose --n_prob 20 --n_samp 4
"""
import argparse, json, os, sys
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.append('./')

import numpy as np


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config',
                   default='config/ogb_pnt_maze/og_pntM_Gi_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512.py')
    p.add_argument('--n_prob', type=int, default=20)
    p.add_argument('--decode', default=None, choices=['argmin', 'filter'],
                   help='override the plan decoder; default = the planner default. '
                        'BALLV2 section 5: never compare numbers across decoders.')
    p.add_argument('--n_samp', type=int, default=4)
    p.add_argument('--stride', type=int, default=8, help='per-frame decode stride')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='logs/lidar_eval/endpoint_decompose.json')
    return p.parse_args()


def main():
    a = _parse()
    import diffuser.utils as utils
    from diffuser.pl_eval.lidar_eval.plan_decode_gate import _build_planner
    from diffuser.datasets.lidar.lidar_localization import scan_distance

    planner, args, epoch = _build_planner(a.config, hist_on=0)
    if a.decode:
        planner._decode_mode = a.decode
    print(f'[gate] plan decoder = {planner._decode_mode}')
    utils.set_seed(a.seed)
    probs, scanner, loc = planner.problems_dict, planner.scanner, planner.loc
    cell = float(scanner.maze_unit)
    grid = loc.grid

    rows = []
    for i_ep in range(a.n_prob):
        st_xy = np.asarray(probs['start_state'][i_ep], dtype=np.float64)[:2]
        gl_xy = np.asarray(probs['goal_pos'][i_ep], dtype=np.float64)[:2]
        st_scan = scanner.scan(*map(float, st_xy), 0.0).astype(np.float32)
        gl_scan = scanner.scan(*map(float, gl_xy), 0.0).astype(np.float32)

        ## calibration floor: how well does the goal scan match the grid at all?
        d_grid = scan_distance(gl_scan, grid.free_scans)[0]
        floor = float(d_grid.min())

        g_cond = {'st_gl': np.stack([st_scan, gl_scan])[:, None, :].astype(np.float32)}
        for i_s in range(a.n_samp):
            m_out = planner.policy.gen_cond_stgl(g_cond=dict(g_cond),
                                                 b_s=args.b_size_per_prob)
            plan = np.asarray(m_out.pick_traj, dtype=np.float32)     # (H, n_beams)
            xy_f = planner._plan_xy                                   # filter decode
            if xy_f is None:
                rows.append(dict(i_ep=i_ep, i_s=i_s, decode_failed=True)); continue

            ## (c) scan space, no decoding
            sd_end = float(scan_distance(plan[-1], gl_scan[None])[0][0])
            ## (b) single-frame decode of the last plan frame
            xy_b = np.asarray(loc.localize_single(plan[-1])['xy'], dtype=np.float64)
            ## (a) filter decode endpoint -- what G2 reports
            xy_a = np.asarray(xy_f[-1], dtype=np.float64)

            ## where do filter and per-frame decodes part company?
            idx = np.arange(0, len(plan), a.stride)
            xy_pf = np.stack([loc.localize_single(plan[k])['xy'] for k in idx])
            xy_ff = np.asarray(xy_f)[np.minimum(idx, len(xy_f) - 1)]
            gap = np.linalg.norm(xy_pf - xy_ff, axis=1)
            first_div = int(idx[int(np.argmax(gap > cell))]) if (gap > cell).any() else -1

            rows.append(dict(
                i_ep=i_ep, i_s=i_s,
                err_filter=float(np.linalg.norm(xy_a - gl_xy)),
                err_single=float(np.linalg.norm(xy_b - gl_xy)),
                scan_dist_end=sd_end, scan_floor=floor,
                gap_p50=float(np.percentile(gap, 50)),
                first_divergence_frame=first_div, n_frames=int(len(plan)),
            ))
        r = [x for x in rows if x['i_ep'] == i_ep and not x.get('decode_failed')]
        if r:
            print(f"[dec] ep {i_ep:3d}  filter={np.mean([x['err_filter'] for x in r]):6.2f} wu  "
                  f"single={np.mean([x['err_single'] for x in r]):6.2f} wu  "
                  f"scan_end={np.mean([x['scan_dist_end'] for x in r]):5.2f} "
                  f"(floor {r[0]['scan_floor']:.2f})  "
                  f"div@{np.mean([x['first_divergence_frame'] for x in r]):.0f}", flush=True)

    ok = [r for r in rows if not r.get('decode_failed')]
    ef = np.array([r['err_filter'] for r in ok])
    es = np.array([r['err_single'] for r in ok])
    sd = np.array([r['scan_dist_end'] for r in ok])
    fl = np.array([r['scan_floor'] for r in ok])
    agg = dict(
        epoch=int(epoch), decode=str(planner._decode_mode), n=len(ok),
        err_filter_p50=float(np.percentile(ef, 50)), filter_within_1cell=float((ef <= cell).mean()),
        err_single_p50=float(np.percentile(es, 50)), single_within_1cell=float((es <= cell).mean()),
        scan_dist_end_p50=float(np.percentile(sd, 50)), scan_floor_p50=float(np.percentile(fl, 50)),
        gap_p50=float(np.percentile([r['gap_p50'] for r in ok], 50)),
        first_div_p50=float(np.percentile([r['first_divergence_frame'] for r in ok], 50)),
    )
    print('\n=============== ENDPOINT DECOMPOSITION ===============')
    print(f"  (a) FILTER decode endpoint : p50 {agg['err_filter_p50']:6.2f} wu   "
          f"within 1 cell {agg['filter_within_1cell']*100:5.1f}%   <- what G2b reports")
    print(f"  (b) SINGLE-frame decode    : p50 {agg['err_single_p50']:6.2f} wu   "
          f"within 1 cell {agg['single_within_1cell']*100:5.1f}%")
    print(f"  (c) SCAN-space distance    : p50 {agg['scan_dist_end_p50']:5.2f}   "
          f"(goal-scan grid floor {agg['scan_floor_p50']:.2f})")
    print(f"  filter vs per-frame gap p50 {agg['gap_p50']:.2f} wu; "
          f"median first divergence at frame {agg['first_div_p50']:.0f} of {ok[0]['n_frames']}")
    if agg['single_within_1cell'] > 0.6 and agg['filter_within_1cell'] < 0.4:
        print('  -> the model AIMS; the sequence FILTER loses it. G2b is an instrument')
        print('     number, and the fix is the decoder, not the planner.')
    elif agg['scan_dist_end_p50'] <= 2.0 * agg['scan_floor_p50']:
        print('  -> the last plan SCAN matches the goal scan, but neither decode')
        print('     recovers it: goal-scan ambiguity, not planner aim.')
    else:
        print('  -> the plan does not end at the goal in ANY representation.')
        print('     G2b is real: the planner is not anchoring. Planner work.')
    print('======================================================')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(dict(summary=agg, per_sample=rows), open(a.out, 'w'), indent=2)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
