"""Does the plan's GENERATED body actually arrive at the goal?

Supersedes the endpoint question that `endpoint_decompose.py` asked badly. The goal
scan is INPAINTED at frame hzn-1 (stgl_sml_policy_v1.py:170) and pinned by
`apply_conditioning`, so `plan[-1]` is the goal scan by construction: comparing it to
the goal scan measures the harness, and decoding it just re-runs the goal-decode
census. Neither says whether the MODEL aimed.

The frames the model actually generates are the body. So probe the tail of the plan
EXCLUDING the pinned final frame, in two representations, and calibrate both:

  scan space  -- || plan[k] - goal_scan || for the last frames, against a calibrated
                 scale: the same distance measured from the goal scan to grid poses
                 exactly 1, 2 and 4 cells away. That converts an abstract scan
                 distance into "how many cells from the goal does this frame look".
  decoded     -- localize_single(plan[k]) distance to the true goal, per frame, no
                 sequence filter, so a filter mode hop cannot contaminate it.

Reported against the run of the whole plan (stride-sampled) so an aiming plan and a
wandering one are distinguishable: an aiming plan's tail approaches the goal in BOTH
representations; a wandering one's does not, and its final pinned frame is a cliff.

Usage:
    python -m diffuser.pl_eval.lidar_eval.plan_body_probe --n_prob 20 --n_samp 4
"""
import argparse, json, os, sys
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.append('./')

import numpy as np

TAIL = (1, 2, 3, 5, 10, 20, 40, 80)     # frames back from the END (1 == pinned frame)


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config',
                   default='config/ogb_pnt_maze/og_pntM_Gi_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512.py')
    p.add_argument('--n_prob', type=int, default=20)
    p.add_argument('--n_samp', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='logs/lidar_eval/plan_body_probe.json')
    return p.parse_args()


def _scan_scale(scanner, loc, gl_xy, gl_scan):
    """What does || scan - goal_scan || look like at 1, 2, 4 cells from the goal?

    Without this the scan distances are unreadable: 0.8 could be 'basically there'
    or 'a room away' depending on the maze."""
    from diffuser.datasets.lidar.lidar_localization import scan_distance
    mu = float(loc.grid.maze_unit)
    fxy = loc.grid.free_xy.astype(np.float64)
    d_cells = np.linalg.norm(fxy - np.asarray(gl_xy)[None], axis=1) / mu
    out = {}
    for c in (1.0, 2.0, 4.0):
        m = (d_cells > c - 0.5) & (d_cells <= c + 0.5)
        if m.any():
            out[f'{c:g}cell'] = float(np.percentile(
                scan_distance(gl_scan, loc.grid.free_scans[m])[0], 50))
    return out


def main():
    a = _parse()
    import diffuser.utils as utils
    from diffuser.pl_eval.lidar_eval.plan_decode_gate import _build_planner
    from diffuser.datasets.lidar.lidar_localization import scan_distance

    planner, args, epoch = _build_planner(a.config, hist_on=0)
    utils.set_seed(a.seed)
    probs, scanner, loc = planner.problems_dict, planner.scanner, planner.loc
    cell = float(scanner.maze_unit)

    rows, scales = [], []
    for i_ep in range(a.n_prob):
        st_xy = np.asarray(probs['start_state'][i_ep], dtype=np.float64)[:2]
        gl_xy = np.asarray(probs['goal_pos'][i_ep], dtype=np.float64)[:2]
        st_scan = scanner.scan(*map(float, st_xy), 0.0).astype(np.float32)
        gl_scan = scanner.scan(*map(float, gl_xy), 0.0).astype(np.float32)
        sc = _scan_scale(scanner, loc, gl_xy, gl_scan)
        scales.append(sc)

        g_cond = {'st_gl': np.stack([st_scan, gl_scan])[:, None, :].astype(np.float32)}
        for i_s in range(a.n_samp):
            m_out = planner.policy.gen_cond_stgl(g_cond=dict(g_cond),
                                                 b_s=args.b_size_per_prob)
            plan = np.asarray(m_out.pick_traj, dtype=np.float32)
            H = len(plan)
            r = dict(i_ep=i_ep, i_s=i_s, n_frames=H)
            for k in TAIL:
                if k > H:
                    continue
                f = plan[H - k]
                r[f'scan_d_{k}'] = float(scan_distance(f, gl_scan[None])[0][0])
                r[f'dec_err_{k}'] = float(np.linalg.norm(
                    np.asarray(loc.localize_single(f)['xy'], dtype=np.float64) - gl_xy))
            rows.append(r)
        rr = [x for x in rows if x['i_ep'] == i_ep]
        msg = '  '.join(f"-{k}:{np.mean([x[f'dec_err_{k}'] for x in rr]):5.1f}"
                        for k in (1, 2, 5, 20, 80) if f'dec_err_{k}' in rr[0])
        print(f"[body] ep {i_ep:3d}  decoded err by frames-from-end  {msg}   "
              f"(1cell scan_d ~ {sc.get('1cell', float('nan')):.2f})", flush=True)

    agg = dict(epoch=int(epoch), n=len(rows), cell=cell)
    for k in TAIL:
        ds = [r[f'dec_err_{k}'] for r in rows if f'dec_err_{k}' in r]
        ss = [r[f'scan_d_{k}'] for r in rows if f'scan_d_{k}' in r]
        if ds:
            agg[f'dec_err_p50_{k}'] = float(np.percentile(ds, 50))
            agg[f'within_1cell_{k}'] = float(np.mean(np.asarray(ds) <= cell))
            agg[f'scan_d_p50_{k}'] = float(np.percentile(ss, 50))
    agg['scale_1cell'] = float(np.median([s['1cell'] for s in scales if '1cell' in s]))
    agg['scale_2cell'] = float(np.median([s['2cell'] for s in scales if '2cell' in s]))
    agg['scale_4cell'] = float(np.median([s['4cell'] for s in scales if '4cell' in s]))

    print('\n=============== PLAN BODY PROBE ===============')
    print(f"  calibration: || scan - goal_scan || is {agg['scale_1cell']:.2f} at 1 cell, "
          f"{agg['scale_2cell']:.2f} at 2 cells, {agg['scale_4cell']:.2f} at 4 cells")
    print(f"  {'frames from end':>16} {'decoded err p50':>16} {'within 1 cell':>14} {'scan dist p50':>14}")
    for k in TAIL:
        if f'dec_err_p50_{k}' in agg:
            tag = f'-{k}' + (' (PINNED)' if k == 1 else '')
            print(f"  {tag:>16} {agg[f'dec_err_p50_{k}']:>13.2f} wu "
                  f"{agg[f'within_1cell_{k}']*100:>12.1f}% {agg[f'scan_d_p50_{k}']:>14.2f}")
    gen = agg.get('dec_err_p50_2', float('inf'))
    if gen <= cell:
        print('  -> the last GENERATED frame decodes at the goal: the model aims, and')
        print('     G2b\'s endpoint error is downstream of it.')
    elif gen <= 3 * cell:
        print('  -> the body arrives near but not at the goal; part aim, part decode.')
    else:
        print('  -> the generated body does NOT arrive. The pinned final frame is a')
        print('     cliff, i.e. the plan is not anchored and G2b is real.')
    print('===============================================')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(dict(summary=agg, per_sample=rows), open(a.out, 'w'), indent=2)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
