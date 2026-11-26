"""Dump RAW generated scan plans to an npz, so every decoder sweep after this is CPU.

`decode_bench` established that the forward filter is sound on realizable inputs and
that what breaks it is a mismatch between the plan's implied inter-frame displacement
and the filter's motion prior (sigma=1.0 lattice = 0.5 wu per decoded step). That
prediction is about REAL generated plans, so dump some once and stop paying for GPU.

Saves, per generated plan: the (tot_hzn, n_beams) scan plan, the problem's true start
and goal xy, and the filter decode the executor would have received.

ALSO saves the PRE-BLEND per-segment trajectories (`segs`, (n_plans, n_comp, hzn, D))
of the picked candidate, which the blended plan cannot be inverted back into. Without
them the seam counterfactual -- what would jump_frac be under a hard handover instead
of an elementwise blend -- is unanswerable on CPU. `segs` is validated on the way out:
`decode_bench.assemble(segs, mode='blend')` is compared against the shipped blender's
own `pick_traj` for every plan, and the max residual is recorded in the npz as
`blend_residual`. If that is not ~0 the segments are the wrong ones and every
downstream counterfactual is void.

Usage:
    python -m diffuser.pl_eval.lidar_eval.dump_plans --per_block 4 --n_samp 2
"""
import argparse, json, os, sys, time
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.append('./')

import numpy as np


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config',
                   default='config/ogb_pnt_maze/og_pntM_Gi_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512.py')
    p.add_argument('--per_block', type=int, default=4, help='problems sampled per block')
    p.add_argument('--n_samp', type=int, default=2, help='plans per problem')
    p.add_argument('--hist', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='logs/lidar_eval/plans_dump.npz')
    p.add_argument('--no_segs', action='store_true',
                   help='skip the pre-blend segment dump (and its blend check)')
    return p.parse_args()


def main():
    a = _parse()
    import diffuser.utils as utils
    from diffuser.pl_eval.lidar_eval.plan_decode_gate import _build_planner, _val_history
    from diffuser.pl_eval.lidar_eval.decode_bench import assemble

    planner, args, epoch = _build_planner(a.config, a.hist)
    utils.set_seed(a.seed)
    probs = planner.problems_dict
    scanner = planner.scanner

    ## all 5 blocks of 20, evenly sampled -- house rule #1
    ep_idxs = [b * 20 + int(round(k * 19 / max(a.per_block - 1, 1)))
               for b in range(5) for k in range(a.per_block)]
    print(f'[dump] epoch={epoch} hist={a.hist} problems={ep_idxs}')

    plans, st_xy, gl_xy, ep_id, filt_xy = [], [], [], [], []
    segs_acc, blend_res = [], []
    pol = planner.policy
    blder = pol.tj_blder
    n_comp, hzn, len_ovlp = int(pol.n_comp), int(blder.hzn), int(blder.len_ovlp)
    blend_type, exp_beta = str(blder.blend_type), float(blder.exp_beta)
    ## `pick_type='rand'` draws from np.random inside the policy and cannot be
    ## reconstructed here, which would silently pair segments with a DIFFERENT
    ## plan's blend. Refuse rather than dump a mismatched pair.
    if not a.no_segs:
        assert pol.pick_type == 'first', (
            f'segment dump needs pick_type="first" to identify the picked '
            f'candidate; got {pol.pick_type!r}. Rerun with --no_segs.')
    print(f'[dump] composer: n_comp={n_comp} hzn={hzn} len_ovlp={len_ovlp} '
          f'blend={blend_type} beta={exp_beta} top_n={pol.top_n}')
    t0 = time.time()
    for i_ep in ep_idxs:
        s = np.asarray(probs['start_state'][i_ep], dtype=np.float64)[:2]
        g = np.asarray(probs['goal_pos'][i_ep], dtype=np.float64)[:2]
        st_scan = scanner.scan(*map(float, s), 0.0).astype(np.float32)
        gl_scan = scanner.scan(*map(float, g), 0.0).astype(np.float32)
        g_cond = {'st_gl': np.stack([st_scan, gl_scan])[:, None, :].astype(np.float32)}
        ## `_gen_cond_stgl` overwrites g_cond['hist'] from the rollout deque, which is
        ## EMPTY offline -- the documented gate bug. Seed it exactly as the gate does.
        if a.hist:
            hw = _val_history(planner, s, planner.hist_len)
            planner._hist.clear()
            planner._hist.extend(hw)
        for _ in range(a.n_samp):
            m_out = planner.policy.gen_cond_stgl(g_cond=dict(g_cond),
                                                 b_s=args.b_size_per_prob)
            plans.append(np.asarray(m_out.pick_traj, dtype=np.float32))
            if not a.no_segs:
                ## Reproduce the policy's own top-n ranking rather than reaching into
                ## it: `trajs_list_np_un` is the FULL candidate set, and the plan that
                ## was blended is candidate s_idxs[0] of each segment (pick_type
                ## 'first' + top_n slice). Verified below by re-blending.
                tl = m_out.trajs_list_np_un
                s_idxs, _ = utils.compute_ovlp_dist(
                    tl, pol.diffusion_model.len_ovlp_cd)
                sg = np.stack([np.asarray(tl[i_c][s_idxs[0]], dtype=np.float32)
                               for i_c in range(n_comp)])
                re_bl = assemble(sg, n_comp, hzn, len_ovlp,
                                 blend_type, exp_beta, mode='blend')
                res = float(np.max(np.abs(
                    re_bl.astype(np.float64) - plans[-1].astype(np.float64))))
                segs_acc.append(sg)
                blend_res.append(res)
                ## FAIL FAST. The aggregate assert below only fires after every plan
                ## has been sampled, so a reconstruction bug would burn the whole GPU
                ## run before saying so -- the same "verdict only at the end" trap
                ## that cost a benchmark run four hours. Check the first plan and stop.
                if len(blend_res) == 1:
                    print(f'[dump] first-plan blend-identity residual {res:.3e}',
                          flush=True)
                    assert res < 1e-3, (
                        f'assemble(mode="blend") does not reproduce the shipped '
                        f'pick_traj on the very first plan (residual {res:.3e}). The '
                        f'reconstructed segments are not the ones that were blended; '
                        f'stopping now rather than sampling the rest.')
            filt_xy.append(np.asarray(planner._plan_xy, dtype=np.float32)
                           if planner._plan_xy is not None
                           else np.full((len(plans[-1]), 2), np.nan, np.float32))
            st_xy.append(s); gl_xy.append(g); ep_id.append(i_ep)
        print(f'  ep {i_ep:3d} done ({time.time()-t0:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    extra = {}
    if not a.no_segs:
        res = np.asarray(blend_res, dtype=np.float64)
        print(f'[dump] blend-identity check over {len(res)} plans: '
              f'max residual {res.max():.3e}  (p50 {np.median(res):.3e})')
        ## float32 round-trip through the two code paths, nothing more.
        assert res.max() < 1e-3, (
            f'assemble(mode="blend") does not reproduce the shipped pick_traj '
            f'(max residual {res.max():.3e}) -- the dumped segments are NOT the ones '
            f'that were blended, so no seam counterfactual may be run on them.')
        extra = dict(segs=np.stack(segs_acc), blend_residual=res,
                     n_comp=n_comp, hzn=hzn, len_ovlp=len_ovlp,
                     blend_type=blend_type, exp_beta=exp_beta)
    np.savez_compressed(
        a.out, plans=np.stack(plans), filter_xy=np.stack(filt_xy),
        start_xy=np.stack(st_xy), goal_xy=np.stack(gl_xy),
        ep_id=np.asarray(ep_id), epoch=epoch, hist=a.hist,
        n_beams=scanner.n_beams, max_range=scanner.max_range,
        maze_unit=scanner.maze_unit,
        ## the planner's OWN lidar_config, so downstream CPU tools rebuild the
        ## same scanner/grid instead of re-deriving it from a hardcoded default
        lidar_config=json.dumps({k: (float(v) if isinstance(v, (int, float))
                                     else v)
                                 for k, v in dict(planner.lidar_config).items()
                                 if not isinstance(v, (dict, list))}),
        **extra)
    print(f'\n[dump] {len(plans)} plans of {plans[0].shape} -> {a.out} '
          f'({os.path.getsize(a.out)/1e6:.1f} MB, {time.time()-t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
