"""Is the plan decoder broken, and if so where? -- CPU only, exact ground truth.

There is no ground truth for a GENERATED plan, so this bench does not use one.
It isolates the DECODER from the PLANNER: render scans with the real scanner from
real dataset trajectories, where every frame's (x, y) is known exactly, assemble
them with the real composer geometry, and score each decoder against that truth.

The experiment rests on one structural fact about `Traj_Blender.blend_traj_lists`:
OUTSIDE the overlap regions the blended plan and a hard-handover plan are
BIT-IDENTICAL (`cnt_v == 1`; the non-overlap frames are copied verbatim from their
owning segment). The two differ only inside the 392 seam frames of 888.

So we score on NON-SEAM frames only. There both decoders see the same valid,
on-manifold, ground-truth scans, and any difference in error is purely the forward
filter's memory having been corrupted by what it ingested in the seams.

  H0 (falsifier)  raise the segment disagreement from zero. If `filter_blend`'s
                  error tracks `filter_hard` across the whole dose-response, the
                  scan-space-blending hypothesis is FALSE.

Usage:
    python -m diffuser.pl_eval.lidar_eval.decode_bench
    python -m diffuser.pl_eval.lidar_eval.decode_bench --n_paths 200 --jobs 8
"""
import argparse, json, os, sys, time
sys.path.append('./')

import numpy as np

from diffuser.datasets.lidar import (
    make_scanner_for_env, load_or_build_grid, GridLocalizer, scan_distance)
from diffuser.guides.comp.traj_blender import blend_2_np_trajs_23d

ENV = 'pointmaze-giant-stitch-v0'
DSET = os.path.expanduser('~/.ogbench/data/pointmaze-giant-stitch-v0.npz')
LIDAR_CONFIG = dict(n_beams=30, max_range=12.0, fov=2.0 * np.pi,
                    fov_centered=False, step=0.1)


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n_paths', type=int, default=200)
    p.add_argument('--n_comp', type=int, default=8)
    p.add_argument('--hzn', type=int, default=160)
    p.add_argument('--len_ovlp', type=int, default=56)
    p.add_argument('--blend_type', default='exp')
    p.add_argument('--exp_beta', type=float, default=2.0)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--consumed', type=int, default=250,
                   help='the window the executor actually reaches (cursor_max p50 175-264)')
    p.add_argument('--join_tol', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='logs/lidar_eval/decode_bench.json')
    return p.parse_args()


# ----------------------------------------------------------------- geometry
def seam_spans(n_comp, hzn, len_ovlp):
    """The overlap (blended) frame spans, exactly as `blend_traj_lists` writes them."""
    S = hzn - len_ovlp
    return [((i + 1) * S, (i + 1) * S + len_ovlp) for i in range(n_comp - 1)]


def blend_weights(len_ovlp, blend_type='exp', beta=2.0):
    """The weight profile `blend_2_np_trajs_23d` applies to traj_1 (w) / traj_2 (1-w).

    Kept as its own function so the hard handover thresholds THE SAME curve the
    blend uses, instead of a hard-coded midpoint that silently disagrees with it.
    """
    t = np.arange(len_ovlp, dtype=np.float64)
    t_end = len_ovlp - 1
    if blend_type in ('exponential', 'exp'):
        e = -beta * t / t_end
        return (np.exp(e) - np.exp(-beta)) / (1 - np.exp(-beta))
    if blend_type == 'cosine':
        return 0.5 * (1 + np.cos(np.pi * t / t_end))
    if blend_type == 'linear':
        return 1 - t / t_end
    if blend_type == 'smoothstep':
        x = t / t_end
        return 1 - (3 * x ** 2 - 2 * x ** 3)
    raise ValueError(blend_type)


def assemble(segs, n_comp, hzn, len_ovlp, blend_type, beta, mode):
    """(n_comp, hzn, D) segments -> (tot_hzn, D), mirroring `blend_traj_lists`.

    mode 'blend': the real elementwise blend. mode 'hard': the same weights,
    thresholded -- each overlap frame is taken whole from its owning segment.
    """
    S = hzn - len_ovlp
    tot = n_comp * hzn - (n_comp - 1) * len_ovlp
    D = segs[0].shape[1]
    out = np.zeros((tot, D), dtype=np.float32)
    ## non-overlap parts, copied verbatim -- identical in both modes
    out[0:S] = segs[0][:S]
    for i in range(1, n_comp - 1):
        a = hzn + (i - 1) * S
        out[a:a + (hzn - 2 * len_ovlp)] = segs[i][len_ovlp:hzn - len_ovlp]
    a = hzn + (n_comp - 2) * S
    out[a:a + S] = segs[n_comp - 1][len_ovlp:]
    ## overlaps
    w = blend_weights(len_ovlp, blend_type, beta)
    for i, (lo, hi) in enumerate(seam_spans(n_comp, hzn, len_ovlp)):
        end_a = segs[i][hzn - len_ovlp:]            # segment i's trailing overlap
        st_b = segs[i + 1][:len_ovlp]               # segment i+1's leading overlap
        if mode == 'blend':
            out[lo:hi] = blend_2_np_trajs_23d(end_a, st_b, blend_type, beta)
        elif mode == 'hard':
            take_a = (w >= 0.5)[:, None]
            out[lo:hi] = np.where(take_a, end_a, st_b)
        else:
            raise ValueError(mode)
    return out


# ----------------------------------------------------------------- truth paths
def build_paths(n_paths, n_frames, join_tol, rng):
    """Chain real episodes into ground-truth paths of `n_frames` (x, y)."""
    d = np.load(DSET)
    xy = d['observations'].astype(np.float64)
    term = d['terminals']
    ends = np.flatnonzero(term)
    assert len(ends), 'no episode boundaries'
    ep_len = ends[0] + 1
    n_ep = len(ends)
    xy = xy[:n_ep * ep_len].reshape(n_ep, ep_len, 2)
    starts = xy[:, 0, :]
    paths, tries = [], 0
    while len(paths) < n_paths and tries < n_paths * 40:
        tries += 1
        used = {int(rng.integers(n_ep))}
        chain = [xy[list(used)[0]]]
        ok = True
        while sum(len(c) for c in chain) < n_frames:
            cur_end = chain[-1][-1]
            dist = np.linalg.norm(starts - cur_end, axis=1)
            dist[list(used)] = np.inf
            k = int(np.argmin(dist))
            if dist[k] > join_tol:
                ok = False
                break
            used.add(k)
            chain.append(xy[k])
        if not ok:
            continue
        paths.append(np.concatenate(chain, 0)[:n_frames])
    assert len(paths) >= 1, 'could not chain any path -- loosen --join_tol'
    return np.stack(paths)                              # (P, n_frames, 2)


def perturb(path_xy, lo, hi, phase, lateral, scanner, rng):
    """Candidate B's poses for window [lo, hi): shifted in pace and offset laterally."""
    n = hi - lo
    idx = np.clip(np.arange(lo, hi) + phase, 0, len(path_xy) - 1)
    p = path_xy[idx].copy()
    if lateral > 0:
        dv = np.gradient(p, axis=0)
        nrm = np.linalg.norm(dv, axis=1, keepdims=True)
        perp = np.stack([-dv[:, 1], dv[:, 0]], 1) / np.maximum(nrm, 1e-9)
        cand = p + lateral * perp * (1 if rng.random() < 0.5 else -1)
        i, j = scanner.world_to_cell(cand[:, 0], cand[:, 1])
        free = ~scanner._is_wall(i, j)                   # keep the offset only in free space
        p[free] = cand[free]
    assert len(p) == n
    return p


# ----------------------------------------------------------------- decoders
def decode(mode, scans, loc, stride, motion_sigma, floor, seed_belief):
    """-> (n_dec, 2) MAP xy at the strided frames, plus the strided indices."""
    H = len(scans)
    idx = np.unique(np.concatenate([np.arange(0, H, stride), [H - 1]]))
    sub = scans[idx].astype(np.float32)
    if mode == 'argmin':
        d = scan_distance(sub, loc.grid.free_scans)      # (n_dec, M)
        return loc.grid.free_xy[np.argmin(d, axis=1)].astype(np.float64), idx
    kw = {}
    if floor:
        kw['uniform_floor'] = floor
    r = loc.localize_sequence(sub, motion_sigma_cells=motion_sigma, theta_blur=0,
                              init_belief=seed_belief, **kw)
    return np.asarray(r['map_xy'], dtype=np.float64), idx


def score(est_xy, idx, truth_xy, non_seam, cell, consumed):
    """Errors on NON-SEAM frames, where blend and hard carry identical scans."""
    keep = non_seam[idx]
    e = np.linalg.norm(est_xy - truth_xy[idx], axis=1)
    ek = e[keep]
    kc = keep & (idx < consumed)
    ec = e[kc]
    bad = e > cell
    if bad.any():
        first = int(idx[int(np.argmax(bad))])
        tail = bad[int(np.argmax(bad)):]
        lock = bool(tail.all())
    else:
        first, lock = -1, False
    return dict(
        err_p50=float(np.percentile(ek, 50)) if len(ek) else float('nan'),
        err_p90=float(np.percentile(ek, 90)) if len(ek) else float('nan'),
        within_1cell=float((ek <= cell).mean()) if len(ek) else float('nan'),
        cons_err_p50=float(np.percentile(ec, 50)) if len(ec) else float('nan'),
        cons_within_1cell=float((ec <= cell).mean()) if len(ec) else float('nan'),
        end_err=float(e[-1]), first_bad_frame=first, lock_in=lock)


def manifold_residual(scans, loc, non_seam, stride=8):
    """min_m ||scan_t - free_scans[m]||, inside vs outside seams. No truth needed."""
    idx = np.arange(0, len(scans), stride)
    d = scan_distance(scans[idx].astype(np.float32), loc.grid.free_scans).min(axis=1)
    ins, out = ~non_seam[idx], non_seam[idx]
    return (float(np.percentile(d[ins], 50)) if ins.any() else float('nan'),
            float(np.percentile(d[out], 50)) if out.any() else float('nan'))


# ----------------------------------------------------------------- driver
## Dose-response. Segment i+1's LEADING overlap is the only thing perturbed, so
## every non-seam frame still carries the exact truth scan for both assemblies --
## which is what makes the non-seam comparison fair.
LEVELS = [
    ## name        phase  lateral  seam_noise  global_noise
    ('none',           0,     0.0,       0.00,         0.00),
    ('phase8',         8,     0.0,       0.00,         0.00),
    ('phase32',       32,     0.0,       0.00,         0.00),
    ('lat2',           0,     2.0,       0.00,         0.00),
    ('lat4',           0,     4.0,       0.00,         0.00),
    ('lat8',           0,     8.0,       0.00,         0.00),
    ('seamnoise1',     0,     0.0,       1.00,         0.00),
    ('seamnoise4',     0,     0.0,       4.00,         0.00),
    ('seam_worst',    32,     8.0,       2.00,         0.00),
    ## global: the WHOLE plan is off-manifold, as a model's reconstruction would be
    ('gnoise0.25',     0,     0.0,       0.00,         0.25),
    ('gnoise0.5',      0,     0.0,       0.00,         0.50),
    ('gnoise1.0',      0,     0.0,       0.00,         1.00),
    ('gnoise2.0',      0,     0.0,       0.00,         2.00),
    ('gnoise4.0',      0,     0.0,       0.00,         4.00),
]

_G = {}


def _init(a):
    _G['a'] = a
    _G['scanner'] = make_scanner_for_env(ENV, LIDAR_CONFIG)
    _G['grid'] = load_or_build_grid(ENV, LIDAR_CONFIG, xy_step=0.5, n_theta=1,
                                    verbose=False)
    _G['loc'] = GridLocalizer(_G['grid'], sigma=1.0)


def run_path(args):
    i_p, path_xy = args
    a, scanner, loc = _G['a'], _G['scanner'], _G['loc']
    n_comp, hzn, ovlp = a.n_comp, a.hzn, a.len_ovlp
    S = hzn - ovlp
    tot = n_comp * hzn - (n_comp - 1) * ovlp
    cell = float(scanner.maze_unit)
    rng = np.random.default_rng(a.seed * 100003 + i_p)

    non_seam = np.ones(tot, dtype=bool)
    for lo, hi in seam_spans(n_comp, hzn, ovlp):
        non_seam[lo:hi] = False

    truth_scans = scanner.scan_batch(path_xy[:tot], np.zeros(tot))
    rows = []
    for name, phase, lateral, noise, gnoise in LEVELS:
        segs = [truth_scans[i * S:i * S + hzn].copy() for i in range(n_comp)]
        for i, (lo, hi) in enumerate(seam_spans(n_comp, hzn, ovlp)):
            p = perturb(path_xy, lo, hi, phase, lateral, scanner, rng)
            s = scanner.scan_batch(p, np.zeros(len(p)))
            if noise:
                s = np.clip(s + rng.normal(0, noise, s.shape),
                            0, scanner.max_range).astype(np.float32)
            segs[i + 1][:ovlp] = s                      # seg i+1's LEADING overlap

        plans = {m: assemble(segs, n_comp, hzn, ovlp, a.blend_type, a.exp_beta, m)
                 for m in ('blend', 'hard')}
        if gnoise:
            ## the SAME perturbation on both assemblies, so the comparison stays paired
            gn = rng.normal(0, gnoise, plans['blend'].shape)
            for m in plans:
                plans[m] = np.clip(plans[m] + gn, 0, scanner.max_range).astype(np.float32)
        if name == 'none':                              # H0 part 1, as an assertion
            assert np.array_equal(plans['blend'], plans['hard']), 'zero-dose mismatch'
            assert np.array_equal(plans['blend'], truth_scans[:tot]), 'zero-dose != truth'

        r_in, r_out = manifold_residual(plans['blend'], loc, non_seam)
        h_in, h_out = manifold_residual(plans['hard'], loc, non_seam)

        for dec, plan_key in (('filter_blend', 'blend'), ('filter_hard', 'hard'),
                              ('argmin', 'blend')):
            est, idx = decode('argmin' if dec == 'argmin' else 'filter',
                              plans[plan_key], loc, a.stride,
                              motion_sigma=1.0, floor=0.0, seed_belief=None)
            rows.append(dict(i_path=i_p, level=name, decoder=dec,
                             **score(est, idx, path_xy[:tot], non_seam, cell, a.consumed)))
        rows[-3]['seam_resid_in'], rows[-3]['seam_resid_out'] = r_in, r_out
        rows[-2]['seam_resid_in'], rows[-2]['seam_resid_out'] = h_in, h_out
    return rows


def main():
    a = _parse()
    n_comp, hzn, ovlp = a.n_comp, a.hzn, a.len_ovlp
    tot = n_comp * hzn - (n_comp - 1) * ovlp
    rng = np.random.default_rng(a.seed)
    need = tot + max(p for _, p, *_ in LEVELS) + 2

    t0 = time.time()
    print(f'[bench] composer geometry: n_comp={n_comp} hzn={hzn} ovlp={ovlp} -> tot={tot}')
    print(f'[bench] seams {seam_spans(n_comp, hzn, ovlp)}')
    n_seam = sum(hi - lo for lo, hi in seam_spans(n_comp, hzn, ovlp))
    print(f'[bench] blended frames {n_seam}/{tot} = {100*n_seam/tot:.1f}%')
    w = blend_weights(ovlp, a.blend_type, a.exp_beta)
    print(f'[bench] handover at t_rel={int(np.argmax(w < 0.5))} '
          f'(w crosses 0.5); global frame {(1)*(hzn-ovlp) + int(np.argmax(w < 0.5))} for seam 0')

    paths = build_paths(a.n_paths, need, a.join_tol, rng)
    print(f'[bench] built {len(paths)} ground-truth paths of {need} frames '
          f'({time.time()-t0:.0f}s)')

    _init(a)
    rows = []
    for i_p in range(len(paths)):
        rows += run_path((i_p, paths[i_p]))
        if (i_p + 1) % 10 == 0:
            print(f'  {i_p+1}/{len(paths)}  ({time.time()-t0:.0f}s)', flush=True)

    report(rows, a)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(dict(config=vars(a), rows=rows), open(a.out, 'w'), indent=1)
    print(f'\nwrote {a.out}')
    return 0


def report(rows, a):
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['level'], r['decoder'])].append(r)
    q = lambda v: float(np.percentile(v, 50)) if len(v) else float('nan')

    print('\n================ NON-SEAM FRAMES (identical scans in both assemblies) ================')
    print(f"{'level':>10} {'decoder':>13} {'err_p50':>8} {'err_p90':>8} {'<=1cell':>8} "
          f"{'cons_p50':>9} {'cons<=1c':>9} {'lock_in':>8} {'end_err':>8}")
    for name, *_ in LEVELS:
        for dec in ('filter_blend', 'filter_hard', 'argmin'):
            r = g[(name, dec)]
            if not r:
                continue
            print(f"{name:>10} {dec:>13} {q([x['err_p50'] for x in r]):>8.2f} "
                  f"{q([x['err_p90'] for x in r]):>8.2f} "
                  f"{100*np.mean([x['within_1cell'] for x in r]):>7.1f}% "
                  f"{q([x['cons_err_p50'] for x in r]):>9.2f} "
                  f"{100*np.mean([x['cons_within_1cell'] for x in r]):>8.1f}% "
                  f"{100*np.mean([x['lock_in'] for x in r]):>7.1f}% "
                  f"{q([x['end_err'] for x in r]):>8.2f}")
        print()

    print('================ H1  manifold residual, inside vs outside seams ================')
    print(f"{'level':>10} {'assembly':>10} {'inside':>8} {'outside':>8} {'ratio':>7}")
    for name, *_ in LEVELS:
        for dec, lab in (('filter_blend', 'blend'), ('filter_hard', 'hard')):
            r = [x for x in g[(name, dec)] if 'seam_resid_in' in x]
            if not r:
                continue
            i, o = q([x['seam_resid_in'] for x in r]), q([x['seam_resid_out'] for x in r])
            print(f"{name:>10} {lab:>10} {i:>8.3f} {o:>8.3f} {i/max(o,1e-9):>7.2f}x")
        print()

    ## ---- H0: does the seam hypothesis survive? -------------------------------
    print('================ H0 VERDICT ================')
    disc = []
    for name, *_ in LEVELS:
        b = g[(name, 'filter_blend')]
        h = g[(name, 'filter_hard')]
        if not b or not h:
            continue
        pair = [(x['cons_err_p50'], y['cons_err_p50']) for x, y in zip(b, h)]
        better = sum(1 for u, v in pair if v < u - 1e-9)
        worse = sum(1 for u, v in pair if u < v - 1e-9)
        disc.append((name, q([x['cons_err_p50'] for x in b]),
                     q([x['cons_err_p50'] for x in h]), better, worse, len(pair)))
    print(f"{'level':>10} {'blend':>8} {'hard':>8} {'hard better':>12} {'worse':>7} {'n':>5}")
    for name, ub, uh, bt, ws, n in disc:
        print(f"{name:>10} {ub:>8.2f} {uh:>8.2f} {bt:>12} {ws:>7} {n:>5}")
    nz = [d for d in disc if d[0] != 'none']
    sep = [d for d in nz if d[1] > d[2] + 0.5]
    if sep:
        print(f"\n  -> filter_hard beats filter_blend at {len(sep)}/{len(nz)} nonzero doses.")
        print("     The seam hypothesis SURVIVES. Proceed to Phase 1-2.")
    else:
        print("\n  -> filter_blend tracks filter_hard across the dose-response.")
        print("     The seam hypothesis is REFUTED. Skip Phase 2; go to floor + argmin.")


if __name__ == '__main__':
    sys.exit(main())
