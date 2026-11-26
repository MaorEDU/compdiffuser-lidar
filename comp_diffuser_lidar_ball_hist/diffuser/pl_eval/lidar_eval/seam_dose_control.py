"""Does `seam_handover_test` respond to the thing it claims to measure? -- CPU only.

`seam_handover_test` compares three composers on REAL generated segments, where there
is no ground truth. That makes it vulnerable to the same failure as any null result: if
the instrument simply cannot separate the arms, every arm scores the same and the
conclusion "the composer is not the problem" would be unfalsifiable.

This is its positive control. Ground-truth dataset paths are cut into the same 8x160/56
segment geometry, then consecutive segments are DISPLACED against each other by a known
dose in world units -- so the two segments sharing an overlap genuinely disagree about
where it is, which is exactly the condition the composer has to absorb on generated
plans. Every arm is then scored with G2a's expressions through the shipped decoder.

At dose 0 the segments are bit-identical in the overlap and all three arms must
coincide (`assemble`'s own zero-dose identity). Above it they must separate, and the
separation calibrates the real result: it says which segment disagreement reproduces
the observed generated-plan `jump_frac`, and therefore how much of the defect a
composer change can reach.

Usage:
    python -m diffuser.pl_eval.lidar_eval.seam_dose_control
    python -m diffuser.pl_eval.lidar_eval.seam_dose_control --n_paths 12
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
from diffuser.pl_eval.lidar_eval.decode_bench import (
    build_paths, assemble, LIDAR_CONFIG, ENV)
from diffuser.pl_eval.lidar_eval.g2a_floor import _DecodeShim, g2a_metrics
from diffuser.pl_eval.lidar_eval.plan_decode_gate import G2A_JUMP_FRAC

ARMS = ('blend', 'hard', 'xyblend')


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n_paths', type=int, default=6)
    p.add_argument('--n_comp', type=int, default=8)
    p.add_argument('--hzn', type=int, default=160)
    p.add_argument('--len_ovlp', type=int, default=56)
    p.add_argument('--blend_type', default='exp')
    p.add_argument('--exp_beta', type=float, default=2.0)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--median', type=int, default=15)
    p.add_argument('--doses', default='0,0.5,1,2,4',
                   help='segment displacement in world units (1 cell = 4 wu)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='logs/lidar_eval/seam_dose_control.json')
    return p.parse_args()


def main():
    a = _parse()
    n_comp, hzn, len_ovlp = a.n_comp, a.hzn, a.len_ovlp
    tot = n_comp * hzn - (n_comp - 1) * len_ovlp
    S = hzn - len_ovlp
    doses = [float(x) for x in a.doses.split(',')]

    scanner = make_scanner_for_env(ENV, LIDAR_CONFIG)
    grid = load_or_build_grid(ENV, LIDAR_CONFIG, xy_step=0.5, n_theta=1, verbose=False)
    loc = GridLocalizer(grid, sigma=1.0)
    cell = float(scanner.maze_unit)
    paths = build_paths(a.n_paths, tot, 0.3, np.random.default_rng(a.seed))
    print(f'[dose] {len(paths)} ground-truth paths  tot={tot}  cell={cell:g} wu  '
          f'doses={doses}', flush=True)

    def dec(scans):
        return _DecodeShim(scanner, loc, 'argmin', a.stride, a.median, 1.0).decode(scans)

    out = {}
    print(f'\n{"dose(wu)":>9} {"cells":>6} ' + ' '.join(f'{k:>9}' for k in ARMS))
    for dose in doses:
        res = {k: [] for k in ARMS}
        for pth in paths:
            xy = np.asarray(pth[:tot], dtype=np.float64)
            segs = []
            for i in range(n_comp):
                seg = xy[i * S:i * S + hzn].copy()
                ## displace every other segment, so each shared overlap is contested
                if i % 2 == 1:
                    seg = seg + np.array([dose, 0.0])
                segs.append(scanner.scan_batch(seg, np.zeros(hzn)))
            segs = np.stack(segs).astype(np.float32)
            for arm in ('blend', 'hard'):
                pl = assemble(segs, n_comp, hzn, len_ovlp,
                              a.blend_type, a.exp_beta, mode=arm)
                res[arm].append(g2a_metrics(dec(pl), scanner, cell)[1])
            sxy = np.stack([dec(segs[i]) for i in range(n_comp)]).astype(np.float32)
            pl = assemble(sxy, n_comp, hzn, len_ovlp,
                          a.blend_type, a.exp_beta, mode='blend').astype(np.float64)
            res['xyblend'].append(g2a_metrics(pl, scanner, cell)[1])
        row = {k: float(np.mean(v)) for k, v in res.items()}
        out[f'{dose:g}'] = row
        print(f'{dose:9.2f} {dose/cell:6.2f} '
              + ' '.join(f'{row[k]:9.4f}' for k in ARMS), flush=True)
        if dose == 0.0:
            assert abs(row['blend'] - row['hard']) < 1e-12, (
                'zero-dose identity violated: with no segment disagreement the blend '
                'and the hard handover must be the same plan.')

    print(f'\n  G2a jump bar = {G2A_JUMP_FRAC}. Zero-dose row is the decoder floor, '
          f'not zero.')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(dict(n_paths=len(paths), n_comp=n_comp, hzn=hzn,
                       len_ovlp=len_ovlp, tot_hzn=tot, cell=cell,
                       blend_type=a.blend_type, exp_beta=a.exp_beta,
                       stride=a.stride, median=a.median,
                       g2a_jump_bar=G2A_JUMP_FRAC, by_dose=out), f, indent=2)
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
