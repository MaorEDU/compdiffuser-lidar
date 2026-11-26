"""Instrument-vs-model decomposition for the G-hist FAIL.

G-hist reported history making x0 error 41.7% WORSE, concentrated in the first
quarter, p=1e-131. That is NOT "the model ignores the channel" (that would be
on==off, p~1). It is the channel being used and being harmful, which is the
signature of a history tensor that differs from the one training fed.

Holding the conditioning regime FIXED (the gate's _tj_cond, which is faithful:
st_ovlp_is_drop=None -> zero overlap feat == training's dropped-overlap case),
vary only WHERE the (window, history) pair comes from:

  A  the training dataloader itself  -- dataset.get_hist(), train distribution
  B  the gate's val re-render        -- _holdout_windows()

A helps + B hurts -> the gate's history construction is the bug (instrument).
A hurts           -> the model/training path is the bug; H and RH really are dead.
"""
import os, sys, json
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.append('./')

import numpy as np
import torch

from diffuser.pl_eval.lidar_eval.hist_holdout_gate import _load, _holdout_windows, _tj_cond
from diffuser.models.helpers import apply_conditioning
import diffuser.utils as utils

CFG = 'config/ogb_pnt_maze/og_pntM_Gi_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512.py'
N_WIN, BATCH, SEED = 256, 32, 0
TSTEPS = [64, 192, 320, 448]


def paired(diffusion, wins, hists, device, horizon, tag):
    """The gate's own measurement, verbatim, on whatever data it is handed."""
    q1, q4 = horizon // 4, horizon - horizon // 4
    on_l, off_l, e_on, e_off, l_on, l_off = [], [], [], [], [], []
    torch.manual_seed(SEED)
    with torch.no_grad():
        for lo in range(0, len(wins), BATCH):
            x0 = torch.from_numpy(wins[lo:lo + BATCH]).to(device)
            hh = torch.from_numpy(hists[lo:lo + BATCH]).to(device)
            b = x0.shape[0]
            cond = {0: x0[:, 0, :].clone(), horizon - 1: x0[:, -1, :].clone()}
            for t in TSTEPS:
                t_2d = torch.full((b, horizon), t, device=device, dtype=torch.long)
                noise = torch.randn_like(x0)
                xt = diffusion.q_sample(x_start=x0, t_2d=t_2d, noise=noise)
                xt = apply_conditioning(xt, cond, 0)
                for name, h in (('on', hh), ('off', None)):
                    pred = diffusion.model(xt, t_2d[:, 0], _tj_cond(b, device, h))
                    err = ((pred - x0) ** 2).mean(dim=2)
                    (on_l if name == 'on' else off_l).append(err.mean(1).cpu().numpy())
                    (e_on if name == 'on' else e_off).append(err[:, :q1].mean(1).cpu().numpy())
                    (l_on if name == 'on' else l_off).append(err[:, q4:].mean(1).cpu().numpy())
    on, off = np.concatenate(on_l), np.concatenate(off_l)
    eo, ef = np.concatenate(e_on), np.concatenate(e_off)
    lo_, lf = np.concatenate(l_on), np.concatenate(l_off)
    d = off - on
    n_help, n_tot = int((d > 0).sum()), len(d)
    z = (n_help - n_tot / 2) / np.sqrt(n_tot / 4)
    from math import erfc, sqrt
    r = dict(
        source=tag, n_paired=n_tot,
        mse_on=float(on.mean()), mse_off=float(off.mean()),
        rel=float((off.mean() - on.mean()) / off.mean()),
        frac_helped=n_help / n_tot, z=float(z), p=float(erfc(abs(z) / sqrt(2))),
        rel_first_q=float((ef.mean() - eo.mean()) / ef.mean()),
        rel_last_q=float((lf.mean() - lo_.mean()) / lf.mean()),
    )
    print(f"\n[{tag}]  x0 MSE  off {r['mse_off']:.6f} -> on {r['mse_on']:.6f} "
          f"({r['rel']*100:+.2f}%)   helped {r['frac_helped']*100:.1f}%  p={r['p']:.2e}")
    print(f"[{tag}]  by position: first quarter {r['rel_first_q']*100:+.2f}%   "
          f"last quarter {r['rel_last_q']*100:+.2f}%", flush=True)
    return r


def main():
    diffusion, normalizer, scanner, env_name, epoch = _load(CFG)
    device = next(diffusion.parameters()).device
    horizon = int(diffusion.horizon)
    hist_len = int(getattr(diffusion.model, 'hist_len', 0))
    print(f'epoch={epoch}  horizon={horizon}  hist_len={hist_len}  device={device}', flush=True)

    ## ---- source A: the training dataloader, i.e. exactly what training fed ----
    run_dir = os.path.join('logs', env_name,
                           'diffusion/og_pntM_Gi_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512_T512')
    ds = utils.load_config(run_dir, 'dataset_config.pkl')()
    rng = np.random.default_rng(SEED)
    idxs = rng.integers(0, len(ds), size=N_WIN)
    wa, ha = [], []
    for i in idxs:
        b = ds[int(i)]
        wa.append(np.asarray(b.obs_trajs, dtype=np.float32))
        ha.append(np.asarray(b.hist_trajs, dtype=np.float32))
    wins_A, hists_A = np.stack(wa), np.stack(ha)

    ## ---- source B: the gate's val re-render ----
    wins_B, hists_B = _holdout_windows(scanner, env_name, horizon, hist_len, N_WIN,
                                       np.random.default_rng(SEED))
    wins_B = normalizer.normalize(wins_B, 'observations').astype(np.float32)
    hists_B = normalizer.normalize(hists_B, 'observations').astype(np.float32)

    ## ---- are the two history tensors even the same kind of object? ----
    print('\n================ HISTORY TENSOR STATS (normalized) ================')
    for tag, w, h in (('A/train', wins_A, hists_A), ('B/gate', wins_B, hists_B)):
        print(f'  {tag:8s} hist {h.shape} mean={h.mean():+.4f} sd={h.std():.4f} '
              f'min={h.min():+.4f} max={h.max():+.4f}')
        print(f'  {tag:8s} win  {w.shape} mean={w.mean():+.4f} sd={w.std():.4f} '
              f'min={w.min():+.4f} max={w.max():+.4f}')
        ## the last history frame is the frame just BEFORE the window's first frame,
        ## so in-distribution they are near-identical; a large gap means the window
        ## and its history are not adjacent in time.
        gap = np.linalg.norm(h[:, -1, :] - w[:, 0, :], axis=1)
        step = np.linalg.norm(np.diff(h, axis=1), axis=2).mean()
        print(f'  {tag:8s} ||hist[-1]-win[0]|| p50={np.percentile(gap,50):.4f} '
              f'p95={np.percentile(gap,95):.4f}   mean |hist step|={step:.4f}')
        ## share of the window that is pure left-padding (all frames equal to hist[0])
        pad = (np.abs(h - h[:, :1, :]).max(axis=2) < 1e-6).mean()
        print(f'  {tag:8s} frac of history frames equal to hist[0] (padding) = {pad:.3f}')
    print('===================================================================')

    res = [paired(diffusion, wins_A, hists_A, device, horizon, 'A/train-dataloader'),
           paired(diffusion, wins_B, hists_B, device, horizon, 'B/gate-val-render')]

    ## ---- cross: train windows + gate-style histories is not meaningful (different
    ## episodes), so the discriminator is A vs B alone.
    print('\n================== VERDICT ==================')
    a, b = res
    if a['rel'] > 0 and b['rel'] < 0:
        print('  A helps, B hurts -> the GATE\'s history construction is the bug.')
        print('  The model learned the channel. Fix _holdout_windows, re-run G-hist,')
        print('  and do NOT cancel H/RH on the current number.')
    elif a['rel'] < 0:
        print('  A hurts too -> the model/training path is at fault, not the gate.')
        print('  H and RH are genuinely dead on this checkpoint.')
    else:
        print('  Mixed/inconclusive -- read the numbers above.')
    print('=============================================')
    out = 'logs/lidar_eval/hist_source_ab.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(epoch=int(epoch), results=res), open(out, 'w'), indent=2)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
