"""G-hist -- is the history channel actually being used?

Run before spending any rollout GPU time on the history arms. It is a paired test on
HELD-OUT episodes with a single checkpoint: identical windows, identical noise,
identical start/goal conditioning, the only difference being whether the observation
history is fed or dropped. If the two are indistinguishable, the model learned to
ignore the channel and the H / RH rollouts are cancelled -- falsify on CPU before
spending GPU, which is the house rule that killed a 60-beam retrain in two seconds.

What it measures, per (window, timestep):

  * ``x0`` denoising error with history vs. with the history dropped, over the same
    noise draw -- a paired difference, so window-to-window variance cancels.
  * the same split by frame position. History should help MOST at the start of the
    window (where continuing the observed motion is informative) and wash out toward
    the goal, which is inpainted anyway. A uniform "improvement" across all frames is
    a red flag: it usually means something other than the history changed.
  * how often history helps (sign test over windows), not just the mean, because one
    pathological window can carry a mean.

Pre-registered pass: the paired mean x0 error must be LOWER with history at the
p<0.01 level on a two-sided sign test, AND the improvement must be larger over the
first quarter of the window than over the last quarter. Mean-only improvements, or
improvements that are flat across frame position, do not pass.

Usage:
    python -m diffuser.pl_eval.lidar_eval.hist_holdout_gate \
        --config config/ogb_pnt_maze/og_pntM_Gi_lidar30_hist32_Cd_Stgl_PadBuf_Ft64_ts512.py \
        --n_win 256 --out logs/lidar_eval/ghist.json
"""

import argparse
import json
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.append('./')

import numpy as np
import torch


def _parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--n_win', type=int, default=256, help='held-out windows')
    p.add_argument('--timesteps', default='64,192,320,448',
                   help='diffusion timesteps to evaluate at')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='')
    return p.parse_args()


def _load(cfg_path):
    """Load the trained EMA diffusion model + its normalizer + the scanner geometry."""
    import diffuser.utils as utils
    from diffuser.ogb_task.ogb_maze_v1.plan_ogb_stgl_sml import Parser
    from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env

    argv_bak = sys.argv
    sys.argv = ['hist_holdout_gate', '--config', cfg_path]
    try:
        args_train = Parser().parse_args('diffusion')
        args = Parser().parse_args('plan')
    finally:
        sys.argv = argv_bak

    lidar_config = args_train.dataset_config.get('lidar_config', None)
    assert lidar_config is not None, f'{cfg_path} is not a LiDAR config'

    latest_e = utils.get_latest_epoch((args.logbase, args.dataset, args_train.exp_name))
    args_train.diffusion_epoch = latest_e
    args.diffusion_epoch = latest_e
    dfu_exp = utils.load_stgl_sml_diffusion(
        args.logbase, args.dataset, args_train.exp_name, epoch=latest_e)
    diffusion = dfu_exp.ema
    diffusion.eval()

    run_dir = os.path.join(args.logbase, args.dataset, args_train.exp_name)
    normalizer = utils.load_config(run_dir, 'dataset_config.pkl')().normalizer
    scanner = make_scanner_for_env(args.dataset, lidar_config)
    return diffusion, normalizer, scanner, args.dataset, latest_e


def _holdout_windows(scanner, env_name, horizon, hist_len, n_win, rng):
    """(n_win, horizon, D) windows + (n_win, hist_len, D) histories from the val set.

    Built by re-rendering scans along real validation trajectories, so the history is
    a real trajectory prefix -- exactly what the rollout's deque holds at a replan."""
    from diffuser.datasets.lidar.hist_window import left_pad_history

    path = os.path.expanduser(f'~/.ogbench/data/{env_name}-val.npz')
    d = np.load(path)
    xy = np.asarray(d['observations'], dtype=np.float64)[:, :2]
    term = np.asarray(d['terminals']).reshape(-1).astype(bool)
    ends = np.flatnonzero(term)
    starts = np.concatenate([[0], ends[:-1] + 1])

    wins, hists = [], []
    tries = 0
    while len(wins) < n_win and tries < n_win * 20:
        tries += 1
        e = rng.integers(len(starts))
        s0, s1 = int(starts[e]), int(ends[e]) + 1
        if s1 - s0 < horizon + 1:
            continue
        ## start far enough in that a REAL history exists (not pure padding)
        lo = int(rng.integers(s0, s1 - horizon))
        seg = np.stack([scanner.scan(float(x), float(y), 0.0) for x, y in xy[lo:lo + horizon]])
        h_lo = max(s0, lo - hist_len)
        pre = (np.stack([scanner.scan(float(x), float(y), 0.0) for x, y in xy[h_lo:lo]])
               if h_lo < lo else np.empty((0, seg.shape[1]), dtype=np.float32))
        wins.append(seg.astype(np.float32))
        hists.append(left_pad_history(pre, hist_len, fill=seg[0]))
    assert len(wins) == n_win, f'only found {len(wins)} usable windows'
    return np.stack(wins), np.stack(hists)


def _tj_cond(batch, device, hist=None):
    """The `both_stgl` regime: frame 0 and frame H-1 inpainted, no overlaps.

    Chosen because training definitely covers it (st_cd_use_inpat & end_cd_use_inpat)
    and because it isolates the history: nothing else in the conditioning varies."""
    return dict(
        do_cond=False,
        is_st_inpat=torch.ones(batch, dtype=torch.bool, device=device),
        is_end_inpat=torch.ones(batch, dtype=torch.bool, device=device),
        st_ovlp_is_drop=None, end_ovlp_is_drop=None,
        st_ovlp_traj=None, end_ovlp_traj=None,
        st_ovlp_t=torch.zeros(batch, dtype=torch.long, device=device),
        end_ovlp_t=torch.zeros(batch, dtype=torch.long, device=device),
        hist_traj=hist,
        hist_is_drop=(torch.zeros(batch, dtype=torch.bool, device=device) if hist is not None
                      else torch.ones(batch, dtype=torch.bool, device=device)),
    )


def main():
    a = _parse()
    from diffuser.models.helpers import apply_conditioning

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    diffusion, normalizer, scanner, env_name, epoch = _load(a.config)
    device = next(diffusion.parameters()).device
    hist_len = int(getattr(diffusion.model, 'hist_len', 0))
    assert hist_len > 0, (
        'this checkpoint has no history encoder -- there is nothing for G-hist to '
        'measure. Train the hist32 config first.')
    horizon = int(diffusion.horizon)
    tsteps = [int(t) for t in a.timesteps.split(',')]

    wins, hists = _holdout_windows(scanner, env_name, horizon, hist_len, a.n_win, rng)
    wins = normalizer.normalize(wins, 'observations').astype(np.float32)
    hists = normalizer.normalize(hists, 'observations').astype(np.float32)

    q1, q4 = horizon // 4, horizon - horizon // 4
    per_win_on, per_win_off = [], []
    early_on, early_off, late_on, late_off = [], [], [], []

    with torch.no_grad():
        for lo in range(0, a.n_win, a.batch):
            x0 = torch.from_numpy(wins[lo:lo + a.batch]).to(device)
            hh = torch.from_numpy(hists[lo:lo + a.batch]).to(device)
            b = x0.shape[0]
            cond = {0: x0[:, 0, :].clone(), horizon - 1: x0[:, -1, :].clone()}
            for t in tsteps:
                t_2d = torch.full((b, horizon), t, device=device, dtype=torch.long)
                noise = torch.randn_like(x0)          # SAME noise for both arms
                xt = diffusion.q_sample(x_start=x0, t_2d=t_2d, noise=noise)
                xt = apply_conditioning(xt, cond, 0)
                for tag, h in (('on', hh), ('off', None)):
                    pred = diffusion.model(xt, t_2d[:, 0], _tj_cond(b, device, h))
                    err = ((pred - x0) ** 2).mean(dim=2)          # (B, H)
                    (per_win_on if tag == 'on' else per_win_off).append(
                        err.mean(dim=1).cpu().numpy())
                    (early_on if tag == 'on' else early_off).append(
                        err[:, :q1].mean(dim=1).cpu().numpy())
                    (late_on if tag == 'on' else late_off).append(
                        err[:, q4:].mean(dim=1).cpu().numpy())
            print(f'[G-hist] {min(lo + a.batch, a.n_win)}/{a.n_win} windows', flush=True)

    on = np.concatenate(per_win_on); off = np.concatenate(per_win_off)
    e_on = np.concatenate(early_on); e_off = np.concatenate(early_off)
    l_on = np.concatenate(late_on);  l_off = np.concatenate(late_off)

    d = off - on                                   # > 0 => history helped
    n_help = int((d > 0).sum()); n_tot = int(len(d))
    ## two-sided sign test against p=0.5, normal approximation (n is in the thousands)
    z = (n_help - n_tot / 2) / np.sqrt(n_tot / 4)
    from math import erfc, sqrt
    p_val = float(erfc(abs(z) / sqrt(2)))

    rel_early = float((e_off.mean() - e_on.mean()) / max(e_off.mean(), 1e-12))
    rel_late = float((l_off.mean() - l_on.mean()) / max(l_off.mean(), 1e-12))

    agg = dict(
        config=a.config, epoch=int(epoch), hist_len=hist_len, n_win=a.n_win,
        timesteps=tsteps, n_paired=n_tot,
        mse_hist_on=float(on.mean()), mse_hist_off=float(off.mean()),
        rel_improvement=float((off.mean() - on.mean()) / max(off.mean(), 1e-12)),
        frac_windows_helped=n_help / n_tot, sign_test_z=float(z), sign_test_p=p_val,
        rel_improvement_first_quarter=rel_early,
        rel_improvement_last_quarter=rel_late,
    )
    agg['pass'] = bool(on.mean() < off.mean() and p_val < 0.01 and rel_early > rel_late)

    print('\n=============== G-hist ===============')
    print(f"  x0 MSE   hist ON {agg['mse_hist_on']:.6f}   OFF {agg['mse_hist_off']:.6f}"
          f"   ({agg['rel_improvement'] * 100:+.2f}%)")
    print(f"  helped on {agg['frac_windows_helped'] * 100:.1f}% of paired windows "
          f"(sign test z={z:.2f}, p={p_val:.2e})")
    print(f"  by position: first quarter {rel_early * 100:+.2f}%   "
          f"last quarter {rel_late * 100:+.2f}%")
    print(f"  -> {'PASS' if agg['pass'] else 'FAIL'}")
    if not agg['pass']:
        print('  ** The model is not using the history. Cancel the H and RH rollouts. **')
        if rel_early <= rel_late:
            print('  (improvement is not concentrated early -- if anything moved, it '
                  'was probably not the history.)')
    print('======================================\n')

    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f'wrote {a.out}')
    return 0 if agg['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
