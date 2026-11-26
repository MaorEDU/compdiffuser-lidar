"""G-parity + history-plumbing tests for the U-Net conditioning branch.

Two claims, both load-bearing:

1. **Parity.** With ``hist_model_config`` absent the denoiser is bit-identical to the
   CompDiffuser release: same parameter tensors, same forward output. The whole repo
   rests on the baseline arm really being the paper's model with only the observation
   swapped, so this is checked mechanically rather than by reading the diff.

2. **Drop semantics.** With the history branch present, ``hist_is_drop=True`` is
   bit-identical to passing no history at all, and a real history actually changes the
   output. That equivalence is what lets one checkpoint serve both the history arm and
   the no-history control.

Run: ``pytest diffuser/pl_eval/lidar_eval/test_hist_parity.py``
(The release copy is located relative to this repo; the parity test skips if absent.)
"""

import os
import sys
import importlib.util

import pytest
import torch

from diffuser.models.cd_stgl_sml_dfu.stgl_sml_temporal_cond_v1 import (
    Unet1D_TjTi_Stgl_Cond_V1,
)

N_BEAMS, HZN, LOVLP, HIST = 30, 160, 56, 32
RELEASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'comp_diffuser_release'))


def _ovlp_cfg(c_traj_hzn=LOVLP):
    return dict(
        c_traj_hzn=c_traj_hzn, in_dim=N_BEAMS, base_dim=32, dim_mults=(1, 2, 3, 4),
        time_dim=32, out_dim=256,
        tjti_enc_config=dict(t_seq_encoder_type='mlp', cnn_out_dim=128,
                             final_mlp_dims=[1280, 512, 256], f_conv_ks=3))


def _net_cfg(**extra):
    cfg = dict(
        t_seq_encoder_type='mlp', cat_t_w=True, resblock_ksize=5,
        st_ovlp_model_config=_ovlp_cfg(), end_ovlp_model_config=_ovlp_cfg(),
        ext_cond_dim=2 * 256, energy_mode=False, time_mlp_config=3,
        inpaint_token_dim=32, inpaint_token_type='const')
    cfg.update(extra)
    return cfg


def _build(cls, **extra):
    torch.manual_seed(0)
    return cls(horizon=HZN, transition_dim=N_BEAMS, base_dim=128,
               dim_mults=(1, 2, 4, 8), time_dim=96,
               network_config=_net_cfg(**extra)).eval()


def _inputs(batch=4, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(batch, HZN, N_BEAMS)
    t = torch.zeros(batch, dtype=torch.long)
    cond = dict(
        is_st_inpat=torch.ones(batch, dtype=torch.bool),
        is_end_inpat=torch.zeros(batch, dtype=torch.bool),
        st_ovlp_is_drop=None,
        end_ovlp_is_drop=torch.zeros(batch, dtype=torch.bool),
        st_ovlp_traj=None,
        end_ovlp_traj=torch.randn(batch, LOVLP, N_BEAMS),
        st_ovlp_t=torch.zeros(batch, dtype=torch.long),
        end_ovlp_t=torch.zeros(batch, dtype=torch.long),
    )
    hist = torch.randn(batch, HIST, N_BEAMS)
    return x, t, cond, hist


def _load_release_cls():
    """Import the release denoiser class from the sibling checkout, if present."""
    path = os.path.join(RELEASE_DIR, 'diffuser', 'models', 'cd_stgl_sml_dfu',
                        'stgl_sml_temporal_cond_v1.py')
    if not os.path.isfile(path):
        return None
    ## the module imports `diffuser.*` -- resolve those against the RELEASE tree so we
    ## really are testing the released code, not this repo's copy of the helpers.
    sys.path.insert(0, RELEASE_DIR)
    try:
        for mod in [m for m in list(sys.modules) if m.startswith('diffuser')]:
            del sys.modules[mod]
        spec = importlib.util.spec_from_file_location('_release_tempcond', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Unet1D_TjTi_Stgl_Cond_V1
    finally:
        sys.path.remove(RELEASE_DIR)


def test_parity_with_release_when_history_absent():
    """G-parity: no hist_model_config => bit-identical to the CompDiffuser release."""
    release_cls = _load_release_cls()
    if release_cls is None:
        pytest.skip(f'no release checkout at {RELEASE_DIR}')

    ours = _build(Unet1D_TjTi_Stgl_Cond_V1)
    theirs = _build(release_cls)

    ours_sd, theirs_sd = ours.state_dict(), theirs.state_dict()
    assert set(ours_sd) == set(theirs_sd), (
        'parameter set differs from the release:\n'
        f'  only ours:   {sorted(set(ours_sd) - set(theirs_sd))}\n'
        f'  only theirs: {sorted(set(theirs_sd) - set(ours_sd))}')
    for k in ours_sd:
        assert torch.equal(ours_sd[k], theirs_sd[k]), f'parameter {k} differs'

    x, t, cond, _ = _inputs()
    with torch.no_grad():
        assert torch.equal(ours(x, t, dict(cond)), theirs(x, t, dict(cond)))


def test_history_drop_is_equivalent_to_no_history():
    model = _build(Unet1D_TjTi_Stgl_Cond_V1, hist_model_config=_ovlp_cfg(HIST))
    x, t, cond, hist = _inputs()
    b = x.shape[0]
    with torch.no_grad():
        dropped = model(x, t, {**cond, 'hist_traj': hist,
                               'hist_is_drop': torch.ones(b, dtype=torch.bool)})
        none = model(x, t, {**cond, 'hist_traj': None, 'hist_is_drop': None})
        absent = model(x, t, dict(cond))
    assert torch.equal(dropped, none)
    assert torch.equal(dropped, absent)


def test_history_changes_the_prediction():
    model = _build(Unet1D_TjTi_Stgl_Cond_V1, hist_model_config=_ovlp_cfg(HIST))
    x, t, cond, hist = _inputs()
    b = x.shape[0]
    with torch.no_grad():
        on = model(x, t, {**cond, 'hist_traj': hist,
                          'hist_is_drop': torch.zeros(b, dtype=torch.bool)})
        off = model(x, t, dict(cond))
    assert not torch.allclose(on, off, atol=1e-6), (
        'the history branch is inert -- conditioning on a real history left the '
        'prediction unchanged')


def test_history_length_mismatch_is_loud():
    model = _build(Unet1D_TjTi_Stgl_Cond_V1, hist_model_config=_ovlp_cfg(HIST))
    x, t, cond, _ = _inputs()
    b = x.shape[0]
    bad = torch.randn(b, HIST + 1, N_BEAMS)
    with pytest.raises(AssertionError):
        with torch.no_grad():
            model(x, t, {**cond, 'hist_traj': bad,
                         'hist_is_drop': torch.zeros(b, dtype=torch.bool)})
