#!/usr/bin/env python3
"""Tests for the ONE shared sensing-geometry resolver (lidar_grid_common).

Runs standalone on CPU, no torch, no pytest:
    python diffuser/pl_eval/lidar_eval/test_lidar_config_resolver.py

Why this file exists: on 2026-07-29 both bridge jobs printed
``[audit] lidar_config from MODULE DEFAULT (unverified!)`` because the resolver
did a FLAT ``cfg['lidar_config']`` lookup while ``train_ogb_stgl_sml.py`` saves
it NESTED at ``cfg['dataset_config']['lidar_config']``. The mr12 arm was correct
only because the module default coincided with the trained value; the mr24 arm
only because of its CLI ``--max_range`` flag. Case 3 below is the exact nested
layout read off the run dir, and it is the one that regressed.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
from diffuser.pl_eval.lidar_eval.lidar_grid_common import (  # noqa: E402
    resolve_lidar_config, assert_npz_matches, json_safe_lidar_config,
    _lidar_cfg_from_mapping, DEFAULT_LIDAR_CONFIG)


class FakeConfig(dict):
    """Stands in for diffuser.utils.config.Config, which is a Mapping."""


def _patch_load_config(saved):
    """Make the resolver's lazy ``from diffuser.utils import load_config`` return
    ``saved`` without importing torch."""
    import types
    mod = sys.modules.get('diffuser.utils')
    if mod is None:
        mod = types.ModuleType('diffuser.utils')
        sys.modules['diffuser.utils'] = mod
    mod.load_config = lambda *a, **k: saved
    return mod


## the real layout, copied from
## logs/pointmaze-giant-stitch-v0/diffusion/og_pntM_Gi_lidar30r24_..._T512/dataset_config.pkl
REAL_NESTED_R24 = FakeConfig({
    'dataset_config': {
        'obs_select_dim': (0, 1), 'dset_type': 'ogb', 'pad_option_2': 'buf',
        'pad_type': 'first', 'extra_pad': 64,
        'lidar_config': {'n_beams': 30, 'max_range': 24.0, 'fov': 6.283185307179586,
                         'step': 0.1, 'obs_layout': 'lidar', 'use_cache': True},
        'goal_band_k': 1},
    'env': 'pointmaze-giant-stitch-v0', 'horizon': 160, 'max_path_length': 300,
})

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def t1_nested_lookup_helper():
    cfg, where = _lidar_cfg_from_mapping(REAL_NESTED_R24)
    assert cfg['max_range'] == 24.0, cfg
    assert 'nested' in where, where


@case
def t2_flat_lookup_still_works():
    flat = FakeConfig({'lidar_config': {'n_beams': 30, 'max_range': 12.0}})
    cfg, where = _lidar_cfg_from_mapping(flat)
    assert cfg['max_range'] == 12.0 and 'flat' in where, (cfg, where)


@case
def t3_real_nested_run_dir_resolves_mr24():
    """THE REGRESSION: this used to KeyError -> MODULE DEFAULT -> silent mr12."""
    _patch_load_config(REAL_NESTED_R24)
    cfg = resolve_lidar_config(logdir='/fake/run/dir')
    assert cfg['max_range'] == 24.0, cfg
    assert cfg['n_beams'] == 30 and isinstance(cfg['n_beams'], int), cfg
    assert isinstance(cfg['max_range'], float), cfg


@case
def t4_missing_key_falls_back_loudly():
    _patch_load_config(FakeConfig({'env': 'x', 'horizon': 160}))
    cfg = resolve_lidar_config(logdir='/fake/run/dir')
    assert cfg == {**DEFAULT_LIDAR_CONFIG,
                   'n_beams': 30, 'max_range': 12.0}, cfg


@case
def t5_cli_overrides_win():
    _patch_load_config(REAL_NESTED_R24)
    cfg = resolve_lidar_config(logdir='/fake/run/dir', max_range=48, n_beams=60)
    assert cfg['max_range'] == 48.0 and cfg['n_beams'] == 60, cfg


@case
def t6_dump_meta_beats_the_run_dir():
    _patch_load_config(REAL_NESTED_R24)
    meta = json.dumps(dict(logdir='/fake/run/dir', epoch=1600000,
                           lidar_config=dict(n_beams=30, max_range=12.0)))
    cfg = resolve_lidar_config(meta=meta)
    assert cfg['max_range'] == 12.0, cfg


@case
def t7_old_meta_without_lidar_config_recovers_via_its_logdir():
    """ball_plans_lidar_1600000.npz has no lidar_config in meta but does record
    logdir -- with the nested lookup that is enough to recover the geometry."""
    _patch_load_config(REAL_NESTED_R24)
    meta = json.dumps(dict(logdir='/fake/run/dir', epoch=1600000, n_comp=8))
    cfg = resolve_lidar_config(meta=meta)
    assert cfg['max_range'] == 24.0, cfg


@case
def t8_npz_guard_catches_range_mismatch():
    ok = np.linspace(0.1, 24.0, 300).reshape(10, 30).astype(np.float32)
    assert_npz_matches(ok, dict(max_range=24.0, n_beams=30), 'fake_r24.npz')
    bad = np.linspace(0.1, 12.0, 300).reshape(10, 30).astype(np.float32)
    try:
        assert_npz_matches(bad, dict(max_range=24.0, n_beams=30), 'fake_mr12.npz')
    except AssertionError as e:
        assert 'Wrong normalizer source' in str(e), str(e)
    else:
        raise AssertionError('mr12 npz was accepted for an mr24 model')


@case
def t9_npz_guard_catches_beam_mismatch():
    b = np.linspace(0.1, 12.0, 600).reshape(10, 60).astype(np.float32)
    try:
        assert_npz_matches(b, dict(max_range=12.0, n_beams=30), 'fake_60beam.npz')
    except AssertionError as e:
        assert '60 beams' in str(e), str(e)
    else:
        raise AssertionError('60-beam npz was accepted for a 30-beam model')


@case
def t10_json_safe_round_trips():
    cfg = dict(n_beams=np.int64(30), max_range=np.float32(24.0), fov=2 * np.pi,
               obs_layout='lidar', use_cache=True, step=0.1)
    out = json_safe_lidar_config(cfg)
    json.dumps(out)                                   ## must not raise
    assert out['n_beams'] == 30 and out['use_cache'] is True, out
    assert out['obs_layout'] == 'lidar', out


if __name__ == '__main__':
    fails = 0
    for fn in CASES:
        try:
            fn()
            print(f'  PASS {fn.__name__}')
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f'  FAIL {fn.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(CASES) - fails}/{len(CASES)} resolver tests passed')
    sys.exit(1 if fails else 0)
