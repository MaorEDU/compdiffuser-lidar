"""
Glue between the raw OGBench datasets and the LiDAR observation representation.

This module:
  * computes a LiDAR scan for every transition of an OGBench maze dataset
    (using ``diffuser.datasets.lidar.lidar_sim``),
  * caches the result on disk so the (expensive) ray-casting is done only once
    and re-used across training runs ("keep an exact copy of the dataset in the
    dedicated LiDAR format"), and
  * assembles the per-transition observation vector.

There is exactly **one** observation layout in this repo: ``'lidar'`` -- the
``n_beams`` scan vector and nothing else. The planner never receives ``(x, y)``,
in any form, at any point. (The previous fork also carried ``xyo_lidar`` /
``xyv_lidar`` / ``full_lidar`` composites for its learned inverse-dynamics arms;
comp_diffuser_ball_v2 has no learned executor -- the ball is driven by the
upstream saturating P-law -- so those layouts are deliberately absent.)

It is invoked from :func:`diffuser.datasets.ogb_dset.ogb_utils.ogb_sequence_dataset`
whenever ``dataset_config['lidar_config']`` is set, so the existing dataset classes
transparently switch to LiDAR observations.
"""

import os
import hashlib
import numpy as np

import diffuser.utils as utils
from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env


## Where the cached LiDAR copies of the datasets live.
LIDAR_CACHE_DIR = os.path.join('data', 'ogb_maze', 'lidar_cache')

## The only valid observation layout: the LiDAR scan vector alone.
LIDAR_OBS_LAYOUTS = ('lidar',)


def lidar_obs_dim(lidar_config):
    """Number of observation channels produced for a given lidar_config."""
    layout = lidar_config.get('obs_layout', 'lidar')
    _check_layout(layout)
    return int(lidar_config.get('n_beams', 30))


def lidar_goal_idxs(lidar_config):
    """Indices (within the observation) that carry the goal scan.

    With the single ``'lidar'`` layout this is simply every channel; the helper is
    kept so call sites read the same as they did in the previous fork."""
    layout = lidar_config.get('obs_layout', 'lidar')
    _check_layout(layout)
    return tuple(range(int(lidar_config.get('n_beams', 30))))


def _check_layout(layout):
    assert layout in LIDAR_OBS_LAYOUTS, (
        f'obs_layout must be one of {LIDAR_OBS_LAYOUTS}, got {layout!r}. '
        f'comp_diffuser_ball_v2 is scan-only by design: there is no learned '
        f'executor and therefore no (x, y)-bearing composite layout.')


def _cache_path(env_name, dset_h5path, scanner):
    cfg = scanner.config_dict()
    key = dict(env=str(env_name), dset=str(dset_h5path), **cfg)
    digest = hashlib.md5(repr(sorted(key.items())).encode()).hexdigest()[:12]
    safe_env = str(env_name).replace('/', '_').replace('\\', '_')
    fn = f'lidar_{safe_env}_{cfg["n_beams"]}beams_r{cfg["max_range"]:g}_{digest}.npz'
    return os.path.join(LIDAR_CACHE_DIR, fn)


def compute_or_load_lidar(env_name, observations, dset_h5path, lidar_config,
                          has_quat=False, use_cache=True):
    """Return the (N, n_beams) LiDAR scans for the given observations, building
    and caching them on first use.

    Returns:
        lidar:   (N, n_beams) float32
        scanner: the :class:`LidarScanner` instance (carries the geometry).
    """
    scanner = make_scanner_for_env(env_name, lidar_config)
    use_cache = use_cache and lidar_config.get('use_cache', True)
    cache_path = _cache_path(env_name, dset_h5path, scanner)

    if use_cache and os.path.isfile(cache_path):
        try:
            cached = np.load(cache_path)
            if cached['lidar'].shape[0] == observations.shape[0]:
                utils.print_color(
                    f'[lidar] loaded cached scans: {cache_path} '
                    f'{cached["lidar"].shape}', c='g')
                return cached['lidar'].astype(np.float32), scanner
            else:
                utils.print_color(
                    f'[lidar] cache size mismatch, recomputing ({cache_path})', c='y')
        except Exception as e:  # noqa: BLE001 - corrupt cache -> recompute
            utils.print_color(f'[lidar] failed to read cache ({e}); recomputing', c='y')

    utils.print_color(
        f'[lidar] computing LiDAR scans for {observations.shape[0]} transitions '
        f'(n_beams={scanner.n_beams}, max_range={scanner.max_range}) ...', c='c')
    lidar = scanner.scan_obs(observations, has_quat=has_quat)

    if use_cache:
        os.makedirs(LIDAR_CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_path, lidar=lidar)
        utils.print_color(f'[lidar] cached scans to {cache_path}', c='g')

    return lidar, scanner


def build_lidar_observations(observations, lidar, obs_layout):
    """Assemble the observation from the LiDAR scans (scan-only layout)."""
    _check_layout(obs_layout)
    return lidar.astype(np.float32)


def apply_lidar_transform(env, observations, lidar_config, dataset_config,
                          terminals=None):
    """Replace raw OGBench observations with the LiDAR observation.

    Args:
        env: an OGBench env (used for its ``.name`` and to pick the maze layout).
        observations: (N, D) raw OGBench observations (full state).
        lidar_config: dict with keys n_beams, max_range, fov, obs_layout, ...
        dataset_config: the surrounding dataset_config (for dset_h5path lookup).
        terminals: accepted and ignored -- the scan-only layout has no per-episode
            derived channel. Kept in the signature so the call site in
            ``ogb_utils.ogb_sequence_dataset`` needs no branch.

    Returns:
        new_obs:       (N, n_beams) float32 observations.
        obs_select_dim: list selecting all channels (so the caller's
                        ``obs[:, obs_select_dim]`` is a no-op pass-through).
    """
    del terminals  # scan-only layout: nothing to derive per episode
    env_name = getattr(env, 'name', str(env))
    obs_layout = lidar_config.get('obs_layout', 'lidar')
    _check_layout(obs_layout)

    ## the ant/humanoid carry a quaternion; the ball's obs is (x, y) -> yaw == 0
    has_quat = observations.shape[1] >= 7
    dset_h5path = getattr(env, 'dset_h5path', None)

    lidar, scanner = compute_or_load_lidar(
        env_name, observations, dset_h5path, lidar_config, has_quat=has_quat)

    new_obs = build_lidar_observations(observations, lidar, obs_layout)
    obs_select_dim = list(range(new_obs.shape[1]))

    utils.print_color(
        f'[lidar] obs_layout="{obs_layout}" -> observation_dim={new_obs.shape[1]} '
        f'(scanner: {scanner.config_dict()})', c='c')

    return new_obs, obs_select_dim
