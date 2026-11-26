"""
Glue between the raw OGBench datasets and the LiDAR observation representation.

This module:
  * computes a LiDAR scan for every transition of an OGBench maze dataset
    (using ``diffuser.datasets.lidar.lidar_sim``),
  * caches the result on disk so the (expensive) ray-casting is done only once
    and re-used across training runs ("keep an exact copy of the dataset in the
    dedicated LiDAR format"), and
  * assembles the per-transition observation vector in one of two layouts:
        - ``'lidar'``       : the n_beams LiDAR vector only  (used by the
                              diffusion planner -- we no longer give it x/y).
        - ``'xyo_lidar'``   : [x, y, cos(yaw), sin(yaw), lidar...]  (used by the
                              inverse-dynamics model, whose spec says it receives
                              position, orientation and LiDAR readings).

It is invoked from :func:`diffuser.datasets.ogb_dset.ogb_utils.ogb_sequence_dataset`
whenever ``dataset_config['lidar_config']`` is set, so the existing dataset
classes (``OgB_SeqDataset_V2``, ``OgB_InvDyn_SeqDataset_V1``) transparently switch
to LiDAR observations.
"""

import os
import hashlib
import numpy as np

import diffuser.utils as utils
from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env, obs_to_xy_yaw


## Where the cached LiDAR copies of the datasets live.
LIDAR_CACHE_DIR = os.path.join('data', 'ogb_maze', 'lidar_cache')

## Valid observation layouts.
##   'lidar'      : LiDAR scan only (the planner's observation).
##   'xyo_lidar'  : [x, y, cos(yaw), sin(yaw), lidar]  (car-like inv-dyn obs).
##   'full_lidar' : [full raw OGBench obs (incl. joint angles + velocities), lidar]
##                  -- restores the proprioception the ant needs to predict 8 leg
##                  torques (the 'xyo_lidar' layout drops obs[7:] = joints+vels).
##   'xyv_lidar'  : [x, y, vx, vy, lidar]  (BALL / pointmaze inv-dyn obs; the project's
##                  4-D point-mass state + scan). OGBench pointmaze observations are
##                  position-only (x, y), so vx, vy are FINITE-DIFFERENCE velocities
##                  (one env step apart, v_0 = 0 at each episode start) computed
##                  per-episode from the ``terminals`` array. The online rollout
##                  must build its live composite the same way (see
##                  ``finite_diff_velocities`` and the ball rollout planner).
LIDAR_OBS_LAYOUTS = ('lidar', 'xyo_lidar', 'full_lidar', 'xyv_lidar')

## Default raw OGBench obs dim for the ant (x,y,z,quat(4),8 joints, velocities).
## Used only by the dim/goal-index helpers for 'full_lidar'; the dataset itself
## derives the true dim from the actual observation array shape.
DEFAULT_RAW_OBS_DIM = 29


def lidar_obs_dim(lidar_config):
    """Number of observation channels produced for a given lidar_config."""
    n_beams = int(lidar_config.get('n_beams', 30))
    layout = lidar_config.get('obs_layout', 'lidar')
    if layout == 'lidar':
        return n_beams
    elif layout == 'xyo_lidar':
        return n_beams + 4  # x, y, cos(yaw), sin(yaw)
    elif layout == 'xyv_lidar':
        return n_beams + 4  # x, y, vx, vy (ball 4-D state)
    elif layout == 'full_lidar':
        raw = int(lidar_config.get('raw_obs_dim', DEFAULT_RAW_OBS_DIM))
        return raw + n_beams  # full raw obs + lidar (lidar appended last)
    raise ValueError(f'Unknown lidar obs_layout: {layout}')


def lidar_goal_idxs(lidar_config):
    """Indices (within the composite observation) that the inverse-dynamics
    model should treat as the "goal" -- i.e. the LiDAR slice. The LiDAR scan is
    always appended LAST, so for every layout the goal is the final n_beams dims."""
    n_beams = int(lidar_config.get('n_beams', 30))
    layout = lidar_config.get('obs_layout', 'lidar')
    if layout == 'lidar':
        return tuple(range(n_beams))
    elif layout == 'xyo_lidar':
        return tuple(range(4, 4 + n_beams))
    elif layout == 'xyv_lidar':
        return tuple(range(4, 4 + n_beams))
    elif layout == 'full_lidar':
        raw = int(lidar_config.get('raw_obs_dim', DEFAULT_RAW_OBS_DIM))
        return tuple(range(raw, raw + n_beams))
    raise ValueError(f'Unknown lidar obs_layout: {layout}')


def _cache_path(env_name, dset_h5path, scanner):
    cfg = scanner.config_dict()
    key = dict(env=str(env_name), dset=str(dset_h5path), **cfg)
    digest = hashlib.md5(repr(sorted(key.items())).encode()).hexdigest()[:12]
    safe_env = str(env_name).replace('/', '_').replace('\\', '_')
    fn = f'lidar_{safe_env}_{cfg["n_beams"]}beams_r{cfg["max_range"]:g}_{digest}.npz'
    return os.path.join(LIDAR_CACHE_DIR, fn)


def compute_or_load_lidar(env_name, observations, dset_h5path, lidar_config,
                          has_quat=True, use_cache=True):
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


def finite_diff_velocities(xy, terminals=None):
    """Per-episode finite-difference velocities for a position-only dataset.

    ``v_t = xy_t - xy_{t-1}`` (displacement per env step; NO division by a wall-
    clock dt, so online code reproduces it exactly by subtracting two consecutive
    observations). ``v = 0`` at the first frame of every episode.

    Args:
        xy: (N, 2) concatenated positions of the whole dataset.
        terminals: (N,) 0/1 float or bool; 1 marks the LAST frame of an episode
            (OGBench convention). None -> treated as one single episode.
    Returns:
        vel: (N, 2) float32.
    """
    xy = np.asarray(xy, dtype=np.float64)
    vel = np.zeros_like(xy)
    vel[1:] = xy[1:] - xy[:-1]
    if terminals is not None:
        term = np.asarray(terminals).reshape(-1).astype(bool)
        assert term.shape[0] == xy.shape[0], (term.shape, xy.shape)
        ## frame t is an episode START if the previous frame was terminal
        ep_start = np.zeros(xy.shape[0], dtype=bool)
        ep_start[0] = True
        ep_start[1:] = term[:-1]
        vel[ep_start] = 0.0
    else:
        vel[0] = 0.0
    return vel.astype(np.float32)


def build_lidar_observations(observations, lidar, obs_layout, has_quat=True,
                             terminals=None):
    """Assemble the composite observation from raw obs + LiDAR scans."""
    if obs_layout == 'lidar':
        return lidar.astype(np.float32)
    elif obs_layout == 'xyo_lidar':
        xy, yaw = obs_to_xy_yaw(observations, has_quat=has_quat)
        comp = np.concatenate(
            [xy, np.cos(yaw)[:, None], np.sin(yaw)[:, None], lidar], axis=1)
        return comp.astype(np.float32)
    elif obs_layout == 'xyv_lidar':
        ## BALL (pointmaze): [x, y, vx, vy, lidar]. The raw obs is position-only,
        ## so velocities are per-episode finite differences (terminals-aware).
        xy = np.asarray(observations, dtype=np.float64)[:, 0:2]
        vel = finite_diff_velocities(xy, terminals=terminals)
        comp = np.concatenate([xy.astype(np.float32), vel, lidar], axis=1)
        return comp.astype(np.float32)
    elif obs_layout == 'full_lidar':
        ## Keep the FULL raw observation (x, y, z, quat, joint angles + velocities)
        ## and append the LiDAR scan last. This gives the inverse-dynamics model the
        ## proprioception it needs to map (state, goal-scan) -> 8 leg torques.
        comp = np.concatenate([np.asarray(observations), lidar], axis=1)
        return comp.astype(np.float32)
    raise ValueError(f'Unknown lidar obs_layout: {obs_layout}')


def apply_lidar_transform(env, observations, lidar_config, dataset_config,
                          terminals=None):
    """Replace raw OGBench observations with the LiDAR composite observation.

    Args:
        env: an OGBench env (used for its ``.name`` and to pick the maze layout).
        observations: (N, D) raw OGBench observations (full state).
        lidar_config: dict with keys n_beams, max_range, fov, obs_layout, ...
        dataset_config: the surrounding dataset_config (for dset_h5path lookup).
        terminals: (N,) episode-end flags of the concatenated dataset. Required
            for the 'xyv_lidar' layout (per-episode finite-diff velocities);
            unused otherwise.

    Returns:
        new_obs:       (N, D') float32 composite observations.
        obs_select_dim: list selecting all D' channels (so the caller's
                        ``obs[:, obs_select_dim]`` is a no-op pass-through).
    """
    env_name = getattr(env, 'name', str(env))
    obs_layout = lidar_config.get('obs_layout', 'lidar')
    assert obs_layout in LIDAR_OBS_LAYOUTS, \
        f'obs_layout must be one of {LIDAR_OBS_LAYOUTS}, got {obs_layout}'
    assert not (obs_layout == 'xyv_lidar' and terminals is None), \
        "obs_layout='xyv_lidar' needs the dataset terminals for per-episode velocities"

    has_quat = observations.shape[1] >= 7  # ant / humanoid carry a quaternion
    dset_h5path = getattr(env, 'dset_h5path', None)

    lidar, scanner = compute_or_load_lidar(
        env_name, observations, dset_h5path, lidar_config, has_quat=has_quat)

    new_obs = build_lidar_observations(observations, lidar, obs_layout,
                                       has_quat=has_quat, terminals=terminals)
    obs_select_dim = list(range(new_obs.shape[1]))

    utils.print_color(
        f'[lidar] obs_layout="{obs_layout}" -> observation_dim={new_obs.shape[1]} '
        f'(scanner: {scanner.config_dict()})', c='c')

    return new_obs, obs_select_dim
