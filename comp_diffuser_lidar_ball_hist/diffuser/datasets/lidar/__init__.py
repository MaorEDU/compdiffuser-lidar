"""
LiDAR-based observation toolkit for CompDiffuser / OGBench maze environments.

This package replaces the direct (x, y) position observation of the agent with a
simulated 2D LiDAR scan: a vector of ray distances measured against the maze walls.
For the ball (OGBench ``pointmaze``) the agent is a point mass with no heading, so
``obs_to_xy_yaw`` auto-detects ``has_quat=False`` and every scan is world-frame
(yaw == 0). See ``lidar_sim.py`` for the simulator and ``BALLV2.md`` for the design.

This is the *vanilla* LiDAR stack: raycaster, reference grid, histogram-filter
localizer (forward filter only -- no Viterbi), live self-localizer, and a PID.
"""

from diffuser.datasets.lidar.lidar_sim import (
    MAZE_MAPS,
    LidarScanner,
    get_maze_map,
    quat_to_yaw,
    obs_to_xy_yaw,
    make_scanner_for_env,
)
from diffuser.datasets.lidar.lidar_grid import LidarGrid, load_or_build_grid
from diffuser.datasets.lidar.lidar_localization import (
    GridLocalizer,
    scan_distance,
    measurement_likelihood,
)
from diffuser.datasets.lidar.lidar_arrival import LiveLocalizer
from diffuser.datasets.lidar.pid_controller import PID2D

__all__ = [
    'MAZE_MAPS',
    'LidarScanner',
    'get_maze_map',
    'quat_to_yaw',
    'obs_to_xy_yaw',
    'make_scanner_for_env',
    'LidarGrid',
    'load_or_build_grid',
    'GridLocalizer',
    'scan_distance',
    'measurement_likelihood',
    'LiveLocalizer',
    'PID2D',
]
