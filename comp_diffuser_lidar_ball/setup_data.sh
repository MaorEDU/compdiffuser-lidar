#!/bin/bash
# One-time data setup for comp_diffuser_ball (run from this repo's root).
#
# The code tree is standalone; only DATA is shared with the parent ant repo /
# OGBench cache:
#   * OGBench datasets live in ~/.ogbench/data (download pointmaze-*-stitch-v0
#     via conda_env/ogb_dset_download.ipynb; nothing to link).
#   * data/ogb_maze/ev_probs: pointmaze problem sets ALREADY COPIED into this
#     repo (9 KB each).
#   * lidar_cache / lidar_grid are (re)built automatically on first use; if the
#     parent repo already built ball caches you can link them to skip that.
set -e
mkdir -p data/ogb_maze/lidar_cache data/ogb_maze/lidar_grid data/ogb_maze/lidar logs slurm_logs

PARENT="${1:-..}"
for d in lidar_cache lidar_grid; do
  if [ -d "$PARENT/data/ogb_maze/$d" ]; then
    for f in "$PARENT/data/ogb_maze/$d"/lidar_pointmaze-* "$PARENT/data/ogb_maze/$d"/pointmaze-*; do
      [ -e "$f" ] && ln -sf "$(readlink -f "$f")" "data/ogb_maze/$d/" && echo "linked $f"
    done
  fi
done
echo "setup done. Datasets expected at ~/.ogbench/data/pointmaze-{medium,large,giant}-stitch-v0*.npz"
