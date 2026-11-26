"""
LiDAR dataset entry point for the diffusion planner.

The LiDAR transform itself happens inside ``ogb_sequence_dataset`` (see
``ogb_utils.py``) whenever ``dataset_config`` carries a ``'lidar_config'`` entry, so
this class inherits all of the well-tested windowing / padding / normalization logic
of ``OgB_SeqDataset_V2`` unchanged.

What it adds is the **observation history** the history-conditioned planner needs:
alongside the ``horizon``-frame training window it returns the ``hist_len`` frames
that immediately precede the window in the same episode. That is exactly what the
closed-loop rollout can supply at a replan -- the scans the ball has already observed
-- so training and rollout see the same object.

Where the history comes from
----------------------------
The buffer is built with ``pad_option_2='buf'`` / ``pad_type='first'`` /
``extra_pad=64``, i.e. every episode is already left-padded with 64 copies of its
first frame. So for a window starting at ``start`` the history is simply
``normed_observations[path_ind, start-hist_len : start]``, and windows that start
inside (or before) the pad get the oldest available frame repeated -- which is the
same "no history yet" distribution the rollout is in during an episode's first
``hist_len`` steps. Requires ``hist_len <= extra_pad``.
"""

from collections import namedtuple

import numpy as np

import diffuser.utils as utils
from diffuser.datasets.lidar.hist_window import left_pad_history
from diffuser.datasets.ogb_dset.ogb_dataset_v2 import OgB_SeqDataset_V2


## Batch_v1 + the history window. Kept as a distinct type so a consumer that has
## not been updated fails loudly on the arity instead of silently mis-unpacking.
Batch_v2 = namedtuple('Batch_v2', 'obs_trajs act_trajs conditions hist_trajs')


def _check_lidar_config(dataset_config):
    lidar_config = dataset_config.get('lidar_config', None)
    assert lidar_config is not None, (
        'LiDAR dataset requires dataset_config["lidar_config"]; '
        'use the regular OgB_SeqDataset_V2 class for raw (x, y) observations.')
    utils.print_color(f'[lidar dataset] lidar_config={lidar_config}', c='c')
    return lidar_config


class OgB_Lidar_SeqDataset_V1(OgB_SeqDataset_V2):
    """Diffusion-planner dataset whose observations are simulated LiDAR scans.

    ``dataset_config['hist_len']``:
        0 (default) -> behaves exactly like ``OgB_SeqDataset_V2`` and returns a
        3-field batch, so a no-history config is bit-identical to the release path.
        >0          -> returns ``Batch_v2`` with a ``(hist_len, dim)`` history window.
    """

    def __init__(self, *args, dataset_config={}, **kwargs):
        _check_lidar_config(dataset_config)
        self.hist_len = int(dataset_config.get('hist_len', 0) or 0)
        super().__init__(*args, dataset_config=dataset_config, **kwargs)

        if self.hist_len > 0:
            assert self.pad_option_2 == 'buf', (
                'history slicing assumes the buffer-level left pad '
                f"(pad_option_2='buf'); got {self.pad_option_2!r}")
            extra_pad = int(dataset_config.get('extra_pad', 0))
            assert self.hist_len <= extra_pad, (
                f'hist_len={self.hist_len} must be <= extra_pad={extra_pad}, else a '
                f'window starting at the episode boundary has no frames to look back on '
                f'and the history would be pure padding for a large share of the data.')
            utils.print_color(
                f'[lidar dataset] history conditioning ON: hist_len={self.hist_len} '
                f'frames (extra_pad={extra_pad})', c='c')

    def get_hist(self, path_ind, start, obs_trajs):
        """The ``hist_len`` normalized frames immediately before ``start``.

        Left-padded by repeating the oldest available frame when the window starts
        within the first ``hist_len`` frames of the (already padded) episode. Built
        through ``left_pad_history`` so it is the same object the rollout assembles
        from observed scans."""
        lo = max(0, int(start) - self.hist_len)
        frames = self.fields.normed_observations[path_ind, lo:int(start)]
        return left_pad_history(frames, self.hist_len, fill=obs_trajs[0])

    def __getitem__(self, idx, eps=1e-4):
        batch = super().__getitem__(idx, eps=eps)
        if self.hist_len <= 0:
            return batch
        path_ind, start, _end = self.indices[idx % self.len_indices]
        hist = self.get_hist(path_ind, start, batch.obs_trajs)
        return Batch_v2(batch.obs_trajs, batch.act_trajs, batch.conditions, hist)
