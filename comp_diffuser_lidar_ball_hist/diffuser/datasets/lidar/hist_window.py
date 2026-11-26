"""One definition of "the observation history", shared by training and rollout.

The history window has to mean the *same thing* on both sides or the history branch
is trained on one distribution and evaluated on another -- silently, and in a way that
looks like a model failure rather than a plumbing bug. So both the dataset
(``OgB_Lidar_SeqDataset_V1.get_hist``) and the closed-loop rollout
(``OgB_Stgl_Sml_Lidar_Ball_MazeEnvPlanner_V1``) build it through this function.

The convention: the window is the ``hist_len`` frames immediately BEFORE the current
one, oldest first, and when there are not enough of them the oldest available frame is
repeated to fill the front. That matches the dataset's own ``pad_type='first'`` left
padding, and it matches the agent's real situation during an episode's first
``hist_len`` steps -- it genuinely has no past, and the best statement of that is "I
have been where I started".
"""

import numpy as np


def left_pad_history(frames, hist_len, fill=None):
    """``frames`` (K, D), oldest first -> (hist_len, D), oldest first.

    K > hist_len  -> keep the last hist_len frames.
    K < hist_len  -> repeat ``frames[0]`` at the front until the window is full.
    K == 0        -> repeat ``fill`` for the whole window (``fill`` is then required).
    """
    hist_len = int(hist_len)
    assert hist_len > 0, hist_len
    frames = np.asarray(frames, dtype=np.float32)
    if frames.size == 0:
        assert fill is not None, 'empty history needs an explicit fill frame'
        frames = np.asarray(fill, dtype=np.float32).reshape(1, -1)
    assert frames.ndim == 2, frames.shape
    if len(frames) >= hist_len:
        return np.ascontiguousarray(frames[-hist_len:])
    pad = np.repeat(frames[:1], hist_len - len(frames), axis=0)
    return np.ascontiguousarray(np.concatenate([pad, frames], axis=0))
