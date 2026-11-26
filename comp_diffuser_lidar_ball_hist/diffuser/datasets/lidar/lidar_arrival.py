"""Live LiDAR self-localization for the closed-loop ball rollout.

One predict+update histogram-filter step per observed scan, exposing the current MAP
cell/belief -- "where the ball thinks it is". Pure numpy, no torch; uses the same
``(x, y, theta)`` reference grid as the offline localization work (``lidar_grid.py`` /
``lidar_localization.py``). For the ball the grid is built with ``n_theta=1`` (the point
mass has no heading), so the theta axis is length 1 throughout.

NOTE: ``LidarArrivalDetector`` (the goal-cell arrival / verify-and-hop rule of the
previous fork) is DELIBERATELY ABSENT. comp_diffuser_ball_v2 is the vanilla LiDAR
baseline: episodes end on the env's own budget, and success comes from the env, never
from a LiDAR self-report.
"""


import numpy as np

from .lidar_localization import GridLocalizer, _gauss_kernel1d


class LiveLocalizer:
    """Online histogram filter over the (x, y, theta) grid.

    Call :meth:`update` once per observed scan. Maintains a belief over all free poses
    (Markov/Bayes filter: predict = spatial+theta diffusion, update = measurement
    likelihood), and returns the MAP cell — the ant's LiDAR-decoded position.
    """

    def __init__(self, loc: GridLocalizer, motion_sigma_cells=1.0, theta_blur=1,
                 eps_mix=0.0):
        """
        loc: a GridLocalizer over the reference grid. For the ball the grid is built
            with ``n_theta=1``, so ``theta_blur`` is inert.
        motion_sigma_cells: spatial diffusion of the predict step, in grid cells per
            observed scan. The ball moves at most 0.2 wu/step on a 0.5 wu lattice.
        eps_mix: share of the free-space UNIFORM mixed into the belief after each
            predict step. ``0.0`` is off and is bit-identical to every run before
            2026-09-04.

            Why it exists: ``loc_err_p50`` measured over bench100 is **bimodal**, not
            noisy -- 0.25 wu on 86% of episodes and ~45 wu on 14%, with nothing in
            between (RETRAIN 8.7). That is a discrete wrong-hypothesis LATCH, and the
            update below is why: it is a pure product, so once the true pose's mass
            underflows, no later scan can bring it back however well it matches. A
            uniform floor is the standard fix for exactly this failure in a histogram
            filter, and it costs the good regime nothing at 1e-3 (the diffusion kernel
            already spreads more mass than that between steps).

            It matters for goal capture specifically: capture homes on this estimate,
            so a latched belief homes to the wrong place.

        The estimate is the raw MAP lattice point. Sub-cell (belief-weighted centroid)
        interpolation of the previous fork is DELIBERATELY ABSENT: it was measured
        closed-loop and was negative -- it predicted 0.37 wu of error offline and
        delivered 6.22 wu in the loop, because damping the estimate feeds back into the
        actions, hence the trajectory, hence the scan sequence the filter conditions on.
        The same applies to the low-pass filter on the PID's position input, which this
        repo pins at 0.
        """
        self.loc = loc
        self.theta_blur = int(theta_blur)
        self._kernel = _gauss_kernel1d(float(motion_sigma_cells))
        self.eps_mix = float(eps_mix)
        assert 0.0 <= self.eps_mix < 1.0, self.eps_mix
        self.reset()

    def reset(self, init_belief=None):
        if init_belief is None:
            b = self.loc._dense_free.astype(np.float64).copy()
        else:
            b = self.loc._to_dense(np.asarray(init_belief, dtype=np.float64))
        self.belief = b / max(b.sum(), 1e-30)
        self.n = 0
        self.est_cell = None
        self.est_xy = None
        self.est_theta = None

    def update(self, scan):
        """One filter step. Returns dict(est_cell, est_xy, est_theta, entropy, conf)."""
        scan = np.asarray(scan, dtype=np.float32)
        if self.n > 0:
            self.belief = self.loc._predict(self.belief, self._kernel, self.theta_blur)
            if self.eps_mix > 0.0:
                ## uniform floor over the FREE poses only -- mixing in walls would
                ## put belief mass where the ball cannot be.
                u = self.loc._dense_free.astype(np.float64)
                u /= max(u.sum(), 1e-30)
                self.belief = (1.0 - self.eps_mix) * self.belief + self.eps_mix * u
        like = self.loc._to_dense(self.loc.likelihood(scan))
        b = self.belief * like
        s = b.sum()
        self.belief = (b / s) if s > 1e-30 else like / max(like.sum(), 1e-30)
        idx = np.unravel_index(int(np.argmax(self.belief)), self.belief.shape)
        x = float(self.loc.grid.x_vals[idx[1]]); y = float(self.loc.grid.y_vals[idx[0]])
        self.est_theta = float(self.loc.grid.theta_vals[idx[2]])
        self.est_cell = self.loc.grid.cell_of_xy(x, y)   # DISCRETE MAP cell, always
        self.est_xy = (x, y)
        self.n += 1
        p = self.belief[self.belief > 0]
        entropy = float(-(p * np.log(p)).sum())
        conf = float(self.belief.max())                  # posterior mass at the MAP pose
        return dict(est_cell=self.est_cell, est_xy=self.est_xy, est_theta=self.est_theta,
                    entropy=entropy, conf=conf)
