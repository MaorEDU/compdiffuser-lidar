"""
Grid-based LiDAR localization + probability mapping for the OGBench mazes.

This implements **Stage 2 (Algorithm & Analysis)** and the localization core of
**Stage 3 (Location Tracking)** of the project plan, on top of the (x, y, theta) reference
grid built by :mod:`diffuser.datasets.lidar.lidar_grid`.

What it provides
----------------
* **Probability mapping** -- turn a scan-distance metric into a likelihood
  ``L = exp(-0.5 (d / sigma)^2)`` (:func:`measurement_likelihood`).  This is the
  "translate uncertainty metrics into likelihood values" step.
* **LiDAR-scan comparison** -- per-beam RMS distance between an observed scan and every
  grid scan (:func:`scan_distance`).
* **Distance-difference map** -- distance from a *target* scan to every grid pose,
  reduced to a per-(x, y) map (:meth:`GridLocalizer.distance_map_to_target`).  This is
  the spec's "compare the target scan to scans from other locations -> a dedicated map
  of distance differences".
* **Single-scan localization** -- the maximum-likelihood pose for one scan, plus the
  full posterior belief (:meth:`GridLocalizer.localize_single`).
* **Sequence / motion-history localization** -- a Markov (histogram) filter that fuses a
  *sequence* of scans as the agent moves, which collapses the single-scan ambiguity of
  self-similar corridors (:meth:`GridLocalizer.localize_sequence`).  This is the
  SLAM-like, in-spec ("no x, y to the model") localization the diagnosis pointed to.
* **Distinctiveness / ambiguity map** -- per (x, y), how far the nearest *other* pose is
  in scan space; small == locally ambiguous (:meth:`GridLocalizer.distinctiveness_map`).

Pure numpy; no torch / mujoco.  The same module is imported by the closed-loop rollout
(for localization-guided plan re-ranking) and by the offline analysis CLIs.
"""

import numpy as np


# ---------------------------------------------------------------------------#
# scan comparison + probability mapping (module-level, reusable)
# ---------------------------------------------------------------------------#
def scan_distance(query, ref, chunk=4096):
    """Per-beam RMS distance between scans.

    Args:
        query: (Q, n_beams) or (n_beams,) observed scan(s), world units.
        ref:   (R, n_beams) reference scans (e.g. grid.free_scans).
        chunk: rows of ``ref`` processed per block (bounds peak memory).
    Returns:
        (Q, R) float32 per-beam RMS distances (``sqrt(mean_beam (q-r)^2)``).
    """
    q = np.atleast_2d(np.asarray(query, dtype=np.float32))
    r = np.asarray(ref, dtype=np.float32)
    Q, R = q.shape[0], r.shape[0]
    out = np.empty((Q, R), dtype=np.float32)
    for s in range(0, R, chunk):
        e = min(s + chunk, R)
        d = q[:, None, :] - r[None, s:e, :]          # (Q, <=chunk, nb)
        out[:, s:e] = np.sqrt((d ** 2).mean(-1))
    return out


def measurement_likelihood(dist, sigma, normalize=False):
    """Probability mapping: scan-distance -> likelihood.

    ``L = exp(-0.5 (dist / sigma)^2)`` -- a Gaussian measurement model where
    ``sigma`` is the expected per-beam RMS mismatch (world units) between a scan
    and the true cell's reference scan (sensor noise + grid quantization +
    model reconstruction error).

    Args:
        dist: array of per-beam RMS distances (any shape).
        sigma: measurement noise scale, world units. Larger == softer belief.
        normalize: if True, divide by the sum so the last-axis is a distribution.
    """
    like = np.exp(-0.5 * (np.asarray(dist, dtype=np.float64) / float(sigma)) ** 2)
    if normalize:
        s = like.sum(axis=-1, keepdims=True)
        like = np.where(s > 0, like / s, like)
    return like


# ---------------------------------------------------------------------------#
# GridLocalizer
# ---------------------------------------------------------------------------#
class GridLocalizer:
    """Localization + probability maps over a :class:`LidarGrid`.

    Args:
        grid: a built :class:`~diffuser.datasets.lidar.lidar_grid.LidarGrid`.
        sigma: measurement noise scale (world units) for the likelihood model.
            Default 1.0 wu (~1/4 cell) -- discriminative given inter-cell scan
            distances are typically 1-4 wu.
    """

    def __init__(self, grid, sigma=1.0):
        assert grid.free_scans is not None, 'grid must be built (call grid.build())'
        self.grid = grid
        self.sigma = float(sigma)
        # flat -> dense scatter indices
        self.rc = grid.free_rc                     # (M, 2)
        self.ti = grid.free_ti                     # (M,)
        self.M = grid.free_scans.shape[0]
        self.nb = grid.n_beams
        self.n_row, self.n_col, self.n_theta = grid.n_row, grid.n_col, grid.n_theta
        # free-cell mask on the dense (row, col, theta) lattice
        self._dense_free = np.zeros((self.n_row, self.n_col, self.n_theta), dtype=bool)
        self._dense_free[self.rc[:, 0], self.rc[:, 1], self.ti] = True

    # -- likelihood over the grid -----------------------------------------#
    def likelihood(self, scan):
        """(M,) measurement likelihood of one observed scan over all free poses."""
        d = scan_distance(scan, self.grid.free_scans)[0]          # (M,)
        return measurement_likelihood(d, self.sigma)

    def _to_dense(self, flat, fill=0.0):
        """Scatter a per-free-pose vector (M,) onto the dense (row,col,theta) grid."""
        dense = np.full((self.n_row, self.n_col, self.n_theta), fill, dtype=np.float64)
        dense[self.rc[:, 0], self.rc[:, 1], self.ti] = flat
        return dense

    # -- single-scan localization -----------------------------------------#
    def localize_single(self, scan, return_belief=False):
        """Maximum-likelihood pose for a single scan.

        Returns a dict with ``xy`` (2,), ``theta``, ``cell`` (i, j), ``rc``
        (row, col), ``pose_idx`` and the top-1 ``distance``. If ``return_belief``
        also returns the normalized posterior over free poses ``belief`` (M,) and
        the (x, y) marginal ``belief_xy`` (n_row, n_col).
        """
        d = scan_distance(scan, self.grid.free_scans)[0]
        k = int(np.argmin(d))
        x, y = self.grid.free_xy[k]
        out = dict(xy=np.array([x, y], dtype=np.float32),
                   theta=float(self.grid.free_theta[k]),
                   cell=self.grid.cell_of_xy(x, y),
                   rc=(int(self.rc[k, 0]), int(self.rc[k, 1])),
                   pose_idx=k, distance=float(d[k]))
        if return_belief:
            like = measurement_likelihood(d, self.sigma)
            belief = like / max(like.sum(), 1e-30)
            out['belief'] = belief
            out['belief_xy'] = self._to_dense(belief).sum(axis=2)   # marginalize theta
        return out

    def posterior_xy(self, scan):
        """(n_row, n_col) posterior over position (theta marginalized) for one scan."""
        like = self.likelihood(scan)
        like = like / max(like.sum(), 1e-30)
        return self._to_dense(like).sum(axis=2)

    # -- distance-difference map to a target ------------------------------#
    def distance_map_to_target(self, target_scan, reduce='min'):
        """Distance from a target scan to every grid pose -> per-(x, y) map.

        This is the spec's core analysis output: scan the target point, compare it
        to scans from every other location, and return a map of the differences.

        Args:
            target_scan: (n_beams,) the target's LiDAR scan.
            reduce: how to collapse the theta axis at each (x, y): 'min' (best
                orientation match, default), 'mean', or an int theta-index.
        Returns:
            dist_xy: (n_row, n_col) float32 map, NaN at wall cells.
            info: dict with the argmin pose (nearest look-alike) and its (x, y).
        """
        d = scan_distance(target_scan, self.grid.free_scans)[0]       # (M,)
        flat = d
        if isinstance(reduce, (int, np.integer)):
            dense = self._to_dense(d, fill=np.nan)
            dist_xy = dense[:, :, int(reduce)]
        else:
            dist_xy = self._reduce_theta(flat, reduce)
        k = int(np.argmin(d))
        info = dict(nearest_pose_idx=k,
                    nearest_xy=self.grid.free_xy[k].astype(np.float32),
                    nearest_theta=float(self.grid.free_theta[k]),
                    nearest_distance=float(d[k]))
        return dist_xy.astype(np.float32), info

    # -- sequence / motion-history localization ---------------------------#
    def localize_sequence(self, scans, motion_sigma_cells=1.0, theta_blur=1,
                          init_belief=None, return_beliefs=False):
        """Markov (histogram) filter over a *sequence* of scans.

        As the agent moves, each new scan updates a belief over all free poses; a
        motion (predict) step diffuses the belief so consecutive measurements must
        be explained by *nearby* poses. This fuses time -> the self-similar
        single-scan look-alikes get pruned because only the true corridor stays
        consistent across the whole sequence.

        Args:
            scans: (T, n_beams) sequence of observed scans (oldest -> newest).
            motion_sigma_cells: std of the per-step spatial diffusion, in *lattice
                steps* (predict-step motion model; ~expected move per scan). Larger
                = trusts motion less / lets the estimate jump further.
            theta_blur: half-width (in theta bins) of the orientation diffusion.
            init_belief: optional (M,) prior; default uniform over free poses.
            return_beliefs: also return the dense belief at every step.
        Returns:
            dict with ``map_xy`` (T, 2) MAP position per step, ``map_theta`` (T,),
            ``entropy`` (T,) belief entropy per step (nats), ``final_belief`` (M,),
            and (if requested) ``beliefs`` list of (row,col,theta) arrays.
        """
        scans = np.asarray(scans, dtype=np.float32)
        assert scans.ndim == 2 and scans.shape[1] == self.nb
        T = scans.shape[0]

        if init_belief is None:
            belief = self._dense_free.astype(np.float64)
        else:
            belief = self._to_dense(np.asarray(init_belief, dtype=np.float64))
        belief /= max(belief.sum(), 1e-30)

        kernel = _gauss_kernel1d(motion_sigma_cells)
        map_xy = np.zeros((T, 2), dtype=np.float32)
        map_theta = np.zeros(T, dtype=np.float32)
        entropy = np.zeros(T, dtype=np.float32)
        beliefs = []

        for t in range(T):
            if t > 0:
                belief = self._predict(belief, kernel, theta_blur)
            like = self.likelihood(scans[t])                      # (M,)
            dense_like = self._to_dense(like)                     # (row,col,theta)
            belief = belief * dense_like
            s = belief.sum()
            if s <= 1e-30:                                        # lost -> reinit from measurement
                belief = dense_like / max(dense_like.sum(), 1e-30)
            else:
                belief /= s
            # MAP estimate
            idx = np.unravel_index(np.argmax(belief), belief.shape)
            map_xy[t] = [self.grid.x_vals[idx[1]], self.grid.y_vals[idx[0]]]
            map_theta[t] = self.grid.theta_vals[idx[2]]
            p = belief[belief > 0]
            entropy[t] = float(-(p * np.log(p)).sum())
            if return_beliefs:
                beliefs.append(belief.copy())

        out = dict(map_xy=map_xy, map_theta=map_theta, entropy=entropy,
                   final_belief=belief[self.rc[:, 0], self.rc[:, 1], self.ti].copy())
        if return_beliefs:
            out['beliefs'] = beliefs
        return out

    # -- Viterbi (max-product) sequence decode ----------------------------#
    def _viterbi_neighbours(self, r_max_cells):
        """Predecessor structure on the free-pose lattice, cached per radius.

        Returns ``(NB, LOGD2)`` where ``NB[m, k]`` is the k-th pose that can
        precede pose ``m`` and ``LOGD2[m, k]`` is the squared world-unit distance
        of that transition (``inf`` for padding slots).
        """
        key = round(float(r_max_cells), 6)
        cache = getattr(self, '_vit_nb_cache', None)
        if cache is None:
            cache = self._vit_nb_cache = {}
        if key in cache:
            return cache[key]
        assert self.n_theta == 1, (
            'Viterbi decode assumes a yaw-fixed grid (n_theta == 1); the ball grid '
            f'is built that way but this one has n_theta={self.n_theta}')
        step = float(self.grid.x_vals[1] - self.grid.x_vals[0])
        # flat pose index at each lattice cell (-1 == wall)
        idx_map = np.full((self.n_row, self.n_col), -1, dtype=np.int64)
        idx_map[self.rc[:, 0], self.rc[:, 1]] = np.arange(self.M)
        R = int(np.floor(r_max_cells + 1e-9))
        offs = [(dr, dc) for dr in range(-R, R + 1) for dc in range(-R, R + 1)
                if (dr * dr + dc * dc) <= r_max_cells ** 2 + 1e-9]
        K = len(offs)
        NB = np.zeros((self.M, K), dtype=np.int64)
        D2 = np.full((self.M, K), np.inf, dtype=np.float64)
        rows, cols = self.rc[:, 0], self.rc[:, 1]
        for k, (dr, dc) in enumerate(offs):
            rr, cc = rows + dr, cols + dc
            ok = (rr >= 0) & (rr < self.n_row) & (cc >= 0) & (cc < self.n_col)
            cand = np.where(ok, idx_map[np.clip(rr, 0, self.n_row - 1),
                                        np.clip(cc, 0, self.n_col - 1)], -1)
            good = cand >= 0
            # self-transition padding: unreachable slots point at the pose itself
            # with inf cost, so they can never win the max
            NB[:, k] = np.where(good, cand, np.arange(self.M))
            D2[good, k] = ((dr * step) ** 2 + (dc * step) ** 2)
        cache[key] = (NB, D2)
        return NB, D2

    def localize_viterbi(self, scans, motion_sigma_cells=1.0, r_max_cells=3.0):
        """Most-likely *path* over the pose lattice for a sequence of scans.

        `localize_sequence` is a forward FILTER that takes the argmax of the
        marginal at each step **independently**, so nothing forces consecutive
        estimates to be reachable from one another. When the posterior is bimodal
        -- and 63.6% of mr12 ball poses have a look-alike >=8 wu away -- the MAP
        hops between modes and the decoded path teleports. Viterbi maximises over
        whole paths under the same Gaussian motion model, so an unreachable step
        cannot appear in the output by construction.

        Measured against the filter (2026-07-31, pure CPU): on 200 ground-truth
        windows median error 0.197 -> 0.167 wu, jump-free 79% -> 100%, and the
        scan-match residual *improves* 0.389 -> 0.372 wu/beam -- i.e. the
        continuous path explains the observations better, it is not smoothing
        information away. On 100 generated 888-frame plans: median cell-jumps
        5 -> 0 and residual 0.955 -> 0.806 wu/beam.

        Args:
            scans: (T, n_beams) sequence, oldest -> newest.
            motion_sigma_cells: std of the transition model in lattice steps,
                same units and default as ``localize_sequence``.
            r_max_cells: transitions beyond this lattice radius are forbidden.
                Must comfortably exceed the true per-step motion.
        Returns:
            dict with ``map_xy`` (T, 2), ``pose_idx`` (T,) and ``score``
            (log-likelihood of the winning path).
        """
        scans = np.asarray(scans, dtype=np.float32)
        assert scans.ndim == 2 and scans.shape[1] == self.nb, scans.shape
        T = scans.shape[0]
        step = float(self.grid.x_vals[1] - self.grid.x_vals[0])
        sigma_wu = max(float(motion_sigma_cells) * step, 1e-6)

        NB, D2 = self._viterbi_neighbours(r_max_cells)
        logtrans = -0.5 * D2 / (sigma_wu ** 2)                # (M, K), -inf padding

        # (T, M) log measurement likelihood
        d = scan_distance(scans, self.grid.free_scans).astype(np.float64)
        ll = -0.5 * (d / self.sigma) ** 2

        if T == 1:
            k = int(np.argmax(ll[0]))
            return dict(map_xy=self.grid.free_xy[[k]].astype(np.float32),
                        pose_idx=np.array([k]), score=float(ll[0, k]))

        delta = ll[0].copy()
        bp = np.zeros((T, self.M), dtype=np.int32)
        ar = np.arange(self.M)
        for t in range(1, T):
            cand = delta[NB] + logtrans                       # (M, K)
            k = np.argmax(cand, axis=1)
            bp[t] = NB[ar, k]
            delta = cand[ar, k] + ll[t]

        path = np.zeros(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 1, 0, -1):
            path[t - 1] = bp[t, path[t]]
        return dict(map_xy=self.grid.free_xy[path].astype(np.float32),
                    pose_idx=path, score=float(delta.max()))

    def _predict(self, belief, kernel, theta_blur):
        """Motion (predict) step: diffuse belief spatially + over theta, re-mask free."""
        b = belief
        # separable spatial blur (rows then cols), per theta slice
        b = _conv1d_axis(b, kernel, axis=0)
        b = _conv1d_axis(b, kernel, axis=1)
        if theta_blur and theta_blur > 0:
            tk = _gauss_kernel1d(float(theta_blur))
            b = _conv1d_axis(b, tk, axis=2, circular=True)
        b = b * self._dense_free                                  # stay in free space
        s = b.sum()
        return b / s if s > 1e-30 else belief

    # -- distinctiveness / ambiguity map ----------------------------------#
    def distinctiveness_map(self, min_cell_gap=3.0, reduce='mean'):
        """Per (x, y): scan-distance to the nearest *far* pose (localizability).

        For every free (x, y), take (over its theta poses) the smallest per-beam
        RMS distance to any pose at least ``min_cell_gap`` cells away. **Small ==
        ambiguous** (a distant look-alike exists); large == locally unique.

        Returns:
            (n_row, n_col) float32 map (NaN at wall cells).
        """
        fs = self.grid.free_scans                                # (M, nb)
        fxy = self.grid.free_xy.astype(np.float64)
        mu = self.grid.maze_unit
        nearest = np.full(self.M, np.inf, dtype=np.float32)
        CH = 1024
        for s in range(0, self.M, CH):
            e = min(s + CH, self.M)
            d = np.sqrt(((fs[s:e, None, :] - fs[None, :, :]) ** 2).mean(-1))   # (blk, M)
            cell_d = np.sqrt(((fxy[s:e, None, :] - fxy[None, :, :]) ** 2).sum(-1)) / mu
            d[cell_d < min_cell_gap] = np.inf                    # ignore self + neighbours
            nearest[s:e] = d.min(1)
        return self._reduce_theta(nearest, reduce).astype(np.float32)

    def _reduce_theta(self, flat, reduce='min'):
        """Reduce a per-free-pose vector (M,) over the theta axis to a (row, col)
        map, returning NaN for (x, y) columns that are entirely wall (no warnings)."""
        free_any = self._dense_free.any(axis=2)                   # (row, col)
        if reduce == 'min':
            dense = self._to_dense(flat, fill=np.inf)
            out = dense.min(axis=2)
        elif reduce == 'mean':
            cnt = self._dense_free.sum(axis=2)
            dense = self._to_dense(flat, fill=0.0)
            out = dense.sum(axis=2) / np.maximum(cnt, 1)
        else:
            raise ValueError(f'reduce={reduce!r}')
        return np.where(free_any, out, np.nan)


# ---------------------------------------------------------------------------#
# small separable-blur helpers (no scipy dependency)
# ---------------------------------------------------------------------------#
def _gauss_kernel1d(sigma, truncate=3.0):
    sigma = max(float(sigma), 1e-6)
    rad = max(int(truncate * sigma + 0.5), 1)
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _conv1d_axis(arr, kernel, axis, circular=False):
    """Convolve ``arr`` with a 1-D ``kernel`` along ``axis`` (zero-pad or circular)."""
    arr = np.asarray(arr, dtype=np.float64)
    rad = (len(kernel) - 1) // 2
    arr_m = np.moveaxis(arr, axis, 0)
    out = np.zeros_like(arr_m)
    n = arr_m.shape[0]
    for off, w in zip(range(-rad, rad + 1), kernel):
        if w == 0:
            continue
        if circular:
            out += w * np.roll(arr_m, -off, axis=0)
        else:
            src0, src1 = max(0, off), min(n, n + off)
            dst0, dst1 = max(0, -off), min(n, n - off)
            out[dst0:dst1] += w * arr_m[src0:src1]
    return np.moveaxis(out, 0, axis)
