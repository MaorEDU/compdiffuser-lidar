"""WITHDRAWN 2026-08-30 -- DO NOT USE. Kept as evidence, not as a code path.

The scoped deliverable is a SIMPLE LiDAR ball that navigates on an observation history
window plus replanning. A new decoding algorithm is outside that scope, so
`LIDAR_DECODE=viterbi` must not be set in the bench or anywhere else. The shipped
default is and remains `argmin`; nothing selects this unless a caller asks for it
explicitly. See RETRAIN_2026-08-29.md section 7.0.

This module is retained because it is the measurement that closed the last open axis on
G2a `jump_frac` (see section 6 and 7 of that document) -- a durable result about the
system, even though the remedy is out of scope.

--------------------------------------------------------------------------------
Globally-consistent scan -> (x, y) decode: beam-Viterbi over the free-space graph.

Why this exists
---------------
The two shipped decoders both fail G2a's `jump_frac <= 0.01`, and they fail it for the
same structural reason measured in RETRAIN_2026-08-29 §6:

* ``argmin`` is **memoryless**. Each frame independently takes the nearest grid pose, so
  in a self-similar maze two near-identical scans can resolve to poses six cells apart.
  Measured: the two segment views of a shared overlap agree to p50 2.77 in scan distance
  (a 1-cell separation scores 4.88) yet decode 24.75 wu = 6.19 cells apart.
* ``filter`` carries a belief, but its motion model is a **Gaussian on the lattice**,
  which is a statement about Euclidean displacement. It cannot express "you must stay in
  the corridor you are already in", because a wall is not far away in Euclidean terms --
  it is far away in *connectivity* terms. Sweeping its sigma traces a monotone trade-off
  with no crossing point: sigma=1 gives jump 0.0140 / anchored 0.550, sigma=8 gives
  0.0267 / 0.825. No setting passes both halves of G2.

This decoder keeps the same emission model (scan distance to every free grid pose) and
replaces the motion model with a **hard connectivity constraint**: consecutive decoded
poses must be joined by a straight segment that lies entirely in free space and is no
longer than ``max_step``. Aliased candidates in an unreachable corridor are then not
merely unlikely, they are *inadmissible*, and the path that survives is the cheapest
globally consistent explanation of the whole scan sequence rather than the per-frame
nearest neighbour.

The search is a beam Viterbi: exact DP over the ``beam`` best-emission states per frame.
Full Viterbi over all 4409 poses is affordable too (``beam=0``) but ~20x slower for no
measured gain, because the true pose is essentially always inside a wide beam.

Nothing here is maze- or range-specific: the free-space graph is derived from the grid
that the planner already builds and caches.
"""
import numpy as np

from .lidar_localization import scan_distance

__all__ = ['FreeSpaceGraph', 'viterbi_decode_xy']


class FreeSpaceGraph:
    """Neighbour lists over a :class:`LidarGrid`'s free poses, with wall-crossing removed.

    Built once per (grid, max_step) and reused by every decode. The wall check is what
    makes this a *connectivity* neighbourhood rather than a Euclidean one: two poses in
    parallel corridors are close in distance and unreachable in fact.
    """

    def __init__(self, grid, max_step=3.0, n_los=9):
        from scipy.spatial import cKDTree

        self.grid = grid
        self.max_step = float(max_step)
        xy = np.asarray(grid.free_xy, dtype=np.float64)
        self.xy = xy
        self.M = len(xy)

        tree = cKDTree(xy)
        pairs = tree.query_ball_point(xy, r=self.max_step)

        ## A straight move between two free lattice points can still cut a wall corner,
        ## so sample the segment and require every sample to be in a free cell. Walls
        ## here are one maze cell (4 wu) thick and max_step defaults below that, so this
        ## mostly matters for diagonal clips -- but "mostly" is not a guarantee.
        scanner = grid.scanner
        ts = np.linspace(0.0, 1.0, int(n_los))[1:-1][:, None]     # interior samples only
        nbr, cnt = [], np.zeros(self.M, dtype=np.int64)
        for p in range(self.M):
            q = np.asarray(pairs[p], dtype=np.int64)
            if len(q) == 0:
                nbr.append(q)
                continue
            seg = xy[p][None, None, :] * (1.0 - ts)[None] + xy[q][:, None, :] * ts[None]
            ii, jj = scanner.world_to_cell(seg[..., 0].ravel(), seg[..., 1].ravel())
            ok = ~scanner._is_wall(ii, jj).reshape(len(q), -1).any(axis=1)
            q = q[ok]
            nbr.append(q)
            cnt[p] = len(q)

        ## pack to a rectangular (M, K) table with -1 padding, so the DP is one gather
        self.K = int(cnt.max()) if self.M else 0
        self.nbr = np.full((self.M, self.K), -1, dtype=np.int64)
        for p, q in enumerate(nbr):
            self.nbr[p, :len(q)] = q
        self.valid = self.nbr >= 0
        ## transition cost: Euclidean length of the move (0 for the padded slots)
        safe = np.where(self.valid, self.nbr, 0)
        self.step_len = np.linalg.norm(xy[safe] - xy[:, None, :], axis=2)
        self.step_len[~self.valid] = 0.0

    def __repr__(self):
        return (f'FreeSpaceGraph(M={self.M}, max_step={self.max_step:g}, K={self.K}, '
                f'mean_deg={self.valid.sum(1).mean():.1f})')


def viterbi_decode_xy(scans, grid, graph, sigma_e=1.0, w_step=0.0,
                      beam=256, start_xy=None, start_tol=None,
                      end_xy=None, end_tol=None):
    """(n, n_beams) scans -> (n, 2) poses, globally consistent by construction.

    Args:
        scans: the (already strided) scan sequence to decode.
        grid: the built :class:`LidarGrid` (supplies ``free_scans`` / ``free_xy``).
        graph: a :class:`FreeSpaceGraph` over the same grid.
        sigma_e: emission scale. Costs are ``scan_distance / sigma_e``, so this sets how
            many world units of scan mismatch are worth one world unit of motion.
        w_step: per-move penalty weight on the step length. 0 leaves the constraint hard
            and the choice among admissible moves purely to the emission.
        beam: keep only this many best-emission states per frame (0 = exact, all states).
        start_xy: optional anchor for frame 0 -- the decoder's own estimate of where it
            currently is. NOT privileged: at episode start this is the initial state, and
            mid-episode the rollout supplies whatever its ladder rung gives it (the
            belief estimate at L3, the env pose at L1). Pass None to leave frame 0 free.
        start_tol: radius for the above; defaults to ``graph.max_step``.
        end_xy: optional anchor for the LAST frame. The intended value is the per-frame
            argmin decode of the final scan -- i.e. exactly what the memoryless decoder
            already concludes from the goal scan the planner was conditioned on, and
            which G2b shows is right 97.5% of the time. This is deliberately NOT the
            benchmark's `goal_pos`: using that would be privileged information the L3
            rung does not grant. The anchor exists because connectivity and endpoint
            accuracy are carried by different evidence -- the graph carries the route,
            the goal scan carries the destination -- and Viterbi alone will trade the
            second away for the first.
        end_tol: radius for the end anchor; defaults to ``graph.max_step``.

    Returns:
        (n, 2) float64 poses, one per input frame.
    """
    scans = np.asarray(scans, dtype=np.float32)
    n = len(scans)
    M = graph.M
    ## (n, M) emission cost -- the same quantity argmin takes a per-row min of
    emis = scan_distance(scans, grid.free_scans).astype(np.float64) / float(sigma_e)

    if beam and beam < M:
        ## restrict each frame to its own best-emission candidates. The DP stays exact
        ## on that support; a state outside every frame's beam could not have won.
        cand = np.argpartition(emis, beam - 1, axis=1)[:, :beam]
        cand.sort(axis=1)
    else:
        cand = np.tile(np.arange(M, dtype=np.int64), (n, 1))
    B = cand.shape[1]

    if start_xy is not None:
        tol = float(graph.max_step if start_tol is None else start_tol)
        d0 = np.linalg.norm(graph.xy[cand[0]] - np.asarray(start_xy, float)[None, :2], axis=1)
        if (d0 <= tol).any():
            ## anchor frame 0 without dropping states: an out-of-tolerance start is
            ## made expensive rather than impossible, so a bad anchor cannot make the
            ## whole sequence undecodable.
            emis0 = emis[0, cand[0]] + np.where(d0 <= tol, 0.0, 1e3)
        else:
            emis0 = emis[0, cand[0]]
    else:
        emis0 = emis[0, cand[0]]

    ## ---- forward pass -------------------------------------------------------
    ## cost[b] = best total cost of a legal path ending at cand[t][b]
    cost = emis0.copy()
    back = np.zeros((n, B), dtype=np.int32)

    ## position of each state within the previous frame's candidate list, for gathering
    for t in range(1, n):
        prev, cur = cand[t - 1], cand[t]
        ## admissible predecessors of every current state, via the free-space graph
        nb = graph.nbr[cur]                      # (B, K) global ids, -1 padded
        ok = graph.valid[cur]
        ## map global predecessor ids -> index into `prev` (or -1 if not in the beam)
        lut = np.full(M, -1, dtype=np.int64)
        lut[prev] = np.arange(len(prev))
        pi = np.where(ok, lut[np.where(ok, nb, 0)], -1)     # (B, K)
        good = pi >= 0
        cand_cost = np.where(
            good, cost[np.where(good, pi, 0)] + float(w_step) * graph.step_len[cur],
            np.inf)
        k = np.argmin(cand_cost, axis=1)
        best = cand_cost[np.arange(len(cur)), k]
        ## A state with no admissible predecessor in the beam is a dead end. Allow it to
        ## restart at a large penalty instead of propagating inf, so a single unlucky
        ## frame cannot make the whole plan undecodable -- the penalty guarantees such a
        ## restart is never preferred to any genuinely connected path.
        dead = ~np.isfinite(best)
        best = np.where(dead, cost.min() + 1e3, best)
        back[t] = np.where(dead, -1, pi[np.arange(len(cur)), k])
        cost = best + emis[t, cur]

    ## ---- end anchor ---------------------------------------------------------
    if end_xy is not None:
        tol = float(graph.max_step if end_tol is None else end_tol)
        dn = np.linalg.norm(graph.xy[cand[n - 1]] - np.asarray(end_xy, float)[None, :2],
                            axis=1)
        if (dn <= tol).any():
            ## soft, for the same reason as the start anchor: an anchor that no
            ## surviving path can reach must not make the plan undecodable.
            cost = cost + np.where(dn <= tol, 0.0, 1e3)

    ## ---- backtrace ----------------------------------------------------------
    out = np.zeros((n, 2), dtype=np.float64)
    b = int(np.argmin(cost))
    for t in range(n - 1, -1, -1):
        out[t] = graph.xy[cand[t][b]]
        nb = int(back[t][b])
        if t and nb < 0:                       # restart point: fall back to this frame's
            nb = int(np.argmin(emis[t - 1, cand[t - 1]]))   # own best-emission state
        b = nb
    return out
