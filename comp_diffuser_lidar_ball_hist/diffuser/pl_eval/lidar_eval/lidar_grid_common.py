"""Shared geometry helpers for the LiDAR eval tooling and the closed-loop executor.

Only line-of-sight lives here. The previous fork's ``make_waypoints`` (BFS-derived
per-segment aim poses handed to the composed planner) is DELIBERATELY ABSENT:
comp_diffuser_ball_v2 conditions the planner on the start scan, the goal scan and the
observation history, and on nothing map-derived.
"""

import numpy as np


def los_clear(scanner, p0, pts, margin=0.6, step=0.5):
    """Which of ``pts`` can be driven at in a STRAIGHT LINE from ``p0``?

    Returns a bool array of shape ``(len(pts),)``. A target is "clear" when the
    segment ``p0 -> pt``, and the same segment offset by +-``margin`` wu
    perpendicular, cross no wall cell. The parallel offsets are what keep the
    ball (which has a radius, unlike the point that the cell test assumes) from
    being sent to scrape a corner diagonally.

    Why this exists. The executor's pursuit target is a point on the DECODED plan,
    and the decoded plan carries residual cell jumps. When the cursor lands just
    before a jump the target sits across a wall; the PID is pure proportional
    (kp=5, kd=ki=0), so it presses into that wall and the ball stops dead. In the
    previous fork this was measured directly: 7/20 episodes sat at a BIT-IDENTICAL
    position at step 888 and step 1776 -- an entire plan spent pinned against one
    wall -- and all 7 failed. Aiming only at points the ball can actually reach in a
    straight line breaks that at its origin.

    Pure numpy over ``scanner.maze_map`` -- the same map that already backs the goal
    scan and the decode grid, so no new privileged information enters. It IS map-
    derived, though, and that is part of the asymmetry this repo owes any external
    comparison (see BALLV2.md, house rules).
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if not len(pts):
        return np.zeros(0, dtype=bool)
    d = pts - p0                                             # (N, 2)
    L = np.linalg.norm(d, axis=1)
    n_s = max(int(np.ceil(L.max() / max(step, 1e-6))) + 1, 2)
    t = np.linspace(0.0, 1.0, n_s)[None, :, None]            # (1, S, 1)
    base = p0[None, None, :] + t * d[:, None, :]             # (N, S, 2)
    ## unit perpendicular per target (zero-length targets are trivially clear)
    perp = np.stack([-d[:, 1], d[:, 0]], axis=1)
    nrm = np.linalg.norm(perp, axis=1, keepdims=True)
    perp = np.divide(perp, nrm, out=np.zeros_like(perp), where=nrm > 1e-9)
    ok = np.ones(len(pts), dtype=bool)
    for off in (0.0, margin, -margin):
        q = base + off * perp[:, None, :]
        i, j = scanner.world_to_cell(q[..., 0], q[..., 1])
        ok &= ~scanner._is_wall(i, j).any(axis=1)
    return ok
