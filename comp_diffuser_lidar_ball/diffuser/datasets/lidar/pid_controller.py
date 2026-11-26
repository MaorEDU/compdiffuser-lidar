"""2-D PID waypoint controller for the OGBench pointmaze ("ball") agent.

The motion controller (BALL.md Phase 4, Arm A): the positional
coordinates (x, y) are governed by a PID law on the waypoint error,

    u = clip( Kp * e  +  Kd * (v_ref - v_lpf)  +  Ki * I[e],  -1, 1 )

with e = (waypoint - position). The OGBench point env integrates
``pos += ~0.2 * clip(u)`` per control step, so:

  * ``Kp=5, Kd=0, Ki=0, pos_lpf=0`` reproduces the upstream repo's proven P-law
    (``act_pred_nm = (goal_cur - obs_cur) * 5`` in ogb_stgl_sml_planner_v1.py,
    'pd_ogb' branch) **bit-exactly** -- that is the safe default.

  * ``pos_lpf`` (EMA smoothing of the *measured position*, beta in [0, 1)) is
    the anti-jitter tool for the L3 belief-PID tier. MEASURED FINDING (see the
    selftest): the point env is a FIRST-ORDER integrator -- a derivative term
    on a noisy position estimate does NOT damp noise-driven jitter (it
    differentiates the noise), while an EMA on the position cuts settled
    jitter ~2.4x at beta=0.7. Use ``pos_lpf~0.7`` when the position comes from
    the localization belief; leave 0 for the true-position tiers.

  * ``Kd`` is kept for completeness (velocity is LPF'd before use); expect it
    to stay 0 for this plant.

  * ``Ki>0`` is the designated fix for the "stops just short of the success
    radius" failure class (steady-state undershoot near the goal). It is only
    integrated inside ``ki_zone`` world-units of the target and is anti-windup
    clamped, so it cannot destabilize the long-range approach.

Velocity convention: displacement per env step (v_t = p_t - p_{t-1}), matching
the dataset's finite-difference velocities (ogb_lidar_utils) -- NO wall-clock
dt anywhere, so offline and online quantities are identical by construction.

Pure numpy; unit-tested via ``python -m diffuser.datasets.lidar.pid_controller``
(the selftest entry point below, no env needed).
"""

import numpy as np


class PID2D:
    """Stateful 2-D PID controller (one instance per episode; call reset())."""

    def __init__(self, kp=5.0, kd=0.0, ki=0.0, i_clamp=1.0, ki_zone=2.0,
                 pos_lpf=0.0, vel_lpf=0.7, out_clip=1.0):
        """
        Args:
            kp, kd, ki: gains (kp=5/kd=0/ki=0/pos_lpf=0 == upstream P-law).
            i_clamp: anti-windup clamp on each component of the error integral.
            ki_zone: integrate only when ||e|| <= ki_zone world-units (the
                last-mile zone); outside it the integral decays so a long
                approach cannot wind it up.
            pos_lpf: EMA coefficient on the measured position (0 = raw). Set
                ~0.7 when the position is a noisy estimate (belief-PID tier).
            vel_lpf: EMA coefficient on the finite-difference velocity used by
                the D-term (only relevant when kd > 0).
            out_clip: symmetric action clip (OGBench point actions in [-1, 1]).
        """
        self.kp = float(kp)
        self.kd = float(kd)
        self.ki = float(ki)
        self.i_clamp = float(i_clamp)
        self.ki_zone = float(ki_zone)
        self.pos_lpf = float(pos_lpf)
        self.vel_lpf = float(vel_lpf)
        self.out_clip = float(out_clip)
        self.reset()

    def reset(self):
        self._integ = np.zeros(2, dtype=np.float64)
        self._prev_pos = None
        self._pos_f = None
        self._vel_f = np.zeros(2, dtype=np.float64)

    @property
    def filtered_pos(self):
        """The position the controller actually acted on last step (or None)."""
        return None if self._pos_f is None else self._pos_f.copy()

    def step(self, pos, target, vel=None, target_vel=None):
        """One control step.

        Args:
            pos: (2,) measured position (true or estimated, world units).
            target: (2,) waypoint position (world units).
            vel: (2,) measured velocity in displacement/step. None -> finite
                difference of the (filtered) positions passed to consecutive
                step() calls (zero on the first call).
            target_vel: (2,) waypoint velocity (feed-forward). None -> 0.
        Returns:
            u: (2,) action, clipped to [-out_clip, out_clip].
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(2)
        target = np.asarray(target, dtype=np.float64).reshape(2)

        ## position EMA (the belief-jitter tool; identity when pos_lpf == 0)
        if self._pos_f is None or self.pos_lpf <= 0.0:
            self._pos_f = pos.copy()
        else:
            self._pos_f = self.pos_lpf * self._pos_f + (1.0 - self.pos_lpf) * pos
        p = self._pos_f
        e = target - p

        if vel is None:
            v_raw = np.zeros(2) if self._prev_pos is None else p - self._prev_pos
        else:
            v_raw = np.asarray(vel, dtype=np.float64).reshape(2)
        self._prev_pos = p.copy()
        self._vel_f = self.vel_lpf * self._vel_f + (1.0 - self.vel_lpf) * v_raw
        v_ref = np.zeros(2) if target_vel is None else \
            np.asarray(target_vel, dtype=np.float64).reshape(2)

        ## last-mile-only integral with anti-windup (decays outside the zone)
        if self.ki > 0.0:
            if float(np.linalg.norm(e)) <= self.ki_zone:
                self._integ = np.clip(self._integ + e, -self.i_clamp, self.i_clamp)
            else:
                self._integ *= 0.9
        u = self.kp * e + self.kd * (v_ref - self._vel_f) + self.ki * self._integ
        return np.clip(u, -self.out_clip, self.out_clip)


## ---------------------------------------------------------------------------
## Self-test: a simulated OGBench-like point kinematics loop (pos += 0.2 * u).
## ---------------------------------------------------------------------------
def _simulate(ctrl, targets, pos0, act_scale=0.2, noise_std=0.0, n_steps=400,
              seed=0, tol=0.05):
    """Track a waypoint list; returns (final_dist_to_last_target, trace)."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos0, dtype=np.float64).copy()
    ctrl.reset()
    trace = [pos.copy()]
    ti = 0
    for _ in range(n_steps):
        tgt = targets[ti]
        meas = pos + (rng.normal(0.0, noise_std, 2) if noise_std > 0 else 0.0)
        u = ctrl.step(meas, tgt)
        pos = pos + act_scale * u
        trace.append(pos.copy())
        if np.linalg.norm(tgt - pos) < tol and ti < len(targets) - 1:
            ti += 1
    return float(np.linalg.norm(targets[-1] - pos)), np.asarray(trace)


def _settled_jitter(trace, last=100):
    return float(np.linalg.norm(np.diff(trace[-last:], axis=0), axis=1).mean())


def _selftest():
    print('[pid selftest] 1. pure P (kp=5) == upstream law: step response')
    d, tr = _simulate(PID2D(kp=5), targets=[np.array([4.0, -2.0])],
                      pos0=[0.0, 0.0], n_steps=60)
    print(f'   final dist {d:.4f} wu after {len(tr)-1} steps')
    assert d < 1e-2, d

    print('[pid selftest] 2. waypoint chain (square, 4 wu sides)')
    sq = [np.array(p, dtype=np.float64) for p in [(4, 0), (4, 4), (0, 4), (0, 0)]]
    d, tr = _simulate(PID2D(kp=5), targets=sq, pos0=[0.0, 0.0], n_steps=300)
    assert d < 5e-2, d
    print(f'   closed the square, final dist {d:.4f} wu')

    print('[pid selftest] 3. noisy position estimate: pos_lpf (EMA) damps jitter,')
    print('   and (measured finding) a D-term does NOT -- first-order plant.')
    js = {}
    for name, kw in [('raw P', dict(kp=5)),
                     ('P + kd=2', dict(kp=5, kd=2.0)),
                     ('P + pos_lpf=0.7', dict(kp=5, pos_lpf=0.7))]:
        j_seeds = []
        for seed in range(5):
            _, tr = _simulate(PID2D(**kw), targets=[np.array([4.0, 0.0])],
                              pos0=[0.0, 0.0], noise_std=0.15, n_steps=200,
                              seed=seed)
            j_seeds.append(_settled_jitter(tr))
        js[name] = float(np.mean(j_seeds))
        print(f'   settled jitter {name}: {js[name]:.4f} wu/step (5-seed mean)')
    assert js['P + pos_lpf=0.7'] < 0.6 * js['raw P'], js
    assert not (js['P + kd=2'] < 0.9 * js['raw P']), \
        'D-term unexpectedly helped a first-order plant -- re-check the model'

    print('[pid selftest] 4. steady-state offset (drag): Ki closes the last mile')
    def sim_drag(ctrl):
        pos = np.zeros(2)
        ctrl.reset()
        tgt = np.array([4.0, 0.0])
        for _ in range(400):
            u = ctrl.step(pos, tgt)
            pos = pos + 0.2 * u + np.array([-0.05, 0.0])   # constant drag
        return float(np.linalg.norm(tgt - pos))
    dP = sim_drag(PID2D(kp=5, ki=0.0))
    dI = sim_drag(PID2D(kp=5, ki=0.5, ki_zone=2.0, i_clamp=2.0))
    print(f'   residual: P-only {dP:.4f} vs PI {dI:.4f} wu')
    assert dI < dP * 0.2, (dI, dP)

    print('[pid selftest] 5. saturation: far target -> |u| == 1 componentwise')
    u = PID2D(kp=5).step([0, 0], [10, -10])
    assert np.allclose(np.abs(u), 1.0), u

    print('[pid selftest] ALL PASS')


if __name__ == '__main__':
    _selftest()

## end of module
