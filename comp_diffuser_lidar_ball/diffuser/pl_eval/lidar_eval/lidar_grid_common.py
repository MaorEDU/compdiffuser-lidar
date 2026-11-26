"""
Shared helpers for the LiDAR (x, y, theta) grid-localization CLIs.

Robust, torch-free imports of the grid + localization library (so every CLI runs
on a bare CPU box) plus the maze-overlay plotting used by all the map scripts.
Plotting convention matches ``lidar_pos_histogram.py`` (world-coordinate extent,
``origin='lower'``, dark wall rectangles).
"""

import os
import json
import importlib.util
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------#
# torch-free imports of our own library modules
# ---------------------------------------------------------------------------#
def _load_module(mod_name, rel_path):
    """Import ``mod_name`` from the package; fall back to a by-path load so the
    CLIs work without importing the (torch-pulling) ``diffuser`` package."""
    try:
        return __import__(mod_name, fromlist=['*'])
    except Exception:  # noqa: BLE001
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, rel_path))
        spec = importlib.util.spec_from_file_location(os.path.basename(rel_path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


LG = _load_module('diffuser.datasets.lidar.lidar_grid', '../../datasets/lidar/lidar_grid.py')
LL = _load_module('diffuser.datasets.lidar.lidar_localization', '../../datasets/lidar/lidar_localization.py')
LS = _load_module('diffuser.datasets.lidar.lidar_sim', '../../datasets/lidar/lidar_sim.py')


# ---------------------------------------------------------------------------#
# maze geometry
# ---------------------------------------------------------------------------#
def maze_bfs_wu(scanner, start_cell):
    """BFS distance (world units) from ``start_cell`` over free maze cells.

    Lives here rather than in ``ball_dump_plans`` because the anatomy tool needs
    it too and must stay importable without torch. ``ball_dump_plans`` re-exports
    it, so there is exactly one implementation of maze distance in the codebase.
    """
    mp = np.asarray(scanner.maze_map)
    Hc, Wc = mp.shape
    BIG = np.inf
    dist = np.full((Hc, Wc), BIG, dtype=np.float64)
    si, sj = int(start_cell[0]), int(start_cell[1])
    if not (0 <= si < Hc and 0 <= sj < Wc) or mp[si, sj] == 1:
        return dist
    dist[si, sj] = 0.0
    from collections import deque
    q = deque([(si, sj)])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < Hc and 0 <= b < Wc and mp[a, b] != 1 and dist[a, b] == BIG:
                dist[a, b] = dist[i, j] + scanner.maze_unit
                q.append((a, b))
    return dist


def maze_shortest_path_cells(scanner, start_cell, goal_cell):
    """Cell-by-cell shortest path start->goal, by descending a BFS field."""
    dist = maze_bfs_wu(scanner, start_cell)
    gi, gj = int(goal_cell[0]), int(goal_cell[1])
    if not np.isfinite(dist[gi, gj]):
        return None
    mp = np.asarray(scanner.maze_map)
    path = [(gi, gj)]
    while dist[path[-1]] > 0:
        i, j = path[-1]
        nxt = None
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if (0 <= a < mp.shape[0] and 0 <= b < mp.shape[1] and mp[a, b] != 1
                    and dist[a, b] == dist[i, j] - scanner.maze_unit):
                nxt = (a, b); break
        if nxt is None:
            return None
        path.append(nxt)
    return path[::-1]


def los_clear(scanner, p0, pts, margin=0.6, step=0.5):
    """Which of ``pts`` can be driven at in a STRAIGHT LINE from ``p0``?

    Returns a bool array of shape ``(len(pts),)``. A target is "clear" when the
    segment ``p0 -> pt``, and the same segment offset by +-``margin`` wu
    perpendicular, cross no wall cell. The parallel offsets are what keep the
    ball (which has a radius, unlike the point that the cell test assumes) from
    being sent to scrape a corner diagonally.

    Why this exists (2026-07-30 night). The executor's pursuit target is a point
    on the DECODED plan, and the decoded plan still carries ~4-6 residual cell
    jumps per 888 frames. When the cursor lands just before a jump the target
    sits across a wall; the PID is pure proportional (kp=5, kd=ki=0), so it
    presses into that wall and the ball stops dead. Measured on one benchmark run: 7/20
    episodes sat at a BIT-IDENTICAL position at step 888 and step 1776 -- an
    entire 888-frame plan spent pinned against one wall -- and all 7 failed.
    Aiming only at points the ball can actually reach in a straight line breaks
    that at its origin.

    Pure numpy over ``scanner.maze_map`` -- the same map that already backs the
    goal scan, the waypoints and the decode grid, so no new privileged
    information enters (see the 2026-07-30 'honest framing' note).
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


def make_waypoints(scanner, starts, goals, n_comp):
    """Per-segment AIM poses: (n_comp-1, P, D) scans at equally spaced points
    along the maze shortest path from each start to its goal.

    Segment i is asked to END at waypoint i; the last segment keeps the real
    goal. Every segment therefore becomes anchored (by its backward overlap, or
    by the start inpaint for segment 0) AND aimed -- the n_comp=1 regime, which
    is the only one this model handles well (G2b 90%, 2 MAP jumps, residual
    0.069, vs 42%/24/0.289 at n_comp=8 with stgl_cond={} interiors).

    The scans are synthesized from the KNOWN maze map, exactly as the goal scan
    already is -- no privileged information beyond what the goal condition and
    the decode grid already use. The model is still never told where it is.
    """
    P = len(starts)
    wps_xy = np.zeros((n_comp - 1, P, 2), np.float64)
    n_fallback = 0
    for p in range(P):
        c = scanner.world_to_cell(np.array([starts[p][0], goals[p][0]]),
                                  np.array([starts[p][1], goals[p][1]]))
        ci, cj = np.asarray(c[0], int), np.asarray(c[1], int)
        path = maze_shortest_path_cells(scanner, (ci[0], cj[0]), (ci[1], cj[1]))
        if path is None or len(path) < 2:
            ## unreachable / degenerate: fall back to straight-line interpolation
            ## in world space. Counted and reported -- never silent.
            n_fallback += 1
            for i in range(n_comp - 1):
                f = (i + 1) / n_comp
                wps_xy[i, p] = starts[p] + f * (np.asarray(goals[p]) - np.asarray(starts[p]))
            continue
        u = scanner.maze_unit
        pxy = np.array([[j * u - scanner.offset_x, i * u - scanner.offset_y]
                        for i, j in path], np.float64)
        ## arc-length parameterisation so waypoints are evenly spaced ALONG the
        ## corridor, not evenly spaced in cell index
        seg = np.linalg.norm(np.diff(pxy, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        for i in range(n_comp - 1):
            wps_xy[i, p] = np.stack([
                np.interp(s[-1] * (i + 1) / n_comp, s, pxy[:, 0]),
                np.interp(s[-1] * (i + 1) / n_comp, s, pxy[:, 1])])
    wps = np.stack([np.stack([scanner.scan(wps_xy[i, p, 0], wps_xy[i, p, 1], 0.0)
                              for p in range(P)]) for i in range(n_comp - 1)])
    print_color(
        f'[waypoints] {n_comp - 1} aim poses/problem from the maze shortest path '
        f'(arc-length spaced); every intermediate segment now gets a goal stgl and '
        f'drops its forward overlap (= the trained stovlp_gl mode), segment 0 becomes '
        f'both_stgl. Median consecutive waypoint spacing '
        f'{np.median(np.linalg.norm(np.diff(np.concatenate([starts[None], wps_xy, np.asarray(goals)[None]], 0), axis=0), axis=2)):.1f} wu '
        f'(a 104-frame stride at data progress pace = 7.9 wu).'
        + (f' !!! {n_fallback}/{P} problems had NO maze path and fell back to '
           f'straight-line waypoints -- these may sit inside walls.' if n_fallback else ''),
        c=('r' if n_fallback else 'c'))
    return wps.astype(np.float32), wps_xy


# ---------------------------------------------------------------------------#
# maze overlay plotting
# ---------------------------------------------------------------------------#
def maze_world_extent(scanner):
    H, W = scanner.maze_map.shape
    u, ox, oy = scanner.maze_unit, scanner.offset_x, scanner.offset_y
    return (-ox - 0.5 * u, (W - 1) * u - ox + 0.5 * u,
            -oy - 0.5 * u, (H - 1) * u - oy + 0.5 * u)


def draw_walls(ax, scanner):
    H, W = scanner.maze_map.shape
    u, ox, oy = scanner.maze_unit, scanner.offset_x, scanner.offset_y
    for i in range(H):
        for j in range(W):
            if scanner.maze_map[i, j] == 1:
                ax.add_patch(Rectangle((j * u - ox - 0.5 * u, i * u - oy - 0.5 * u),
                                       u, u, facecolor='0.25', edgecolor='none', zorder=3))


def grid_img_extent(grid):
    """imshow extent for a (n_row, n_col) grid map, centered on the lattice points."""
    h = grid.xy_step / 2.0
    return (grid.x_vals[0] - h, grid.x_vals[-1] + h,
            grid.y_vals[0] - h, grid.y_vals[-1] + h)


def plot_xy_map(grid, values, out_png, title, cmap='viridis', label='value',
                marks=None, log=False):
    """Render a (n_row, n_col) per-(x, y) map over the maze layout.

    marks: optional list of dicts {xy:(x,y), c:color, m:marker, s:size, label:str}.
    """
    scanner = grid.scanner
    H, W = scanner.maze_map.shape
    v = np.array(values, dtype=np.float64)
    if log:
        v = np.log1p(v - np.nanmin(v))
    fig, ax = plt.subplots(figsize=(0.55 * W, 0.55 * H))
    im = ax.imshow(v, origin='lower', extent=grid_img_extent(grid),
                   cmap=cmap, aspect='equal', zorder=1)
    draw_walls(ax, scanner)
    if marks:
        for mk in marks:
            ax.scatter([mk['xy'][0]], [mk['xy'][1]], c=mk.get('c', 'r'),
                       marker=mk.get('m', '*'), s=mk.get('s', 220),
                       edgecolors='white', linewidths=1.3, zorder=5,
                       label=mk.get('label'))
        if any('label' in mk for mk in marks):
            ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    xmin, xmax, ymin, ymax = maze_world_extent(scanner)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, fraction=0.035, label=label)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


def plot_cell_heatmap(values, free, scanner, out_png, title, cmap='viridis',
                      label='value', vcenter=None, annotate=False, marks=None,
                      ax=None, fmt='{:.2f}', clim=None, cbar=True):
    """Render one value per *maze cell* (not per lattice point) over the maze.

    Canonical home of the per-cell renderer (it used to live in
    ``lidar_pos_histogram.py``, which now imports it from here).

    values: per-free-cell values aligned with ``free``, or -- when ``free`` is
        None -- a full (H, W) array already laid out on the maze.
    free:   list of (i, j) free cells matching ``values``, or None.
    vcenter: when given, use a diverging colormap normalized so this value sits
        at the midpoint (used to put a decision threshold like ``tau`` at the
        colour break, so below/above is readable at a glance).
    annotate: print each cell's value inside the cell.
    ax:     draw into an existing axes (for multi-panel figures) instead of
        creating and saving a figure. ``out_png`` is then ignored.
    clim:   (vmin, vmax) colour limits; ignored when ``vcenter`` is set.
    cbar:   draw a colorbar (turn off for multi-panel figures sharing one).
    """
    H, W = scanner.maze_map.shape
    # maze_world_extent spans exactly cell-edge to cell-edge, so a (H, W) image
    # drawn with this extent puts cell (i, j) on its world center (j*u-ox, i*u-oy).
    xmin, xmax, ymin, ymax = maze_world_extent(scanner)
    if free is None:
        cells = np.array(values, dtype=np.float64)
        assert cells.shape == (H, W), f'expected {(H, W)} map, got {cells.shape}'
    else:
        cells = np.full((H, W), np.nan)
        for (i, j), v in zip(free, values):
            cells[i, j] = v

    norm = None
    if vcenter is not None:
        finite = cells[np.isfinite(cells)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            # TwoSlopeNorm needs lo < vcenter < hi; nudge if the data is one-sided
            eps = max(1e-9, 1e-6 * max(1.0, abs(float(vcenter))))
            lo = min(lo, float(vcenter) - eps)
            hi = max(hi, float(vcenter) + eps)
            norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=float(vcenter), vmax=hi)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(0.6 * W, 0.6 * H))
    else:
        fig = ax.figure
    im = ax.imshow(cells, origin='lower', extent=[xmin, xmax, ymin, ymax],
                   cmap=cmap, aspect='equal', zorder=1, interpolation='nearest',
                   norm=norm)
    if clim is not None and norm is None:
        im.set_clim(*clim)
    draw_walls(ax, scanner)
    if annotate:
        u, ox, oy = scanner.maze_unit, scanner.offset_x, scanner.offset_y
        for i in range(H):
            for j in range(W):
                if not np.isfinite(cells[i, j]):
                    continue
                ax.text(j * u - ox, i * u - oy, fmt.format(cells[i, j]),
                        ha='center', va='center', fontsize=5.5, color='white',
                        zorder=4,
                        path_effects=[pe.withStroke(linewidth=1.2, foreground='black')])
    if marks:
        for mk in marks:
            ax.scatter([mk['xy'][0]], [mk['xy'][1]], c=mk.get('c', 'r'),
                       marker=mk.get('m', '*'), s=mk.get('s', 220),
                       edgecolors='white', linewidths=1.3, zorder=5,
                       label=mk.get('label'))
        if any('label' in mk for mk in marks):
            ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
    if cbar:
        fig.colorbar(im, ax=ax, fraction=0.035, label=label)
    if own_fig:
        fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
        return out_png
    return im


def plot_trajectory(grid, out_png, title, true_xy=None, est_xy=None, marks=None):
    """Overlay a true and/or estimated (x, y) trajectory on the maze."""
    scanner = grid.scanner
    H, W = scanner.maze_map.shape
    fig, ax = plt.subplots(figsize=(0.55 * W, 0.55 * H))
    draw_walls(ax, scanner)
    if true_xy is not None:
        ax.plot(true_xy[:, 0], true_xy[:, 1], '-o', ms=3, lw=1.5, color='tab:green',
                label='true', zorder=4)
    if est_xy is not None:
        ax.plot(est_xy[:, 0], est_xy[:, 1], '-x', ms=4, lw=1.2, color='tab:red',
                label='estimated', zorder=4)
    if marks:
        for mk in marks:
            ax.scatter([mk['xy'][0]], [mk['xy'][1]], c=mk.get('c', 'k'),
                       marker=mk.get('m', '*'), s=mk.get('s', 200),
                       edgecolors='white', linewidths=1.2, zorder=6, label=mk.get('label'))
    xmin, xmax, ymin, ymax = maze_world_extent(scanner)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=11); ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------#
# data loading (raw OGBench npz -> obs + episode ids)
# ---------------------------------------------------------------------------#
def load_obs_npz(npz_path):
    d = np.load(os.path.expanduser(npz_path), allow_pickle=True)
    obs = np.asarray(d['observations'], dtype=np.float32)
    term = None
    for k in ('terminals', 'dones', 'timeouts'):
        if k in d.files:
            term = np.asarray(d[k]).reshape(-1).astype(bool)
            break
    if term is not None and term.shape[0] == obs.shape[0]:
        ep_id = np.cumsum(np.concatenate([[0], term[:-1].astype(np.int64)]))
    else:
        ep_id = np.zeros(obs.shape[0], dtype=np.int64)
    return obs, ep_id


def lidar_config_from_args(args):
    return dict(n_beams=args.n_beams, max_range=args.max_range, fov=2.0 * np.pi)


# ---------------------------------------------------------------------------#
# sensing geometry: ONE resolver for every CLI on the measurement path
# ---------------------------------------------------------------------------#
DEFAULT_LIDAR_CONFIG = dict(n_beams=30, max_range=12.0, fov=2 * np.pi, step=0.1)


def print_color(s, *args, c='r'):
    """Torch-free stand-in for ``diffuser.utils.print_color`` (same ANSI codes),
    so this module stays importable on a bare CPU box."""
    code = dict(r='\033[31m', b='\033[34m', y='\033[33m', g='\033[32m',
                c='\033[36m').get(c, '\033[36m')
    print(code, end='')
    print(s, *args, '\033[39m')


def _lidar_cfg_from_mapping(cfg):
    """Pull ``lidar_config`` out of a saved config, NESTED key first.

    ``train_ogb_stgl_sml.py`` saves ``dataset_config.pkl`` with the dataset
    kwargs nested one level down, i.e. ``cfg['dataset_config']['lidar_config']``.
    A flat ``cfg['lidar_config']`` lookup therefore raises ``KeyError`` and the
    caller silently falls back to the module default -- which is exactly what
    both 2026-07-29 bridge runs printed (`MODULE DEFAULT (unverified!)`). The
    mr12 arm was correct only because the default happened to coincide with the
    trained value and the mr24 arm only because of its CLI flag. Try nested,
    then flat, and report which one hit.
    """
    if cfg is None:
        return None, ''
    try:
        ds = cfg.get('dataset_config') if hasattr(cfg, 'get') else None
    except Exception:  # noqa: BLE001
        ds = None
    if ds is not None:
        try:
            lc = ds['lidar_config']
            if lc:
                return dict(lc), 'nested dataset_config.lidar_config'
        except Exception:  # noqa: BLE001
            pass
    try:
        lc = cfg['lidar_config']
        if lc:
            return dict(lc), 'flat lidar_config'
    except Exception:  # noqa: BLE001
        pass
    return None, ''


def resolve_lidar_config(logdir='', meta=None, n_beams=0, max_range=0.0, quiet=False):
    """Sensing geometry for the eval path -- MUST come from the checkpoint.

    **2026-07-29 incident (read before "simplifying" this back to a constant).**
    The dump module used to hardcode ``max_range=12.0`` and feed that one dict to
    BOTH ``make_scanner_for_env`` (which synthesizes the start/goal conditioning
    scans) AND ``load_or_build_grid`` (the G2 decode reference). The mr24 arm was
    dumped through it: an mr24-trained model got mr12 conditioning (OOD), an mr12
    normalizer, and its output was decoded against an mr12 grid. Every G2 number
    from that run was meaningless -- and the failure was silent, because frame 0
    is inpainted and round-trips through any normalizer, so ``start err`` still
    printed its usual healthy 0.37 wu.

    Resolution order: explicit CLI override > dump ``meta`` > run dir's saved
    ``dataset_config.pkl`` (nested key, then flat) > module default (loud warning).
    """
    cfg, src = None, ''
    if meta:
        try:
            m = json.loads(meta) if isinstance(meta, str) else meta
            lc = m.get('lidar_config') if hasattr(m, 'get') else None
            if lc:
                cfg, src = dict(lc), 'dump meta'
            elif not logdir and m.get('logdir'):
                ## dumps that predate lidar_config-in-meta still record the run
                ## dir; with the nested lookup below that is enough to recover
                ## the geometry instead of falling through to the default.
                logdir = str(m['logdir'])
        except Exception:  # noqa: BLE001
            cfg = None
    if cfg is None and logdir:
        try:
            from diffuser.utils import load_config          # lazy: pulls torch
            saved = load_config(logdir, 'dataset_config.pkl')
            cfg, where = _lidar_cfg_from_mapping(saved)
            if cfg is None:
                raise KeyError(
                    'lidar_config absent under both dataset_config.lidar_config '
                    f'and lidar_config; keys={sorted(saved.keys())}')
            src = f'ckpt dataset_config.pkl ({where})'
        except Exception as e:  # noqa: BLE001
            print_color(f'[audit] could not read lidar_config from {logdir}: {e}', c='y')
    if cfg is None:
        cfg, src = dict(DEFAULT_LIDAR_CONFIG), 'MODULE DEFAULT (unverified!)'
    cfg = {**DEFAULT_LIDAR_CONFIG, **cfg}
    if n_beams:
        cfg['n_beams'], src = int(n_beams), src + ' + CLI n_beams'
    if max_range:
        cfg['max_range'], src = float(max_range), src + ' + CLI max_range'
    ## cache keys / scanner allocation want exact types
    cfg['n_beams'], cfg['max_range'] = int(cfg['n_beams']), float(cfg['max_range'])
    if not quiet:
        print_color(
            f"[audit] lidar_config from {src}: n_beams={cfg['n_beams']} "
            f"max_range={cfg['max_range']} -> grid cache will say "
            f"mr{int(cfg['max_range'])}. CHECK THIS MATCHES THE TRAINED MODEL.",
            c=('g' if 'DEFAULT' not in src else 'r'))
    return cfg


def assert_npz_matches(lidar, lidar_cfg, npz_path):
    """The ``--npz`` normalizer source must have been generated at the model's
    max_range. An mr12 npz maxes out at exactly 12.0, an mr24 npz at 24.0, so a
    single max() catches the leak that the 2026-07-29 mr24 dump shipped with."""
    obs_max, mr = float(np.asarray(lidar).max()), float(lidar_cfg['max_range'])
    assert abs(obs_max - mr) < 0.5, (
        f'--npz {npz_path} has max scan {obs_max:.2f} wu but the model was trained at '
        f'max_range={mr:.1f}. Wrong normalizer source -> the dump is invalid. '
        f'Regenerate: python -m diffuser.datasets.lidar.gen_lidar_dataset --env <env> '
        f'--max_range {mr:.0f} --out <..._r{int(mr)}.npz>')
    assert int(np.asarray(lidar).shape[-1]) == int(lidar_cfg['n_beams']), (
        f'--npz has {np.asarray(lidar).shape[-1]} beams, model wants {lidar_cfg["n_beams"]}')
    print_color(f'[audit] npz OK: max scan {obs_max:.2f} wu == max_range {mr:.1f}', c='g')


def json_safe_lidar_config(cfg):
    """numpy scalars / non-primitives -> json-serializable (fov is np.pi-derived)."""
    out = {}
    for k, v in dict(cfg).items():
        if isinstance(v, (bool, str)) or v is None:
            out[k] = v
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


def add_grid_args(ap):
    """Common grid/localization CLI arguments."""
    ap.add_argument('--env', default='antmaze-giant-stitch-v0')
    ap.add_argument('--n_beams', type=int, default=30)
    ap.add_argument('--max_range', type=float, default=12.0)
    ap.add_argument('--xy_step', type=float, default=1.0,
                    help='grid spatial resolution, world units (1 cell = maze_unit)')
    ap.add_argument('--n_theta', type=int, default=12,
                    help='orientation bins over 360 deg (12 == 30 deg/bin)')
    ap.add_argument('--sigma', type=float, default=1.0,
                    help='measurement-likelihood noise scale (world units)')
    ap.add_argument('--cache_root', default='data/ogb_maze/lidar_grid')
    return ap
