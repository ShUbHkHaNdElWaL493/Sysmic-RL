"""Support-surface detection: find the flat top of the robot's own 'table-like extrusion' by ray casting.

Rays are cast straight down on a grid over the search box, only against collision geoms (group 0) of
robot bodies (arms, objects and the destination table are hidden). The largest connected flat patch
becomes the object source. Cells are stored in world coordinates at scan time plus the body they hit,
so if the surface is part of a moving body (e.g. the lift carriage) it is translated with that body.
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np


class SupportSurface:
    def __init__(self, body_id, body_name, x0, y0, step, grid, z, ref_pos):
        self.body_id, self.body_name = body_id, body_name
        self.x0, self.y0, self.step, self.grid, self.z = x0, y0, step, grid, z
        self.ref_pos = np.array(ref_pos, dtype=np.float64)   # body position at scan time
        self._valid = {}

    @property
    def area(self): return float(self.grid.sum() * self.step ** 2)

    def offset(self, data):
        return data.xpos[self.body_id] - self.ref_pos         # translation since the scan

    def bbox(self, data):
        ij = np.argwhere(self.grid)
        lo = np.array([self.x0, self.y0]) + ij.min(0) * self.step
        hi = np.array([self.x0, self.y0]) + (ij.max(0) + 1) * self.step
        o = self.offset(data)
        return lo + o[:2], hi + o[:2], self.z + o[2]

    def valid_cells(self, radius):
        """Cell centres (scan frame) where a disk of `radius` fits entirely on the surface."""
        key = round(radius, 4)
        if key not in self._valid:
            k = int(np.ceil(radius / self.step))
            g = np.pad(self.grid, k, constant_values=False)
            ok = np.ones_like(self.grid, dtype=bool)
            nx, ny = self.grid.shape
            for di in range(-k, k + 1):
                for dj in range(-k, k + 1):
                    if (di * di + dj * dj) * self.step ** 2 <= radius ** 2:
                        ok &= g[k + di:k + di + nx, k + dj:k + dj + ny]
            ij = np.argwhere(ok)
            self._valid[key] = np.array([self.x0, self.y0]) + (ij + 0.5) * self.step
        return self._valid[key]

    def sample(self, rng, radius, data):
        cells = self.valid_cells(radius)
        if len(cells) == 0:
            raise RuntimeError(f"No spot on the support surface fits an object of radius {radius:.3f} m. "
                               "Check SourceConfig / run the probe.")
        xy = cells[rng.integers(len(cells))] + rng.uniform(-0.5, 0.5, 2) * self.step
        o = self.offset(data)
        return np.array([xy[0] + o[0], xy[1] + o[1], self.z + o[2]])


def _components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for i, j in zip(*np.nonzero(mask)):
        if seen[i, j]:
            continue
        comp, q = [], deque([(i, j)])
        seen[i, j] = True
        while q:
            a, b = q.popleft()
            comp.append((a, b))
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = a + da, b + db
                if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    q.append((x, y))
        out.append(comp)
    return out


def scan_heights(model, data, hide, xs, ys, z_from=3.0):
    """Ray-cast down over the grid. Returns (z[nx,ny], body[nx,ny]); z=nan where nothing was hit."""
    saved = model.geom_group.copy()
    model.geom_group[hide] = 5
    mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)       # collision geoms (group 0) only
    z = np.full((len(xs), len(ys)), np.nan)
    body = np.full((len(xs), len(ys)), -1, dtype=int)
    gid = np.zeros(1, dtype=np.int32)
    vec = np.array([0.0, 0.0, -1.0])
    try:
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                dist = mujoco.mj_ray(model, data, np.array([x, y, z_from]), vec, mask, 1, -1, gid)
                if dist >= 0 and gid[0] >= 0:
                    z[i, j] = z_from - dist
                    body[i, j] = model.geom_bodyid[gid[0]]
    finally:
        model.geom_group[:] = saved
    return z, body


def detect_support_surface(model, data, hide, cfg) -> SupportSurface:
    if cfg.manual is not None:
        cx, cy, z, hx, hy = cfg.manual
        bid = model.body(cfg.body).id if cfg.body else 0
        x0, y0 = cx - hx, cy - hy
        nx, ny = int(round(2 * hx / cfg.step)), int(round(2 * hy / cfg.step))
        grid = np.ones((nx, ny), dtype=bool)
        return SupportSurface(bid, model.body(bid).name, x0, y0, cfg.step, _shrink(grid, cfg), z, data.xpos[bid])

    xs = np.arange(cfg.search_x[0], cfg.search_x[1], cfg.step)
    ys = np.arange(cfg.search_y[0], cfg.search_y[1], cfg.step)
    z, body = scan_heights(model, data, hide, xs, ys)
    zmin, zmax = cfg.search_z
    ok = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if cfg.body:
        ok &= body == model.body(cfg.body).id
    best = None
    bins = np.round(np.where(ok, z, np.nan) / 0.005)
    for b in np.unique(bins[ok]):
        for comp in _components(bins == b):
            if best is None or len(comp) > len(best[1]):
                best = (b, comp)
    if best is None or len(best[1]) * cfg.step ** 2 < cfg.min_area:
        raise RuntimeError(
            "Could not find a flat support surface on the robot in the search box "
            f"x={cfg.search_x} y={cfg.search_y} z={cfg.search_z}. Run `python -m sysmic_rl.pickplace.probe` "
            "to inspect, then widen the box or set SourceConfig.manual=(cx, cy, z, hx, hy).")
    comp = best[1]
    grid = np.zeros((len(xs), len(ys)), dtype=bool)
    for i, j in comp:
        grid[i, j] = True
    zs = np.array([z[i, j] for i, j in comp])
    bodies = np.array([body[i, j] for i, j in comp])
    bid = int(np.bincount(bodies).argmax())
    ij = np.argwhere(grid)
    sub = grid[ij[:, 0].min():ij[:, 0].max() + 1, ij[:, 1].min():ij[:, 1].max() + 1]
    x0, y0 = xs[ij[:, 0].min()] - cfg.step / 2, ys[ij[:, 1].min()] - cfg.step / 2
    return SupportSurface(bid, model.body(bid).name, x0, y0, cfg.step, _shrink(sub, cfg),
                          float(np.median(zs)), data.xpos[bid])


def _shrink(grid, cfg):
    k = int(round(cfg.edge_margin / cfg.step))
    if k <= 0:
        return grid
    g = np.pad(grid, k, constant_values=False)
    out = np.ones_like(grid)
    nx, ny = grid.shape
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            out &= g[k + di:k + di + nx, k + dj:k + dj + ny]
    return out
