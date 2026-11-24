"""Procedural tableware (plate, glass, bowl, mug) built from MuJoCo primitives.

Object frame origin = centre of the base bottom, +z up. Parking and placement
are therefore trivial (z = table height).  Hollow shapes are approximated with
a ring of tilted boxes (compound collision), solid ones with a cylinder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _qz(a): return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))
def _qy(a): return (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def _f(v): return " ".join(f"{x:.5f}" for x in v)


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    height: float
    radius: float        # outer footprint radius
    mass: float
    grasp_style: str     # "top" (pinch the rim from above) | "side" (wrap the body)
    rim_radius: float    # axis -> rim distance used by the expert for "top"
    rgba: tuple


def make_object_specs(s: float = 1.0) -> dict:
    return {
        "plate": ObjectSpec("plate", 0.0236 * s, 0.065 * s, 0.25, "top", 0.062 * s, (0.9, 0.9, 0.95, 1)),
        "bowl": ObjectSpec("bowl", 0.0443 * s, 0.0675 * s, 0.30, "top", 0.066 * s, (0.3, 0.6, 0.9, 1)),
        "glass": ObjectSpec("glass", 0.09 * s, 0.03 * s, 0.20, "side", 0.0, (0.6, 0.85, 0.9, 0.6)),
        "mug": ObjectSpec("mug", 0.08 * s, 0.047 * s, 0.25, "side", 0.0, (0.9, 0.4, 0.3, 1)),
    }


def _geom(name, gtype, size, pos, quat, mass, rgba, fr):
    return (f'<geom name="{name}" type="{gtype}" size="{_f(size)}" pos="{_f(pos)}" '
            f'quat="{_f(quat)}" mass="{mass:.5f}" rgba="{_f(rgba)}" friction="{_f(fr)}" condim="4"/>')


def _walls(prefix, i0, n, r_b, z0, length, tilt, thick, mass, rgba, fr):
    out = []
    rc = r_b + 0.5 * length * math.sin(tilt)
    zc = z0 + 0.5 * length * math.cos(tilt)
    w = 1.08 * rc * math.tan(math.pi / n)
    for i in range(n):
        th = 2 * math.pi * i / n
        q = _qmul(_qz(th), _qy(tilt))
        out.append(_geom(f"{prefix}_g{i0 + i}", "box", (thick / 2, w, length / 2),
                         (rc * math.cos(th), rc * math.sin(th), zc), q, mass, rgba, fr))
    return out


def object_geoms_xml(name: str, s: float, fr) -> str:
    spec = make_object_specs(s)[name]
    rgba, M, I = spec.rgba, spec.mass, (1, 0, 0, 0)
    p = f"obj_{name}"
    g = []
    if name in ("plate", "bowl"):
        n = 16
        base_r, base_hh = (0.06, 0.004) if name == "plate" else (0.03, 0.004)
        r_b, z0, L, tilt = (0.058, 0.006, 0.02, 0.5) if name == "plate" else (0.028, 0.006, 0.055, 0.8)
        g.append(_geom(f"{p}_g0", "cylinder", (base_r * s, base_hh * s, 0), (0, 0, base_hh * s), I, 0.4 * M, rgba, fr))
        g += _walls(p, 1, n, r_b * s, z0 * s, L * s, tilt, 0.004 * s, 0.6 * M / n, rgba, fr)
    elif name == "glass":
        g.append(_geom(f"{p}_g0", "cylinder", (0.03 * s, 0.045 * s, 0), (0, 0, 0.045 * s), I, M, rgba, fr))
    elif name == "mug":
        g.append(_geom(f"{p}_g0", "cylinder", (0.035 * s, 0.04 * s, 0), (0, 0, 0.04 * s), I, 0.85 * M, rgba, fr))
        g.append(_geom(f"{p}_g1", "box", (0.004 * s, 0.012 * s, 0.02 * s), (0, 0.047 * s, 0.04 * s), I, 0.15 * M, rgba, fr))
    else:
        raise ValueError(f"unknown object {name}")
    return "\n      ".join(g)
