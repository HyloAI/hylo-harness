"""Grasp quality from contacts: point contacts with Coulomb friction, their friction cones as a few edge forces,
the wrenches those forces exert on the object, force closure and the largest-ball (epsilon) measure of the wrench
hull, and the effort needed to hold the object against gravity.

Conventions: a contact is a point p on the object surface and the unit normal n pointing INTO the object (the
direction the finger pushes). Forces are unit normal force plus friction up to mu (friction cone edges
f_j = n + mu (cos t_j u + sin t_j v)). Torques are taken about the object's reference point c and divided by a
length rho (usually the object's radius), so forces and torques are commensurate:
    w = [f, (p - c) x f / rho].
The epsilon measure is the distance from the origin to the nearest facet of the convex hull of every contact's
edge wrenches: the largest wrench of any direction the grasp resists with unit total normal force (summed over
contacts, an L1 bound). It is positive exactly when the grasp is force closure; zero or negative otherwise.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError


def tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def cone_edges(normal: np.ndarray, mu: float, k: int = 8) -> np.ndarray:
    """(k, 3) edge forces of the linearised friction cone around `normal` (unit normal component)."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    if mu <= 0:
        return n[None].copy()
    u, v = tangent_basis(n)
    t = 2 * np.pi * np.arange(k) / k
    return n[None] + mu * (np.cos(t)[:, None] * u[None] + np.sin(t)[:, None] * v[None])


def contact_wrenches(points: np.ndarray, normals: np.ndarray, centre: np.ndarray, mu: float, k: int = 8,
                     rho: float = 1.0) -> np.ndarray:
    """(m * k, 6) primitive wrenches of m contacts (inward normals)."""
    P, N = np.asarray(points, float).reshape(-1, 3), np.asarray(normals, float).reshape(-1, 3)
    c = np.asarray(centre, float)
    out = []
    for p, n in zip(P, N):
        F = cone_edges(n, mu, k)
        out.append(np.hstack([F, np.cross(p - c, F) / rho]))
    return np.vstack(out) if out else np.zeros((0, 6))


def epsilon(W: np.ndarray, tol: float = 1e-9) -> float:
    """Signed distance from the origin to the boundary of the convex hull of the wrenches W (n, 6): positive inside
    (force closure; the value is the largest-ball radius), negative outside, 0 when the hull is flat (rank < 6)."""
    W = np.asarray(W, float)
    if len(W) < 7 or np.linalg.matrix_rank(W - W.mean(0), tol=1e-7) < 6:
        return 0.0
    try:
        hull = ConvexHull(W)
    except QhullError:
        try:
            hull = ConvexHull(W, qhull_options="QJ")
        except QhullError:
            return 0.0
    # facet equations a.x + b <= 0 inside, |a| = 1: the origin's signed distance to facet i is -b_i
    d = -hull.equations[:, -1]
    e = float(d.min())
    return 0.0 if abs(e) < tol else e


def force_closure(W: np.ndarray, tol: float = 1e-6) -> bool:
    return epsilon(W) > tol


def grasp_epsilon(points: np.ndarray, normals: np.ndarray, centre: np.ndarray, mu: float = 0.5, k: int = 8,
                  rho: float = 1.0) -> float:
    return epsilon(contact_wrenches(points, normals, centre, mu, k, rho))


def holding_effort(points: np.ndarray, normals: np.ndarray, centre: np.ndarray, external: np.ndarray,
                   mu: float = 0.5, k: int = 8, rho: float = 1.0) -> float:
    """Smallest total normal force (in units of |external force|) whose cone forces balance the external wrench
    (e.g. gravity [0, 0, -m g, 0, 0, 0] at the centre): sum of edge weights with sum lambda_j w_j = -external.
    inf when no combination balances it."""
    W = contact_wrenches(points, normals, centre, mu, k, rho)
    if not len(W):
        return float("inf")
    ext = np.asarray(external, float)
    scale = float(np.linalg.norm(ext[:3])) or 1.0
    res = linprog(np.ones(len(W)), A_eq=W.T, b_eq=-ext / scale, bounds=(0, None), method="highs")
    return float(res.fun) if res.status == 0 else float("inf")


__all__ = ["cone_edges", "contact_wrenches", "epsilon", "force_closure", "grasp_epsilon", "holding_effort",
           "tangent_basis"]
