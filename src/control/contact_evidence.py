"""Conservative interpretation of PhysX contact points; no simulation imports."""
import numpy as np


def classify_contact(separations, impulses):
    """Return proximity, loaded_touch, or guarded (including invalid evidence).

    Only >=2 mm separation at EVERY point with EXACT zero impulse is proximity.
    Support requires a near-touch point (<=0.5 mm) carrying a nonzero impulse.
    Positive-gap speculative forces are never ignored or counted as support.
    Callers must still restrict permitted actor pairs and task phases.
    """
    try:
        gap = np.asarray(separations, dtype=float)
        force = np.asarray(impulses, dtype=float)
    except (ValueError, TypeError):
        return "guarded"
    if (gap.ndim != 1 or gap.size == 0 or force.shape != (gap.size, 3)
            or not np.isfinite(gap).all() or not np.isfinite(force).all()):
        return "guarded"
    loaded = np.any(force != 0., axis=1)
    if np.all(gap >= .002) and not loaded.any():
        return "proximity"
    if np.any((gap <= .0005) & loaded):
        return "loaded_touch"
    return "guarded"