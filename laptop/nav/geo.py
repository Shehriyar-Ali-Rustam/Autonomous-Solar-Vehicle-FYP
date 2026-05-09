#!/usr/bin/env python3
"""Geographic helpers: distance + bearing on a sphere (haversine).

Inputs are decimal degrees. Outputs:
  haversine_m  -> distance in metres
  bearing_deg  -> compass bearing 0=N, 90=E, 180=S, 270=W
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS points, in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from (lat1, lon1) toward (lat2, lon2), 0..360."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def heading_error_deg(current: float, target: float) -> float:
    """Signed shortest angle from current heading to target heading.
    Positive = need to turn right, negative = need to turn left.
    Range: (-180, +180].
    """
    diff = (target - current + 540.0) % 360.0 - 180.0
    return diff
