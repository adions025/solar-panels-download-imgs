# ids.py
from __future__ import annotations
import hashlib
from typing import Iterable
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from shapely.geometry.polygon import orient
from shapely.ops import transform

def _round_geom(g, prec: int):
    def _r(x, y, z=None):
        if z is None:
            return (round(x, prec), round(y, prec))
        return (round(x, prec), round(y, prec), round(z, prec))
    return transform(_r, g)

def _normalize_polygon(p: Polygon) -> Polygon:
    # Fix winding: exterior CCW, holes CW
    p = orient(p, sign=1.0)
    # Sort interior rings by area descending for deterministic order
    if p.interiors:
        rings = sorted(p.interiors, key=lambda r: LinearRing(r).envelope.area, reverse=True)
        return Polygon(p.exterior.coords, [list(r.coords) for r in rings])
    return p

def _normalize_mpoly(mp: MultiPolygon) -> MultiPolygon:
    # Normalize each polygon, then sort parts by (area desc, centroid x, centroid y)
    parts = [ _normalize_polygon(p) for p in mp.geoms ]
    parts.sort(key=lambda p: (-p.area, round(p.centroid.x, 6), round(p.centroid.y, 6)))
    return MultiPolygon(parts)

def normalize_geometry(geom, rounding_precision: int = 3):
    """
    Returns a geometry normalized for stable hashing:
      - coordinate rounding,
      - fixed ring orientation,
      - deterministic ordering of holes and multiparts.
    Use EPSG:25831 before calling this (meters).
    """
    g = _round_geom(geom, rounding_precision)
    if isinstance(g, Polygon):
        return _normalize_polygon(g)
    if isinstance(g, MultiPolygon):
        return _normalize_mpoly(g)
    # For other types, try to polygonize or just return rounded geometry
    return g

def geometry_hash_id(geom, rounding_precision: int = 3, digest_len: int = 12) -> str:
    """
    Stable, geometry-only ID. Same city/year not required for uniqueness.
    """
    g = normalize_geometry(geom, rounding_precision)
    # Canonical WKB: big-endian, no SRID
    try:
        from shapely import wkb
        wkb_bytes = wkb.dumps(g, hex=False, byte_order=1, include_srid=False)  # Shapely 2.x
    except Exception:
        from shapely import wkb
        wkb_bytes = wkb.dumps(g, hex=False, big_endian=True, include_srid=False)  # Shapely 1.x

    h = hashlib.blake2b(wkb_bytes, digest_size=digest_len)
    return h.hexdigest()

