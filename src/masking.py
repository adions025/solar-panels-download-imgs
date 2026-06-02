# masking.py
from __future__ import annotations
from rasterio import features
from shapely.geometry import mapping

def white_outside_polygon(rgb_or_rgba, polygon_geom, transform):
    h, w = rgb_or_rgba.shape[1], rgb_or_rgba.shape[2]
    mask = features.rasterize(
        [(mapping(polygon_geom), 1)],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        all_touched=True,   # <- IMPORTANT: thin/corner buildings still hit pixels
        dtype="uint8",
    ).astype(bool)

    if not mask.any():
        print("[MASK WARN] building burned 0 pixels in this cutout")

    out = rgb_or_rgba.copy()
    bands = out.shape[0]
    for b in range(min(3, bands)):
        band = out[b]
        band[~mask] = 255
        out[b] = band
    if bands >= 4:
        out[3][:] = 255
    return out

