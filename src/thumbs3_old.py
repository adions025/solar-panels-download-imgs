#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 09:02:52 2025

@author: jborgeh
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
from typing import List, Tuple

import geopandas as gpd
import pandas as pd

from shapely.geometry import box as sbox
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from rasterio import features
from rasterio.io import MemoryFile
from rasterio.errors import NotGeoreferencedWarning

from icgc_old import (
    get_municipality_gdf,
    compute_bbox_with_margin,
    pick_image_size,
    territorial_getmap_rgb,
)
from catastro_atom import download_municipality_buildings
from masking import white_outside_polygon
from ids import geometry_hash_id

# ---- Defaults ----
DEFAULT_MARGIN_M = 0.5
UNIFORM_MPP = 0.25  # uniform target resolution (25 cm/px) for all tiles
MAX_W = 2048  # cap W/H so requests are manageable
MAX_H = 2048
DEFAULT_OUTROOT = "."
DEFAULT_DEBUG_FIRST = 0

warnings.simplefilter("ignore", NotGeoreferencedWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Thumbnails from ICGC Territorial orthophoto (full coverage) at a uniform resolution."
    )
    p.add_argument("--city", required=True, help="Municipality name, e.g. 'Nulles'.")
    p.add_argument("--year-ini", type=int, required=True, help="Initial year (inclusive).")
    p.add_argument("--year-end", type=int, required=True, help="Final year (inclusive).")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN_M, help="Margin around building (meters).")
    p.add_argument("--mpp", type=float, default=UNIFORM_MPP, help="Uniform meters-per-pixel for requests.")
    p.add_argument("--outroot", default=DEFAULT_OUTROOT, help="Output root (CITY/YEAR/*.png).")
    p.add_argument("--cadastre", help="Path to local cadastral shapefile (buildings).")
    p.add_argument("--cadastre-source", choices=["file", "atom"], default="file",
                   help="Where to get buildings: local file or Cadastre ATOM download.")
    p.add_argument("--province", default=None,
                   help="Province name for ATOM (e.g., 'Tarragona'). Optional but speeds lookup.")
    p.add_argument("--atom-cache", default="",
                   help="Optional GPKG path to cache the ATOM download (read next runs).")
    p.add_argument("--debug-first", type=int, default=DEFAULT_DEBUG_FIRST,
                   help="Save RAW WMS for first N buildings per year (under OUTROOT/_debug/..).")
    p.add_argument("--limit", type=int, default=0, help="Process at most this many buildings (0=all).")
    p.add_argument("--alta-field", default="FECHAALTA",
                   help="Field name for building start date (YYYYMMDD).")
    p.add_argument("--baja-field", default="FECHABAJA",
                   help="Field name for building end date (YYYYMMDD or 99999999 for active).")
    p.add_argument("--aoi-mode", choices=["muni", "from-cadastre", "bbox"], default="muni",
                   help="Area-of-interest source: municipality (default), union of cadastre geometries, or explicit bbox.")
    p.add_argument("--aoi-bbox", default="",
                   help="AOI bbox as 'minx,miny,maxx,maxy' (use with --aoi-mode=bbox).")
    p.add_argument("--aoi-crs", default="EPSG:25831",
                   help="CRS of --aoi-bbox (default EPSG:25831).")
    return p.parse_args()


def ensure_epsg25831(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise RuntimeError("Input layer has no CRS. Please define/reproject to EPSG:25831.")
    if str(gdf.crs).upper() != "EPSG:25831":
        gdf = gdf.to_crs("EPSG:25831")
    return gdf


# --- bounding box parsing -----------------------------------------------
def _parse_bbox(s: str):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--aoi-bbox must be 'minx,miny,maxx,maxy'")
    return tuple(parts)  # (minx, miny, maxx, maxy)


# --- date parsing & cleaning -----------------------------------------------

def _year_from_yyyymmdd_series(s: pd.Series) -> pd.Series:
    """20020327 -> 2002 robust to strings/ints/NaN. Returns Int64 (nullable)."""
    v = pd.to_numeric(s, errors="coerce")
    return (v // 10000).astype("Int64")


def clean_cadastre_records(
        gdf: gpd.GeoDataFrame,
        alta_field: str = "FECHAALTA",
        baja_field: str = "FECHABAJA",
) -> gpd.GeoDataFrame:
    """
    - Parse years once -> __alta_y, __baja_y (Int64).
    - FECHABAJA==99999999 => open-ended -> __baja_y = 9999.
    - Drop rows with:
        * same start/end year (short-lived) when NOT open-ended,
        * end year < start year (corrupt).
      Missing dates are kept (treated as always-existing later).
    """
    if alta_field not in gdf.columns or baja_field not in gdf.columns:
        return gdf.copy()  # nothing to clean

    alta_y = _year_from_yyyymmdd_series(gdf[alta_field])
    baja_raw = pd.to_numeric(gdf[baja_field], errors="coerce")
    baja_y = (baja_raw // 10000).astype("Int64")
    # open-ended
    is_open = baja_raw == 99999999
    baja_y = baja_y.where(~is_open, 9999)

    # flags
    same_year = (~is_open) & (alta_y.notna()) & (baja_y.notna()) & (alta_y == baja_y)
    end_before_start = (alta_y.notna()) & (baja_y.notna()) & (baja_y < alta_y)

    keep_mask = ~(same_year | end_before_start)
    cleaned = gdf.loc[keep_mask].copy()
    cleaned["__alta_y"] = alta_y[keep_mask]
    cleaned["__baja_y"] = baja_y[keep_mask]
    return cleaned


def filter_by_cadastre_year(
        gdf: gpd.GeoDataFrame,
        year: int,
        alta_field: str = "FECHAALTA",
        baja_field: str = "FECHABAJA",
) -> gpd.GeoDataFrame:
    """
    Keep rows whose life cycle contains 'year': start < year < end.
    FECHABAJA==99999999 => open-ended.
    If date fields are absent, return gdf unchanged (with a one-line info).
    """
    # Case 1: cleanup already created parsed columns
    if "__alta_y" in gdf.columns and "__baja_y" in gdf.columns:
        alta_y = gdf["__alta_y"]
        baja_y = gdf["__baja_y"]

    # Case 2: raw fields exist -> parse them (aligned to gdf.index)
    elif (alta_field in gdf.columns) and (baja_field in gdf.columns):
        alta_y = _year_from_yyyymmdd_series(gdf[alta_field])
        baja_raw = pd.to_numeric(gdf[baja_field], errors="coerce")
        baja_y = (baja_raw // 10000).astype("Int64")
        # open-ended
        baja_y = baja_y.where(baja_raw != 99999999, 9999)

    # Case 3: fields absent -> skip temporal filter
    else:
        print(f"[YEAR {year}] INFO: '{alta_field}'/'{baja_field}' not found -> no temporal filter applied.")
        return gdf.copy()

    # Ensure alignment to gdf index and fill sentinels
    alta_y = alta_y.reindex(gdf.index).fillna(-9999)
    baja_y = baja_y.reindex(gdf.index).fillna(9999)

    mask = (alta_y < year) & (year < baja_y)  # strict, per your spec
    return gdf.loc[mask].copy()


def robust_city_clip(buildings: gpd.GeoDataFrame, muni_geom: BaseGeometry) -> gpd.GeoDataFrame:
    try:
        cand_idx = list(buildings.sindex.intersection(muni_geom.bounds))
        candidates = buildings.iloc[cand_idx]
    except Exception:
        minx, miny, maxx, maxy = muni_geom.bounds
        candidates = buildings.cx[minx:maxx, miny:maxy]
    return candidates[candidates.geometry.intersects(muni_geom)].copy()


def precompute_jobs(
        bld_city: gpd.GeoDataFrame, mpp: float, margin_m: float, max_side: int
) -> List[Tuple[str, BaseGeometry, Tuple[float, float, float, float], int, int]]:
    jobs = []
    for _, row in bld_city.iterrows():
        geom = row.geometry
        if geom.is_empty:
            continue
        bbox = compute_bbox_with_margin(geom, margin_m)
        width, height = pick_image_size(bbox, mpp, max_px=max_side)
        bid = geometry_hash_id(geom, rounding_precision=3)
        jobs.append((bid, geom, bbox, width, height))
    return jobs


def save_png_array(path: str, arr) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = {"driver": "PNG", "width": arr.shape[2], "height": arr.shape[1], "count": arr.shape[0], "dtype": "uint8"}
    with MemoryFile() as mf:
        with mf.open(**profile) as tmp:
            tmp.write(arr)
        with open(path, "wb") as f:
            f.write(mf.read())


def write_building_index(bld_city: gpd.GeoDataFrame, outdir: str):
    out_csv = os.path.join(outdir, "building_index.csv")
    out_gpkg = os.path.join(outdir, "building_index.gpkg")
    rows = []
    for _, r in bld_city.iterrows():
        gid = geometry_hash_id(r.geometry, 3)
        c = r.geometry.centroid
        rows.append({"id": gid, "area_m2": float(r.geometry.area), "centroid_e": float(c.x), "centroid_n": float(c.y)})
    df = pd.DataFrame(rows).drop_duplicates("id")
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(out_csv, index=False)
    # GPKG with geometries + id
    g = bld_city.copy()
    g["id"] = [geometry_hash_id(gx, 3) for gx in g.geometry]
    g[["id", "geometry"]].to_file(out_gpkg, layer="buildings", driver="GPKG")
    print(f"[INDEX] wrote {out_csv} and {out_gpkg}")


def qc_metrics(arr, geom, transform):
    """Return mask_coverage (0..1), raw stats (min,max,std) over the whole tile."""
    h, w = arr.shape[1], arr.shape[2]
    mask = features.rasterize([(geom, 1)], out_shape=(h, w), transform=transform,
                              fill=0, all_touched=True, dtype="uint8").astype(bool)
    coverage = float(mask.mean()) if mask.size else 0.0
    return coverage, float(arr.min()), float(arr.max()), float(arr.std())


def main():
    a = parse_args()
    CITY = a.city
    YEAR_INI = a.year_ini
    YEAR_END = a.year_end
    MARGIN_M = a.margin
    MPP = a.mpp
    OUTROOT = a.outroot
    DEBUG_FIRST = max(0, a.debug_first)
    LIMIT = a.limit

    # 1) Buildings
    if a.cadastre_source == "file":
        CAD = a.cadastre
        if not os.path.isfile(CAD):
            raise SystemExit(f"[INPUT] --cadastre file not found: {CAD}")
        bld = ensure_epsg25831(gpd.read_file(CAD))
        print(f"[CADASTRE:file] Loaded {len(bld)} features from {CAD}")
    # ATOM path
    else:
        cache_path = os.path.expanduser(a.atom_cache) if a.atom_cache else ""
        if cache_path and os.path.isfile(cache_path):
            bld = ensure_epsg25831(gpd.read_file(cache_path))
            print(f"[CADASTRE:atom] Loaded {len(bld)} features from cache {cache_path}")
        else:
            from catastro_atom import list_municipalities  # import here to avoid cycles
            try:
                print(f"[CADASTRE:atom] Downloading INSPIRE Buildings for '{CITY}'"
                      + (f", province '{a.province}'" if a.province else "") + " …")
                bld = download_municipality_buildings(CITY, province=a.province).to_crs("EPSG:25831")
                print(f"[CADASTRE:atom] Downloaded {len(bld)} features.")
            except Exception as e:
                print(f"[CADASTRE:atom] ERROR: {e}")
                if a.province:
                    # dump the full list for the province to help the user pick the exact spelling
                    try:
                        names = list_municipalities(province=a.province)
                        print(f"[CADASTRE:atom] ATOM municipalities for province '{a.province}':")
                        for name in names:
                            print("   -", name)
                    except Exception as e2:
                        print(f"[CADASTRE:atom] Also failed listing province '{a.province}': {e2}")
                raise  # keep behavior: stop the run on error

    # AOI (Area Of Interest) geometry in EPSG:25831
    if a.aoi_mode == "muni":
        # Municipality polygon (ArcGIS/ICGC), reproject to 25831
        muni = ensure_epsg25831(get_municipality_gdf(CITY))
        aoi_geom = muni.geometry.iloc[0]
        aoi_label = CITY
    elif a.aoi_mode == "from-cadastre":
        # Use the union of the cadastre geometries (perfect for partial-city tests)
        uu = unary_union(bld.geometry)
        # In rare cases unary_union can return a GeometryCollection take its envelope
        aoi_geom = (uu if uu.is_valid and not uu.is_empty else bld.unary_union).buffer(0).envelope
        aoi_label = "cadastre-union"
    elif a.aoi_mode == "bbox":
        if not a.aoi_bbox:
            raise SystemExit("--aoi-bbox is required when --aoi-mode=bbox")
        bbox = _parse_bbox(a.aoi_bbox)
        aoi = gpd.GeoDataFrame(geometry=[sbox(*bbox)], crs=a.aoi_crs)
        aoi = ensure_epsg25831(aoi)
        aoi_geom = aoi.geometry.iloc[0]
        aoi_label = "custom-bbox"
    else:
        raise SystemExit("Unknown --aoi-mode")

    # Pretty-print AOI info
    area_km2 = round(aoi_geom.area / 1e6, 3)
    minx, miny, maxx, maxy = aoi_geom.bounds
    print(
        f"[AOI] {a.aoi_mode}='{aoi_label}' | area ≈ {area_km2} km² | bounds 25831 = ({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")

    # Clip buildings to AOI (fast sindex + intersects). If your file is already scoped, this keeps them as-is.
    bld_city = robust_city_clip(bld, aoi_geom)
    print(f"[CLIP] Buildings in file: {len(bld)} within AOI: {len(bld_city)}")
    if len(bld_city) == 0:
        raise SystemExit("[CLIP ERROR] No buildings intersect the AOI.")

    # NEW: clean odd historical records (short-lived, end<start). Keeps parsed years.
    before = len(bld_city)
    bld_city = clean_cadastre_records(bld_city, alta_field=a.alta_field, baja_field=a.baja_field)
    dropped = before - len(bld_city)
    print(f"[CLEAN] Dropped {dropped} anomalous records (short-lived or end<start). Remaining: {len(bld_city)}")

    # Optional limit for quick tests
    if LIMIT > 0 and len(bld_city) > LIMIT:
        bld_city = bld_city.head(LIMIT).copy()
        print(f"[LIMIT] Limiting to first {LIMIT} buildings for this run.")

    # Precompute per-building jobs
    precomp = precompute_jobs(bld_city, MPP, MARGIN_M, max_side=max(MAX_W, MAX_H))
    print(f"City: {CITY} | Buildings: {len(precomp)} | Years: {YEAR_INI}–{YEAR_END} | Uniform m/px: {MPP}")

    city_root = os.path.join(OUTROOT, CITY.replace(" ", "_"))
    debug_root = os.path.join(OUTROOT, "_debug", CITY.replace(" ", "_"))
    write_building_index(bld_city, city_root)

    for year in range(YEAR_INI, YEAR_END + 1):
        # 1) Year-specific cadastre filter
        bld_year = filter_by_cadastre_year(
            bld_city, year, alta_field=a.alta_field, baja_field=a.baja_field
        )
        if bld_year.empty:
            print(f"[{year}] No active buildings for this year (after FECHAALTA/FECHABAJA filter). Skipping.")
            continue

        # 2) Build jobs for THIS year only
        precomp = precompute_jobs(bld_year, MPP, MARGIN_M, max_side=max(MAX_W, MAX_H))
        print(f"[{year}] Active buildings: {len(precomp)} of {len(bld_city)} total in city")

        out_dir = os.path.join(city_root, str(year))
        raw_dir = os.path.join(debug_root, str(year), "_raw")
        saved_any = False
        saved_raw = 0

        print(f"[{year}] Territorial TIME={year}")
        total = len(precomp)
        stars_total = 20
        stars_printed = 0
        qc_rows = []
        for idx, (bid, geom, bbox, width, height) in enumerate(precomp, start=1):
            # progress status (one star every 5%)
            should_have = (idx * stars_total) // total  # integer floor
            while stars_printed < should_have:
                print("*", end="", flush=True)
                stars_printed += 1
            # quick overlap sanity
            if not sbox(*bbox).intersects(geom):
                print(f"[BBOX WARN] no overlap id={bid} bbox={bbox}")

            try:
                ds, arr = territorial_getmap_rgb(bbox, width, height, year)
            except Exception as e:
                print(f"  [WMS WARN {year}] id={bid}: {e}")
                continue

            # Save a few RAWs for visual check
            if saved_raw < DEBUG_FIRST:
                raw_png = os.path.join(raw_dir, f"{bid}_raw.png")
                save_png_array(raw_png, arr)
                saved_raw += 1
                # quick stats in console
                print(f"  [DEBUG] RAW min/max={arr.min()}/{arr.max()} → {raw_png}")

            cov, rmin, rmax, rstd = qc_metrics(arr, geom, ds.transform)

            # Mask outside building to white
            masked = white_outside_polygon(arr, geom, ds.transform)

            if not saved_any:
                os.makedirs(out_dir, exist_ok=True)

            out_png = os.path.join(out_dir, f"{bid}.png")
            try:
                save_png_array(out_png, masked)
                saved_any = True
            except Exception as e:
                print(f"  [WRITE WARN {year}] id={bid}: {e}")

            qc_rows.append({"id": bid, "year": year, "mask_coverage": cov,
                            "raw_min": rmin, "raw_max": rmax, "raw_std": rstd,
                            "path": out_png})
        print()
        if not saved_any:
            print(f"[SKIP {year}] No thumbnails saved (errors or empty results).")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
