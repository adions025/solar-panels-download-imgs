#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel yearly thumbnail generation for ICGC Territorial orthophotos.
Runs the per-year work in parallel to speed up long ranges; inside each year,
buildings are still processed sequentially to avoid overloading the WMS.
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import geopandas as gpd
import pandas as pd

from shapely.geometry import box as sbox
from shapely.ops import unary_union
from rasterio.errors import NotGeoreferencedWarning

from icgc_old import get_municipality_gdf, territorial_getmap_rgb
from catastro_atom import download_municipality_buildings
from masking import white_outside_polygon
from ids import geometry_hash_id

# Reuse helpers from the sequential script to avoid code drift
from thumbs import (
    DEFAULT_MARGIN_M,
    UNIFORM_MPP,
    MAX_W,
    MAX_H,
    DEFAULT_OUTROOT,
    DEFAULT_DEBUG_FIRST,
    ensure_epsg25831,
    clean_cadastre_records,
    filter_by_cadastre_year,
    robust_city_clip,
    precompute_jobs,
    save_png_array,
    write_building_index,
    qc_metrics,
    _parse_bbox,
)

warnings.simplefilter("ignore", NotGeoreferencedWarning)
print_lock = Lock()

def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel thumbnails from ICGC Territorial orthophoto (per-year concurrency)."
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
    p.add_argument("--max-workers", type=int, default=4,
                   help="Maximum concurrent years (default: 4, capped to number of years).")
    return p.parse_args()


def load_buildings(a: argparse.Namespace, city: str) -> gpd.GeoDataFrame:
    if a.cadastre_source == "file":
        cad = a.cadastre
        if not cad or not os.path.isfile(cad):
            raise SystemExit(f"[INPUT] --cadastre file not found: {cad}")
        bld = ensure_epsg25831(gpd.read_file(cad))
        log(f"[CADASTRE:file] Loaded {len(bld)} features from {cad}")
        return bld

    cache_path = os.path.expanduser(a.atom_cache) if a.atom_cache else ""
    if cache_path and os.path.isfile(cache_path):
        bld = ensure_epsg25831(gpd.read_file(cache_path))
        log(f"[CADASTRE:atom] Loaded {len(bld)} features from cache {cache_path}")
        return bld

    from catastro_atom import list_municipalities  # import here to avoid cycles
    try:
        log(f"[CADASTRE:atom] Downloading INSPIRE Buildings for '{city}'"
            + (f", province '{a.province}'" if a.province else "") + " …")
        bld = download_municipality_buildings(city, province=a.province).to_crs("EPSG:25831")
        log(f"[CADASTRE:atom] Downloaded {len(bld)} features.")
        if cache_path:
            try:
                bld.to_file(cache_path, layer="buildings", driver="GPKG")
                log(f"[CADASTRE:atom] Cached to {cache_path}")
            except Exception as e:  # noqa: BLE001
                log(f"[CADASTRE:atom] Cache write failed: {e}")
        return bld
    except Exception as e:
        log(f"[CADASTRE:atom] ERROR: {e}")
        if a.province:
            try:
                names = list_municipalities(province=a.province)
                log(f"[CADASTRE:atom] ATOM municipalities for province '{a.province}':")
                for name in names:
                    log(f"   - {name}")
            except Exception as e2:  # noqa: BLE001
                log(f"[CADASTRE:atom] Also failed listing province '{a.province}': {e2}")
        raise


def resolve_aoi(a: argparse.Namespace, city: str, bld: gpd.GeoDataFrame):
    if a.aoi_mode == "muni":
        muni = ensure_epsg25831(get_municipality_gdf(city))
        return muni.geometry.iloc[0], city
    if a.aoi_mode == "from-cadastre":
        uu = unary_union(bld.geometry)
        return (uu if uu.is_valid and not uu.is_empty else bld.unary_union).buffer(0).envelope, "cadastre-union"
    if a.aoi_mode == "bbox":
        if not a.aoi_bbox:
            raise SystemExit("--aoi-bbox is required when --aoi-mode=bbox")
        bbox = _parse_bbox(a.aoi_bbox)
        aoi = ensure_epsg25831(gpd.GeoDataFrame(geometry=[sbox(*bbox)], crs=a.aoi_crs))
        return aoi.geometry.iloc[0], "custom-bbox"
    raise SystemExit("Unknown --aoi-mode")


def process_year(
    year: int,
    bld_city: gpd.GeoDataFrame,
    city_root: str,
    debug_root: str,
    bld_all_len: int,
    args: argparse.Namespace,
):
    bld_year = filter_by_cadastre_year(
        bld_city, year, alta_field=args.alta_field, baja_field=args.baja_field
    )
    if bld_year.empty:
        log(f"[{year}] No active buildings for this year (after FECHAALTA/FECHABAJA filter). Skipping.")
        return

    precomp = precompute_jobs(bld_year, args.mpp, args.margin, max_side=max(MAX_W, MAX_H))
    log(f"[{year}] Active buildings: {len(precomp)} of {bld_all_len} total in city")

    out_dir = os.path.join(city_root, str(year))
    raw_dir = os.path.join(debug_root, str(year), "_raw")
    saved_any = False
    saved_raw = 0

    log(f"[{year}] Territorial TIME={year}")
    total = len(precomp)
    stars_total = 20
    stars_printed = 0
    qc_rows = []
    for idx, (bid, geom, bbox, width, height) in enumerate(precomp, start=1):
        should_have = (idx * stars_total) // total
        while stars_printed < should_have:
            log(f"[{year}] *")
            stars_printed += 1
        if not sbox(*bbox).intersects(geom):
            log(f"[{year}] [BBOX WARN] no overlap; id={bid} bbox={bbox}")

        try:
            ds, arr = territorial_getmap_rgb(bbox, width, height, year)
        except Exception as e:  # noqa: BLE001
            log(f"[{year}] [WMS WARN] id={bid}: {e}")
            continue

        if saved_raw < args.debug_first:
            raw_png = os.path.join(raw_dir, f"{bid}_raw.png")
            save_png_array(raw_png, arr)
            saved_raw += 1
            log(f"[{year}] [DEBUG] RAW min/max={arr.min()}/{arr.max()} → {raw_png}")

        cov, rmin, rmax, rstd = qc_metrics(arr, geom, ds.transform)
        masked = white_outside_polygon(arr, geom, ds.transform)

        if not saved_any:
            os.makedirs(out_dir, exist_ok=True)

        out_png = os.path.join(out_dir, f"{bid}.png")
        try:
            save_png_array(out_png, masked)
            saved_any = True
        except Exception as e:  # noqa: BLE001
            log(f"[{year}] [WRITE WARN] id={bid}: {e}")

        qc_rows.append({
            "id": bid,
            "year": year,
            "mask_coverage": cov,
            "raw_min": rmin,
            "raw_max": rmax,
            "raw_std": rstd,
            "path": out_png,
        })

    if not saved_any:
        log(f"[{year}] [SKIP] No thumbnails saved (errors or empty results).")
    else:
        log(f"[{year}] Done. Saved {len([r for r in qc_rows if os.path.isfile(r['path'])])} PNGs.")


def main():
    a = parse_args()
    CITY = a.city
    YEAR_INI = a.year_ini
    YEAR_END = a.year_end

    bld = load_buildings(a, CITY)
    bld = clean_cadastre_records(bld, alta_field=a.alta_field, baja_field=a.baja_field)

    aoi_geom, aoi_label = resolve_aoi(a, CITY, bld)
    area_km2 = round(aoi_geom.area / 1e6, 3)
    minx, miny, maxx, maxy = aoi_geom.bounds
    log(f"[AOI] {a.aoi_mode}='{aoi_label}' | area ≈ {area_km2} km² | bounds 25831 = ({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")

    bld_city = robust_city_clip(bld, aoi_geom)
    log(f"[CLIP] Buildings in file: {len(bld)}; within AOI: {len(bld_city)}")
    if len(bld_city) == 0:
        raise SystemExit("[CLIP ERROR] No buildings intersect the AOI.")

    if a.limit > 0 and len(bld_city) > a.limit:
        bld_city = bld_city.head(a.limit).copy()
        log(f"[LIMIT] Limiting to first {a.limit} buildings for this run.")

    city_root = os.path.join(a.outroot, CITY.replace(" ", "_"))
    debug_root = os.path.join(a.outroot, "_debug", CITY.replace(" ", "_"))
    write_building_index(bld_city, city_root)

    years = list(range(YEAR_INI, YEAR_END + 1))
    max_workers = max(1, min(a.max_workers, len(years)))
    log(f"[PARALLEL] Running {len(years)} years with max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                process_year,
                year,
                bld_city,
                city_root,
                debug_root,
                len(bld_city),
                a,
            ): year
            for year in years
        }
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                log(f"[{year}] ERROR: {e}")

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
