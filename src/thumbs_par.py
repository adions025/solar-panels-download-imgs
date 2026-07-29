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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import geopandas as gpd
import pandas as pd

from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box as sbox
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union
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

SUBMUNICIPAL_AOIS = {
    "valldoreix": {
        "cadastre_city": "Sant Cugat del Vallès",
        "aoi_file": "result.geojson",
    },
}

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
    p.add_argument("--cadastre-city", default="",
                   help="Municipality to download from Cadastre when --city is a submunicipal AOI.")
    p.add_argument("--atom-cache", default="",
                   help="Optional GPKG path to cache the ATOM download (read next runs).")
    p.add_argument("--gt-polygons", default="",
                   help="Optional GPKG with GT/building polygons. Uses its 'id' column for output names.")
    p.add_argument("--gt-layer", default="",
                   help="Layer name for --gt-polygons. Defaults to the first layer.")
    p.add_argument("--gt-filter", choices=["all", "positive"], default="all",
                   help="For --gt-polygons: process all rows or only rows with GT_YEAR != 0.")
    p.add_argument("--output-prefix", default="build",
                   help="Output filename prefix (default: build). Use 'raster' to write raster_<id>.png.")
    p.add_argument("--omit-year-in-name", action="store_true",
                   help="Do not append _YEAR to output filenames.")
    p.add_argument("--ids-from-dir", nargs="*", default=[],
                   help="Optional folder(s) with raster/build PNGs; only matching IDs will be processed.")
    p.add_argument("--keep-multipart-buildings", action="store_true",
                   help="Do not split cadastral MultiPolygon buildings into separate images.")
    p.add_argument("--debug-first", type=int, default=DEFAULT_DEBUG_FIRST,
                   help="Save RAW WMS for first N buildings per year (under OUTROOT/_debug/..).")
    p.add_argument("--limit", type=int, default=0, help="Process at most this many buildings (0=all).")
    p.add_argument("--alta-field", default="FECHAALTA",
                   help="Field name for building start date (YYYYMMDD).")
    p.add_argument("--baja-field", default="FECHABAJA",
                   help="Field name for building end date (YYYYMMDD or 99999999 for active).")
    p.add_argument("--aoi-mode", choices=["muni", "from-cadastre", "bbox", "file"], default="muni",
                   help="Area-of-interest source: municipality (default), union of cadastre geometries, or explicit bbox.")
    p.add_argument("--aoi-file", default="",
                   help="AOI boundary file (GeoJSON/Shapefile/GPKG) used with --aoi-mode=file.")
    p.add_argument("--aoi-bbox", default="",
                   help="AOI bbox as 'minx,miny,maxx,maxy' (use with --aoi-mode=bbox).")
    p.add_argument("--aoi-crs", default="EPSG:25831",
                   help="CRS of --aoi-bbox (default EPSG:25831).")
    p.add_argument("--max-workers", type=int, default=4,
                   help="Maximum concurrent years (default: 4, capped to number of years).")
    return p.parse_args()


def _city_key(city: str) -> str:
    return re.sub(r"\s+", " ", city.strip().replace("_", " ")).casefold()


def _safe_filename_part(value) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return text.strip("_")


def _cadastre_feature_id(row) -> str:
    for field in ("localId", "LOCALID", "gml_id", "GML_ID", "ID", "id"):
        if field in row and pd.notna(row[field]):
            value = re.sub(r"\D+", "", str(row[field]))
            if value:
                return value
    return ""


def _output_image_id(args: argparse.Namespace, row, original_id: str, idx: int, year: int) -> str:
    if args.gt_polygons:
        feature_id = row.get("output_base_id") or _cadastre_feature_id(row)
        name = f"{args.output_prefix}_{feature_id or idx}"
        return name if args.omit_year_in_name else f"{name}_{year}"
    return original_id


def _ids_from_dirs(paths: list[str]) -> set[str]:
    ids = set()
    pattern = re.compile(r"^(?:build|raster)_(?P<id>.+?)(?:_\d{4})?\.png$", re.IGNORECASE)

    for raw_path in paths:
        folder = os.path.abspath(os.path.expanduser(raw_path))
        if not os.path.isdir(folder):
            raise SystemExit(f"[INPUT] --ids-from-dir folder not found: {raw_path}")

        for name in os.listdir(folder):
            match = pattern.match(name)
            if not match:
                continue
            image_id = match.group("id")
            ids.add(image_id.split("_", 1)[0])
    return ids


def _filter_by_allowed_ids(gdf: gpd.GeoDataFrame, allowed_ids: set[str]) -> gpd.GeoDataFrame:
    if not allowed_ids:
        return gdf

    def row_id(row) -> str:
        return str(row.get("source_feature_id") or row.get("output_base_id") or row.get("id")).split("_", 1)[0]

    mask = gdf.apply(lambda row: row_id(row) in allowed_ids, axis=1)
    return gdf.loc[mask].copy()


def _polygon_parts(geom: BaseGeometry):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [part for part in geom.geoms if not part.is_empty]
    if hasattr(geom, "geoms"):
        parts = []
        for part in geom.geoms:
            parts.extend(_polygon_parts(part))
        return parts
    return []


def _explode_building_parts(
    bld_city: gpd.GeoDataFrame,
    clip_geom: BaseGeometry | None = None,
    label: str = "AOI",
) -> gpd.GeoDataFrame:
    rows = []
    id_counts: dict[str, int] = {}

    for fallback_idx, (_, row) in enumerate(bld_city.iterrows(), start=1):
        source_id = _cadastre_feature_id(row)
        geom = row.geometry.intersection(clip_geom) if clip_geom is not None else row.geometry
        for part in _polygon_parts(geom):
            if part.is_empty:
                continue
            base_id = source_id or str(fallback_idx)
            id_counts[base_id] = id_counts.get(base_id, 0) + 1

            new_row = row.copy()
            new_row.geometry = part
            new_row["source_feature_id"] = source_id
            new_row["feature_part"] = id_counts[base_id]
            new_row["output_base_id"] = f"{base_id}_{id_counts[base_id]}"
            rows.append(new_row)

    if not rows:
        raise SystemExit(f"[{label}] No polygonal building parts remain after clipping/exploding.")

    out = gpd.GeoDataFrame(rows, crs=bld_city.crs)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    return out.reset_index(drop=True)


def write_output_id_building_index(bld_city: gpd.GeoDataFrame, outdir: str):
    out_csv = os.path.join(outdir, "building_index.csv")
    out_gpkg = os.path.join(outdir, "building_index.gpkg")
    rows = []
    for _, row in bld_city.iterrows():
        gid = str(row.get("output_base_id") or row.get("id"))
        c = row.geometry.centroid
        rows.append({
            "id": gid,
            "area_m2": float(row.geometry.area),
            "centroid_e": float(c.x),
            "centroid_n": float(c.y),
        })
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates("id").to_csv(out_csv, index=False)

    g = bld_city.copy()
    g["id"] = g["output_base_id"].astype(str)
    g[["id", "geometry"]].to_file(out_gpkg, layer="buildings", driver="GPKG")
    log(f"[INDEX] wrote {out_csv} and {out_gpkg}")


def _find_aoi_file(path: str) -> str:
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(os.path.dirname(os.getcwd()), path),
        os.path.join(os.path.dirname(__file__), path),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise SystemExit(f"[INPUT] --aoi-file not found: {path}")


def _find_existing_file(path: str) -> str:
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(os.path.dirname(os.getcwd()), path),
        os.path.join(os.path.dirname(__file__), path),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return ""


def _line_to_polygon(geom: BaseGeometry):
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        if len(coords) < 3:
            return geom
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        return poly.buffer(0) if not poly.is_valid else poly
    if isinstance(geom, MultiLineString):
        polygons = []
        closed_lines = []
        for line in geom.geoms:
            line_poly = _line_to_polygon(line)
            if isinstance(line_poly, (Polygon, MultiPolygon)):
                polygons.append(line_poly)
            else:
                closed_lines.append(line)
        if polygons:
            return unary_union(polygons).buffer(0)
        polygons = list(polygonize(unary_union(closed_lines)))
        if polygons:
            return unary_union(polygons).buffer(0)
    return geom


def _to_polygonal_aoi(gdf: gpd.GeoDataFrame):
    geoms = [_line_to_polygon(geom) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    polygonal = [
        geom.buffer(0) for geom in geoms
        if isinstance(geom, (Polygon, MultiPolygon)) and not geom.is_empty
    ]
    polygonal = [geom for geom in polygonal if geom.is_valid and not geom.is_empty]
    if polygonal:
        return unary_union(polygonal).buffer(0)

    lines = [
        geom for geom in geoms
        if isinstance(geom, (LineString, MultiLineString)) and not geom.is_empty
    ]
    polygons = list(polygonize(unary_union(lines))) if lines else []
    if polygons:
        return unary_union(polygons).buffer(0)
    raise SystemExit("[AOI:file] Boundary file does not contain polygonal or closed line geometry.")


def load_aoi_file(path: str):
    aoi_path = _find_aoi_file(path)
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        raise SystemExit(f"[AOI:file] File has no CRS: {aoi_path}")
    gdf = ensure_epsg25831(gdf)
    return _to_polygonal_aoi(gdf), aoi_path


def apply_submunicipal_defaults(a: argparse.Namespace) -> argparse.Namespace:
    defaults = SUBMUNICIPAL_AOIS.get(_city_key(a.city))
    if not defaults:
        return a

    if not a.cadastre_city:
        a.cadastre_city = defaults["cadastre_city"]
    if a.aoi_mode == "muni" and not a.aoi_file:
        a.aoi_mode = "file"
        a.aoi_file = defaults["aoi_file"]
    return a


def load_gt_polygons(a: argparse.Namespace) -> gpd.GeoDataFrame:
    import fiona

    gt_path = _find_existing_file(a.gt_polygons)
    if not gt_path:
        raise SystemExit(f"[INPUT] --gt-polygons file not found: {a.gt_polygons}")

    layer = a.gt_layer or fiona.listlayers(gt_path)[0]
    gdf = ensure_epsg25831(gpd.read_file(gt_path, layer=layer))
    if "id" not in gdf.columns:
        raise SystemExit(f"[GT] Layer '{layer}' in {gt_path} has no 'id' column.")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    rows = []
    for _, row in gdf.iterrows():
        base_id = re.sub(r"\D+", "", str(row["id"]))
        if not base_id:
            continue

        parts = _polygon_parts(row.geometry)
        if not parts:
            continue

        for part_idx, part in enumerate(parts, start=1):
            new_row = row.copy()
            new_row.geometry = part
            new_row["source_feature_id"] = base_id
            new_row["feature_part"] = part_idx
            new_row["output_base_id"] = base_id if len(parts) == 1 else f"{base_id}_{part_idx}"
            rows.append(new_row)

    out = gpd.GeoDataFrame(rows, crs=gdf.crs)
    log(f"[GT] Loaded {len(gdf)} rows from {gt_path} layer '{layer}'")
    log(f"[GT] Polygon parts after multipart explode: {len(out)}")
    return out.reset_index(drop=True)


def load_buildings(a: argparse.Namespace, city: str) -> gpd.GeoDataFrame:
    if a.gt_polygons:
        return load_gt_polygons(a)

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
    if a.aoi_mode == "file":
        if not a.aoi_file:
            raise SystemExit("--aoi-file is required when --aoi-mode=file")
        return load_aoi_file(a.aoi_file)
    raise SystemExit("Unknown --aoi-mode")


def process_year(
    year: int,
    bld_city: gpd.GeoDataFrame,
    city_root: str,
    debug_root: str,
    bld_all_len: int,
    args: argparse.Namespace,
):
    if args.gt_polygons:
        gt_col = f"GT_{year}"
        if args.gt_filter == "positive":
            if gt_col not in bld_city.columns:
                raise SystemExit(f"[GT] --gt-filter=positive requires column '{gt_col}'.")
            bld_year = bld_city.loc[bld_city[gt_col].fillna(0) != 0].copy()
            log(f"[{year}] GT polygons with {gt_col} != 0: {len(bld_year)}")
        else:
            bld_year = bld_city.copy()
    else:
        bld_year = filter_by_cadastre_year(
            bld_city, year, alta_field=args.alta_field, baja_field=args.baja_field
        )
    if bld_year.empty:
        log(f"[{year}] No active buildings for this year (after FECHAALTA/FECHABAJA filter). Skipping.")
        return

    bld_jobs = bld_year.reset_index(drop=True)
    precomp = precompute_jobs(bld_jobs, args.mpp, args.margin, max_side=max(MAX_W, MAX_H))
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
    for idx, (job, (_, row)) in enumerate(zip(precomp, bld_jobs.iterrows()), start=1):
        bid, geom, bbox, width, height = job
        row_data = row.to_dict()
        out_id = _output_image_id(args, row_data, bid, idx, year)
        should_have = (idx * stars_total) // total
        while stars_printed < should_have:
            log(f"[{year}] *")
            stars_printed += 1
        if not sbox(*bbox).intersects(geom):
            log(f"[{year}] [BBOX WARN] no overlap; id={out_id} bbox={bbox}")

        try:
            ds, arr = territorial_getmap_rgb(bbox, width, height, year)
        except Exception as e:  # noqa: BLE001
            log(f"[{year}] [WMS WARN] id={out_id}: {e}")
            continue

        if saved_raw < args.debug_first:
            raw_png = os.path.join(raw_dir, f"{out_id}_raw.png")
            save_png_array(raw_png, arr)
            saved_raw += 1
            log(f"[{year}] [DEBUG] RAW min/max={arr.min()}/{arr.max()} → {raw_png}")

        cov, rmin, rmax, rstd = qc_metrics(arr, geom, ds.transform)
        out_arr = white_outside_polygon(arr, geom, ds.transform)

        if not saved_any:
            os.makedirs(out_dir, exist_ok=True)

        out_png = os.path.join(out_dir, f"{out_id}.png")
        try:
            save_png_array(out_png, out_arr)
            saved_any = True
        except Exception as e:  # noqa: BLE001
            log(f"[{year}] [WRITE WARN] id={out_id}: {e}")

        qc_rows.append({
            "id": out_id,
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
    a = apply_submunicipal_defaults(parse_args())
    CITY = a.city
    CADASTRE_CITY = a.cadastre_city or CITY
    YEAR_INI = a.year_ini
    YEAR_END = a.year_end

    if a.gt_polygons:
        log(f"[GT] Using GT polygons for city/AOI '{CITY}'")
    elif CADASTRE_CITY != CITY:
        log(f"[CADASTRE] Using municipality '{CADASTRE_CITY}' for city/AOI '{CITY}'")
    bld = load_buildings(a, CADASTRE_CITY)
    if not a.gt_polygons:
        bld = clean_cadastre_records(bld, alta_field=a.alta_field, baja_field=a.baja_field)

    aoi_geom, aoi_label = resolve_aoi(a, CITY, bld)
    area_km2 = round(aoi_geom.area / 1e6, 3)
    minx, miny, maxx, maxy = aoi_geom.bounds
    log(f"[AOI] {a.aoi_mode}='{aoi_label}' | area ≈ {area_km2} km² | bounds 25831 = ({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")

    bld_city = robust_city_clip(bld, aoi_geom)
    log(f"[CLIP] Buildings in file: {len(bld)}; within AOI: {len(bld_city)}")
    if len(bld_city) == 0:
        raise SystemExit("[CLIP ERROR] No buildings intersect the AOI.")

    city_key = _city_key(CITY)
    if not a.gt_polygons and not a.keep_multipart_buildings:
        before = len(bld_city)
        clip_geom = aoi_geom if city_key == "valldoreix" else None
        bld_city = _explode_building_parts(bld_city, clip_geom=clip_geom, label=CITY)
        log(f"[{CITY}] Exploded polygon parts: {before} features -> {len(bld_city)} images")

    if a.ids_from_dir:
        allowed_ids = _ids_from_dirs(a.ids_from_dir)
        before = len(bld_city)
        bld_city = _filter_by_allowed_ids(bld_city, allowed_ids)
        log(f"[IDS] Loaded {len(allowed_ids)} IDs from folder(s); selected {len(bld_city)} of {before} geometries")
        if len(bld_city) == 0:
            raise SystemExit("[IDS] No geometries match the IDs from --ids-from-dir.")

    if a.limit > 0 and len(bld_city) > a.limit:
        bld_city = bld_city.head(a.limit).copy()
        log(f"[LIMIT] Limiting to first {a.limit} buildings for this run.")

    city_root = os.path.join(a.outroot, CITY.replace(" ", "_"))
    debug_root = os.path.join(a.outroot, "_debug", CITY.replace(" ", "_"))
    if a.gt_polygons:
        write_output_id_building_index(bld_city, city_root)
    else:
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
