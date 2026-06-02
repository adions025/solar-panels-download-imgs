# icgc_territorial.py
from __future__ import annotations
import requests
import unicodedata
import geopandas as gpd
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from osm_admin import get_admin_boundary_gdf

TARGET_CRS = "EPSG:25831"

# Full coverage WMS (supports TIME=year)
TERR_WMS_URL = "https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms"
TERR_LAYER_RGB_TIME = "ortofoto_color_serie_anual"  # WMS-Time series layer

# ArcGIS REST municipalities (returns GeoJSON in EPSG:25831)
ARCGIS_MUNIS_QUERY = (
    "https://geoserveis.icgc.cat/vector01/rest/services/divisions_administratives_wfs/MapServer/2/query"
)

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def get_municipality_gdf(city_name: str, province_name: str | None = None) -> gpd.GeoDataFrame:
    """
    Robust municipality lookup via ArcGIS REST (EPSG:25831).
    - Handles short names (e.g., 'Sant Cugat').
    - Picks best candidate by string score (exact > startswith > contains), then biggest area.
    - If no good match, raises with a few suggestions.
    """
    q = city_name.strip()
    qn = _norm(q)

    # Build a WHERE filter; ArcGIS layer has NOMMUNI and NOMCOMAR (comarca); there isn’t a province field here,
    # so province filtering is best done on the client if you have a list of acceptable names.
    def _search(where: str) -> gpd.GeoDataFrame:
        params = {
            "where": where,
            "outFields": "NOMMUNI,NOMCOMAR",
            "returnGeometry": "true",
            "outSR": "25831",
            "f": "geojson",
        }
        r = requests.get(ARCGIS_MUNIS_QUERY, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        feats = data.get("features", [])
        if not feats:
            return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
        gdf = gpd.GeoDataFrame.from_features(feats, crs=TARGET_CRS)
        return gdf

    # Try progressively looser server-side filters
    gdf = _search(f"UPPER(NOMMUNI) = '{q.upper()}'")
    if gdf.empty:
        # contains (case-insensitive)
        gdf = _search(f"NOMMUNI LIKE '%{q.upper()}%'")

    if gdf.empty:
        raise RuntimeError(f"Municipality '{city_name}' not found by the ICGC service.")

    # Score by text similarity
    def _score(name: str) -> int:
        n = _norm(name)
        if n == qn: return 100
        if n.startswith(qn): return 80
        if qn in n: return 60
        return 0

    gdf["__score"] = gdf["NOMMUNI"].astype(str).map(_score)
    # If province_name is given, gently prefer candidates whose comarca string contains it
    if province_name:
        pn = _norm(province_name)
        gdf["__prov_bonus"] = gdf["NOMCOMAR"].astype(str).map(lambda s: 10 if pn in _norm(s) else 0)
    else:
        gdf["__prov_bonus"] = 0

    # Compute area for tie-breakers (bigger first)
    # with gpd.option_context("mode.copy_on_write", False):
    gdf["__area"] = gdf.geometry.area

    # Pick best
    gdf = gdf.sort_values(["__score", "__prov_bonus", "__area"], ascending=[False, False, False])
    best = gdf.head(1).drop(columns=["__score", "__prov_bonus", "__area"]).reset_index(drop=True)

    # If the best score is low (no real textual hit), produce a helpful error with suggestions
    top_score = gdf["__score"].iloc[0]
    if top_score == 0:
        suggestions = ", ".join(gdf["NOMMUNI"].head(5).tolist())
        raise RuntimeError(
            f"Municipality '{city_name}' ambiguous or not found. Did you mean one of: {suggestions} ? "
            f"Try the full name (e.g., 'Sant Cugat del Valles' / 'Sant Cugat del Vallès')."
        )

    return best

def territorial_getmap_rgb(
    bbox_25831: tuple[float, float, float, float],
    width: int,
    height: int,
    year: int,
):
    """
    Fetch a Territorial orthophoto RGB tile (TIME=year) as a georeferenced in-memory raster.
    Uses WMS 1.1.1 (SRS=EPSG:25831).
    Returns (dataset, array) where array is uint8 [bands, H, W].
    """
    minx, miny, maxx, maxy = bbox_25831
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": TERR_LAYER_RGB_TIME,
        "STYLES": "",
        "SRS": TARGET_CRS,
        "BBOX": f"{minx:.3f},{miny:.3f},{maxx:.3f},{maxy:.3f}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/png",
        "TRANSPARENT": "FALSE",
        "TIME": str(int(year)),
        "EXCEPTIONS": "application/vnd.ogc.se_xml",
    }
    r = requests.get(TERR_WMS_URL, params=params, timeout=90)
    r.raise_for_status()
    t0 = r.text.lstrip()[:200].lower()
    if t0.startswith("<?xml") and "exception" in t0:
        raise RuntimeError(f"WMS Exception (Territorial, {year}): {r.text[:300]} ...")

    with MemoryFile(r.content) as mf:
        with mf.open() as src:
            arr = src.read()
            count = src.count

    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    mem = MemoryFile()
    ds = mem.open(
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=count,
        dtype="uint8",
        crs=TARGET_CRS,
        transform=transform,
    )
    ds.write(arr)
    return ds, arr

def compute_bbox_with_margin(geom, margin_m: float):
    minx, miny, maxx, maxy = geom.bounds
    return (minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m)

def pick_image_size(bbox: tuple[float, float, float, float], m_per_px: float, max_px: int = 2048) -> tuple[int, int]:
    minx, miny, maxx, maxy = bbox
    w_m = maxx - minx
    h_m = maxy - miny
    w_px = max(1, int(round(w_m / m_per_px)))
    h_px = max(1, int(round(h_m / m_per_px)))
    scale = max(w_px / max_px, h_px / max_px, 1.0)
    return int(w_px / scale), int(h_px / scale)
