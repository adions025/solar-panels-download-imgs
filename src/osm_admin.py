# osm_admin.py
from __future__ import annotations
import time, unicodedata, requests, geopandas as gpd

NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = "YourApp/1.0 (contact: you@example.com)"  # be polite; set your email/app

def _norm(s:str)->str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower().strip()

def get_admin_boundary_gdf(place: str, country: str|None=None,
                           admin_level: int|None=8,
                           target_crs: str|None=None) -> gpd.GeoDataFrame:
    """
    Fetch an administrative boundary polygon from OSM via Nominatim.
    Returns 1-row GeoDataFrame (default EPSG:4326). Set target_crs to reproject.
    """
    q = f"{place}, {country}" if country else place
    params = {
        "q": q,
        "format": "jsonv2",
        "addressdetails": 1,
        "polygon_geojson": 1,
        "limit": 10,
    }
    r = requests.get(NOMINATIM, params=params, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    results = r.json()

    if not results:
        raise RuntimeError(f"No OSM result for '{q}'")

    # Prefer boundary=administrative, and matching admin_level if requested
    def score(rec):
        s = 0
        if rec.get("class") == "boundary" and rec.get("type") == "administrative": s += 3
        if admin_level and str(rec.get("extratags", {}).get("admin_level")) == str(admin_level): s += 2
        if country and _norm(country) in _norm(str(rec.get("display_name",""))): s += 1
        name = rec.get("name") or rec.get("display_name","")
        if _norm(place) in _norm(name): s += 1
        return s

    # Ensure extratags are present (ask details again if missing)
    for rec in results:
        if "extratags" not in rec:
            time.sleep(1.1)
            r2 = requests.get(f"https://nominatim.openstreetmap.org/details.php",
                              params={"osmtype": rec["osm_type"][0].upper(),
                                      "osmid": rec["osm_id"],
                                      "format": "json"},
                              headers={"User-Agent": UA}, timeout=30)
            if r2.ok:
                rec["extratags"] = r2.json().get("extratags", {})

    best = max(results, key=score)
    if "geojson" not in best:
        raise RuntimeError("OSM record lacks polygon geometry")

    gdf = gpd.GeoDataFrame.from_features([{
        "type":"Feature", "geometry": best["geojson"], "properties": best
    }], crs="EPSG:4326")

    if target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

