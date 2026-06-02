# catastro_atom.py
from __future__ import annotations
import io, re, zipfile, unicodedata, requests, geopandas as gpd
import xml.etree.ElementTree as ET
from typing import List, Tuple

# from urllib.parse import urljoin

# ---- ATOM index candidates (try in order) ----
ATOM_BU_INDEX_CANDIDATES = [
    # Current, official (lowercase 'buildings')
    "https://www.catastro.hacienda.gob.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml",
    "http://www.catastro.hacienda.gob.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml",
    # Legacy/minhap fallback
    "https://www.catastro.minhap.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml",
    "http://www.catastro.minhap.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml",
    # (very old uppercase paths – rarely needed, but harmless to try)
    "https://www.catastro.hacienda.gob.es/INSPIRE/Buildings/ES.SDGC.BU.atom.xml",
    "http://www.catastro.hacienda.gob.es/INSPIRE/Buildings/ES.SDGC.BU.atom.xml",
]

UA = "Thumbs3/1.0 (+contact: you@example.com)"
NS = {"a": "http://www.w3.org/2005/Atom"}

# Common articles used in Spanish/Catalan/Galician/Portuguese municipality names
ARTICLES = ["el", "la", "los", "las", "lo", "l", "els", "les", "o", "a", "os", "as"]  # ES/CA/GL/PT tiny set

# ------------- utils: normalization & canonicalization -----------------

ARTICLES = ["el", "la", "los", "las", "lo", "l", "els", "les", "o", "a", "os", "as"]


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm(s: str) -> str:
    s = _strip_accents(s).lower()
    s = s.replace("’", "'")
    # collapse punctuation to spaces (keep () and / and - for now)
    s = re.sub(r"[^\w\s()/-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canon_name(raw: str) -> str:
    """
    Canonicalize both user input and ATOM titles:
      - remove leading numeric code (e.g. '43102-')
      - remove trailing 'buildings'
      - move trailing article (with or without parentheses) to front
      - split L'… -> 'l …'
      - strip accents, lowercase, collapse spaces
    """
    s = _norm(raw)

    # 1) strip leading code like "43102-" (4+ digits then dash)
    s = re.sub(r"^\d{4,}\s*-\s*", "", s)

    # 2) remove trailing word 'buildings'
    s = re.sub(r"\bbuildings\b$", "", s).strip()

    # 3a) trailing article with parentheses: "foo (els)" -> "els foo"
    m = re.match(r"^(.*)\s+\((%s)\)$" % "|".join(ARTICLES), s)
    if m:
        core, art = m.groups()
        s = f"{art} {core}".strip()
    else:
        # 3b) trailing article without parentheses: "foo els" -> "els foo"
        m2 = re.match(r"^(.*)\s+(%s)$" % "|".join(ARTICLES), s)
        if m2:
            core, art = m2.groups()
            s = f"{art} {core}".strip()

    # 4) split "l'espluga" (already accentless) -> "l espluga"
    s = re.sub(r"\bl['’]\s*", "l ", s)

    # 5) turn dashes into spaces and collapse again
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------- ATOM helpers ---------------------

def _fetch_xml(url: str) -> ET.Element:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return ET.fromstring(r.content)


def _open_first_working_atom(candidates: List[str]) -> ET.Element:
    last_err = None
    for u in candidates:
        try:
            return _fetch_xml(u)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"None of the ATOM Buildings index URLs worked. Last error: {last_err}")


def _atom_entries(feed: ET.Element) -> List[Tuple[str, str]]:
    """Yield (title, href) for each entry in an ATOM feed."""
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in feed.findall("a:entry", ns):
        title = e.findtext("a:title", default="", namespaces=ns) or ""
        link = e.find("a:link", ns)
        href = link.get("href") if link is not None else None
        if title and href:
            out.append((title, href))
    return out


# --------------------- Public API ----------------------

def list_municipalities(province: str | None = None, article: str | None = None) -> List[str]:
    """
    Return ATOM entry titles for a province (or all), optionally filtered by article
    (e.g., 'els' will return titles like 'PALLARESOS (ELS)' and those starting with 'ELS ').
    """
    root = _open_first_working_atom(ATOM_BU_INDEX_CANDIDATES)
    prov_entries = _atom_entries(root)

    if province:
        wanted_prov = _norm(province)
        prov_entries = [p for p in prov_entries if wanted_prov in _norm(p[0])]
        if not prov_entries:
            raise RuntimeError(f"Province '{province}' not found in Buildings ATOM index")

    results: List[str] = []
    for prov_title, prov_href in prov_entries:
        feed_xml = _fetch_xml(prov_href)
        for title, _ in _atom_entries(feed_xml):
            if article:
                a = _norm(article)
                tnorm = _norm(title)
                if not (re.search(rf"\(\s*{a}\s*\)", tnorm) or tnorm.startswith(f"{a} ")):
                    continue
            results.append(title.strip())
    return sorted(set(results), key=str.upper)


def download_municipality_buildings(municipality: str, province: str | None = None) -> gpd.GeoDataFrame:
    """
    Download INSPIRE Buildings for a municipality via ATOM.
    Returns a GeoDataFrame (CRS as in the GML, often EPSG:4258; reproject as needed).
    Matching is robust to article position, accents, and spacing.
    """
    root = _open_first_working_atom(ATOM_BU_INDEX_CANDIDATES)
    prov_entries = _atom_entries(root)

    # Limit to one province if provided
    if province:
        wanted_prov = _norm(province)
        prov_entries = [p for p in prov_entries if wanted_prov in _norm(p[0])]
        if not prov_entries:
            raise RuntimeError(f"Province '{province}' not found in Buildings ATOM index")

    wanted = _canon_name(municipality)
    # last_titles_sample: List[str] = []

    # Scan province feeds for the municipality entry
    for prov_title, prov_href in prov_entries:
        feed_xml = _fetch_xml(prov_href)
        entries = _atom_entries(feed_xml)
        # last_titles_sample = [t for t, _ in entries[:12]]  # keep a few for debug

        for title, href in entries:
            title_can = _canon_name(title)
            # exact canonical match preferred; substring fallback helps with odd feed tails
            if title_can == wanted or wanted in title_can:
                z = requests.get(href, timeout=120)
                z.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
                    gml_names = [n for n in zf.namelist() if n.lower().endswith(".gml")]
                    if not gml_names:
                        raise RuntimeError("ZIP has no GML file.")
                    with zf.open(gml_names[0]) as gmlf:
                        return gpd.read_file(gmlf)
    # if last_titles_sample:
    #     # No match → show a few examples to help the user see feed spelling
    #     print("[ATOM DEBUG] Entries in last province scanned:")
    #     list_municipalities (province)
    raise RuntimeError(f"Municipality '{municipality}' not found in ATOM feeds (province={province or 'ANY'}).")


# --------------------- CLI smoke tests (optional) ----------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CATATRO ATOM helper")
    ap.add_argument("--list", action="store_true", help="List municipalities in a province")
    ap.add_argument("--province", default=None, help="Province name to list/filter (e.g., 'Tarragona')")
    ap.add_argument("--city", default=None, help="Municipality to download (returns count only)")
    args = ap.parse_args()

    if args.list:
        for t in list_municipalities(province=args.province):
            print(t)
    elif args.city:
        gdf = download_municipality_buildings(args.city, province=args.province)
        print(f"Downloaded {len(gdf)} building features.")
    else:
        ap.print_help()
