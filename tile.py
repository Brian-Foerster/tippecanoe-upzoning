"""
Convert GeoJSON outputs to PMTiles using GDAL's PMTiles driver (via pyogrio).
Tiles both parcel-level and ownership-merged site-level data.

Run after analyze.py and merge_ownership.py.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pyogrio

JOBS = [
    {
        "in":      Path("data/upzoning_scores.geojson"),
        "out":     Path("upzoning.pmtiles"),
        "layer":   "parcels",
        "keep":    ["ILR", "land_value_per_sqft", "Address", "ClassCode", "land_av", "imp_av"],
        "minzoom": "10",
        "maxzoom": "16",
        "requires": "analyze.py",
    },
    {
        "in":      Path("data/upzoning_sites.geojson"),
        "out":     Path("upzoning_sites.pmtiles"),
        "layer":   "sites",
        "keep":    ["ILR", "land_value_per_sqft", "addresses", "owner",
                    "n_parcels", "ClassCode", "land_av", "imp_av", "upzoning_score"],
        "minzoom": "10",
        "maxzoom": "16",
        "requires": "merge_ownership.py",
    },
]

any_missing = False
for job in JOBS:
    if not job["in"].exists():
        print(f"SKIP {job['out'].name} — {job['in']} not found (run {job['requires']} first)")
        any_missing = True

if any_missing and all(not j["in"].exists() for j in JOBS):
    sys.exit("No input files found.")

for job in JOBS:
    if not job["in"].exists():
        continue

    src = job["in"]
    dst = job["out"]
    print(f"\n{'-'*50}")
    print(f"Reading {src} ({src.stat().st_size / 1e6:.0f} MB)...")
    gdf = gpd.read_file(src)
    keep = ["geometry"] + [c for c in job["keep"] if c in gdf.columns]
    out  = gdf[keep].copy()
    print(f"  {len(out):,} features, {len(keep)-1} attribute columns")

    if dst.exists():
        dst.unlink()

    print(f"Writing {dst}...")
    pyogrio.write_dataframe(
        out,
        str(dst),
        layer=job["layer"],
        dataset_options={"MINZOOM": job["minzoom"], "MAXZOOM": job["maxzoom"]},
    )

    size_mb = dst.stat().st_size / 1e6
    print(f"Done — {dst} is {size_mb:.1f} MB")
    if size_mb > 90:
        print("WARNING: >90 MB — consider Git LFS for GitHub Pages")
