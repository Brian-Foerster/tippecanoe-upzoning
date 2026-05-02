"""
Convert upzoning_scores.geojson → upzoning.pmtiles using GDAL's PMTiles
driver (bundled with pyogrio). Output goes to the repo root so index.html
can reference it with a simple relative path.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pyogrio

IN = Path("data/upzoning_scores.geojson")
OUT = Path("upzoning.pmtiles")

# Only these columns end up in the tiles — keeps file size down
KEEP = [
    "tier", "ILR", "land_value_per_sqft",
    "Address", "ClassCode", "land_av", "imp_av",
    "geometry",
]

if not IN.exists():
    sys.exit(f"ERROR: {IN} not found — run analyze.py first")

print(f"Reading {IN} ({IN.stat().st_size / 1e6:.0f} MB)...")
gdf = gpd.read_file(IN)
out = gdf[[c for c in KEEP if c in gdf.columns]].copy()
print(f"  {len(out):,} features")

print(f"Writing {OUT}...")
pyogrio.write_dataframe(
    out,
    str(OUT),
    layer="parcels",
    dataset_options={
        "MINZOOM": "10",
        "MAXZOOM": "16",
    },
)

size_mb = OUT.stat().st_size / 1e6
print(f"Done — {OUT} is {size_mb:.1f} MB")
if size_mb > 90:
    print("WARNING: file is large for GitHub Pages (>90 MB) — consider Git LFS")
