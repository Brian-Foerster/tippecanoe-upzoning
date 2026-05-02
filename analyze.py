"""
Land demand intensity screen for Tippecanoe County parcels.

Score = land_value_per_sqft  (pure demand signal)
underbuilt = 1 if ILR < UNDERBUILT_THRESHOLD  (binary filter, not continuous)

ILR is used only as a flag, not divided into the score. This avoids
conflating an unreliable assessor split with a demand signal and matches
practitioner convention (ESRI, Seattle OPCD, Auckland study) where ILR
is treated as a threshold filter, not a continuous regressor.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/Brian-Foerster/"
    "Greater-Lafayette-Corridor-Analysis/main/data/raw/land_av.geojson"
)
DATA_PATH = Path("data/land_av.geojson")
OUT_GEOJSON = Path("data/upzoning_scores.geojson")
OUT_CSV = Path("data/upzoning_scores.csv")

# Indiana State Plane East (feet) — matches the source data projection
CRS_PROJECTED = "EPSG:2966"

# ILR threshold for "underbuilt" flag.
# Literature standard is 1.0-1.5; county median is 6.2.
# 3.0 covers the bottom ~10% of the county distribution — parcels where
# the building is worth less than 3x the land, a meaningful local signal.
UNDERBUILT_THRESHOLD = 3.0


def fetch_data():
    if DATA_PATH.exists():
        print(f"Using cached {DATA_PATH}")
        return
    print("Downloading land_av.geojson (~83 MB)...")
    with requests.get(DATA_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(DATA_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%", end="", flush=True)
    print("\nDone.")


def compute_scores(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    gdf = gdf.copy()
    n_input = len(gdf)

    land = pd.to_numeric(gdf["land_av"], errors="coerce").fillna(0)
    imp = pd.to_numeric(gdf["imp_av"], errors="coerce").fillna(0)

    # Reproject to get area in square feet
    gdf_proj = gdf.to_crs(CRS_PROJECTED)
    gdf["area_sqft"] = gdf_proj.geometry.area

    # Drop unscreenable parcels
    MIN_AREA_SQFT = 500  # removes degenerate slivers; p1 of the county distribution
    no_land = land <= 0
    no_area = gdf["area_sqft"] < MIN_AREA_SQFT
    valid = ~no_land & ~no_area
    dropped = {"land_av=0": int(no_land.sum()), f"area<{MIN_AREA_SQFT}sqft": int(no_area.sum())}

    gdf = gdf[valid].copy()
    land = land[valid]
    imp = imp[valid]

    # Core metrics
    gdf["ILR"] = (imp / land).round(4)
    gdf["land_value_per_sqft"] = (land / gdf["area_sqft"]).round(4)

    # Score = pure land demand intensity (land $/sqft)
    gdf["upzoning_score"] = gdf["land_value_per_sqft"].round(4)

    # Underbuilt flag: ILR below threshold is a binary signal, not a divisor
    gdf["underbuilt"] = (gdf["ILR"] < UNDERBUILT_THRESHOLD).astype(int)

    # Decile tiers: 10 = highest demand, 1 = lowest
    gdf["tier"] = pd.qcut(gdf["upzoning_score"], q=10, labels=False, duplicates="drop") + 1

    dropped["total"] = n_input - len(gdf)
    return gdf, dropped


def main():
    fetch_data()

    print("Loading parcels...")
    gdf = gpd.read_file(DATA_PATH)
    print(f"  {len(gdf):,} parcels loaded")

    print("Computing scores...")
    gdf, dropped = compute_scores(gdf)
    n_dropped = sum(v for k, v in dropped.items() if k != "total")
    detail = ", ".join(f"{k}: {v}" for k, v in dropped.items())
    print(f"  {len(gdf):,} parcels scored, {n_dropped:,} dropped ({detail})")

    keep = [
        "StKeyFull", "Address", "ClassCode", "TownshipName",
        "land_av", "imp_av", "total_av",
        "Year_Built", "SalesAmount", "SalesDate", "FinishLivingArea",
        "area_sqft", "ILR", "underbuilt", "land_value_per_sqft", "upzoning_score", "tier",
        "geometry",
    ]
    out = gdf[[c for c in keep if c in gdf.columns]]

    print(f"Writing {OUT_GEOJSON}...")
    out.to_file(OUT_GEOJSON, driver="GeoJSON")

    print(f"Writing {OUT_CSV}...")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    print("\nTop 20 parcels by upzoning opportunity:")
    top = (
        out.drop(columns="geometry")
        .nlargest(20, "upzoning_score")
        [["Address", "ClassCode", "land_av", "imp_av", "ILR", "land_value_per_sqft", "upzoning_score", "tier"]]
    )
    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(top.to_string(index=False))

    print("\nScore distribution by tier:")
    summary = (
        out.drop(columns="geometry")
        .groupby("tier")["upzoning_score"]
        .agg(["min", "median", "max", "count"])
        .rename(columns={"min": "score_min", "median": "score_median", "max": "score_max", "count": "n_parcels"})
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
