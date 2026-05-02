"""
Groups contiguous parcels owned by the same entity into development sites,
then scores each site using the same ILR-based formula as analyze.py.

Pipeline: analyze.py → merge_ownership.py → tile.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

SCORES_PATH  = Path("data/upzoning_scores.geojson")
OWNERS_PATH  = Path("data/owners.csv")
OUT_GEOJSON  = Path("data/upzoning_sites.geojson")
OUT_CSV      = Path("data/upzoning_sites.csv")

PARCELS_API = (
    "https://maps.tippecanoe.in.gov/server/rest/services/"
    "Parcels_FeatureService/FeatureServer/0/query"
)
CRS_PROJECTED    = "EPSG:2966"   # Indiana State Plane East (feet)
ADJACENCY_BUFFER = 1.0           # feet — catches shared-boundary parcels


# ---------------------------------------------------------------------------
# Union-Find (no extra dependencies)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank   = {x: 0  for x in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self):
        buckets = defaultdict(list)
        for item in self.parent:
            buckets[self.find(item)].append(item)
        return list(buckets.values())


# ---------------------------------------------------------------------------
# Owner data
# ---------------------------------------------------------------------------

def fetch_owners() -> pd.DataFrame:
    if OWNERS_PATH.exists():
        print(f"Using cached {OWNERS_PATH}")
        return pd.read_csv(OWNERS_PATH, dtype=str)

    print("Fetching owner names from Tippecanoe ArcGIS API (paginated)...")
    records, offset, page = [], 0, 2000

    while True:
        r = requests.get(PARCELS_API, params={
            "where": "1=1",
            "outFields": "StKeyFull,mdeededOwner",
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": page,
            "f": "json",
        }, timeout=60)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        if not features:
            break
        for f in features:
            a = f["attributes"]
            records.append({"StKeyFull": a.get("StKeyFull"), "mdeededOwner": a.get("mdeededOwner")})
        offset += len(features)
        print(f"  {offset:,} records...", end="\r")
        if not data.get("exceededTransferLimit", False):
            break

    print(f"\n  {len(records):,} owner records fetched")
    df = pd.DataFrame(records).drop_duplicates("StKeyFull")
    df.to_csv(OWNERS_PATH, index=False)
    return df


def normalize_owner(name) -> str | None:
    """Uppercase, strip punctuation, collapse whitespace."""
    if not name or not isinstance(name, str):
        return None
    name = name.upper().strip()
    if name in ("", "NAN", "NONE", "NULL"):
        return None
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+",    " ", name).strip()
    return name or None


# ---------------------------------------------------------------------------
# Adjacency and site building
# ---------------------------------------------------------------------------

def connected_components(indices: list, proj_geoms: list) -> list[list]:
    """
    Return connected components of touching/overlapping parcels using
    a 1-ft buffer and an STRtree for efficiency.
    """
    if len(indices) == 1:
        return [indices]

    buffered = [g.buffer(ADJACENCY_BUFFER) for g in proj_geoms]
    tree = STRtree(buffered)
    uf = UnionFind(range(len(indices)))

    for i, geom_i in enumerate(buffered):
        for j in tree.query(geom_i):
            if j > i and buffered[j].intersects(geom_i):
                uf.union(i, j)

    return [[indices[k] for k in group] for group in uf.groups()]


def make_site(cluster: gpd.GeoDataFrame) -> dict:
    land_av = float(cluster["land_av"].sum())
    imp_av  = float(cluster["imp_av"].sum())

    merged_geom = unary_union([make_valid(g) for g in cluster.geometry.values])
    area_sqft = float(
        gpd.GeoSeries([merged_geom], crs=cluster.crs)
        .to_crs(CRS_PROJECTED)
        .area.iloc[0]
    )

    ILR   = (imp_av  / land_av)  if land_av   > 0 else None
    lvpsf = (land_av / area_sqft) if area_sqft > 0 else None
    score = (lvpsf / (ILR + 1.0)) if (ILR is not None and lvpsf is not None) else None

    addrs = [a for a in cluster["Address"].dropna().unique() if a]
    modes = cluster["ClassCode"].dropna()

    return {
        "owner":               cluster["owner_norm"].iloc[0],
        "n_parcels":           len(cluster),
        "addresses":           "; ".join(addrs[:3]) + (" ..." if len(addrs) > 3 else ""),
        "ClassCode":           modes.mode().iloc[0] if len(modes) > 0 else None,
        "land_av":             round(land_av),
        "imp_av":              round(imp_av),
        "area_sqft":           round(area_sqft, 1),
        "ILR":                 round(ILR,   4) if ILR   is not None else None,
        "land_value_per_sqft": round(lvpsf, 4) if lvpsf is not None else None,
        "upzoning_score":      round(score, 4) if score is not None else None,
        "geometry":            merged_geom,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not SCORES_PATH.exists():
        sys.exit(f"ERROR: {SCORES_PATH} not found — run analyze.py first")

    print("Loading scored parcels...")
    gdf = gpd.read_file(SCORES_PATH)
    print(f"  {len(gdf):,} parcels")

    owners_df = fetch_owners()
    owners_df["owner_norm"] = owners_df["mdeededOwner"].apply(normalize_owner)

    gdf = gdf.merge(owners_df[["StKeyFull", "owner_norm"]], on="StKeyFull", how="left")
    # Parcels with no matched owner become isolated single-parcel sites
    gdf["owner_norm"] = gdf["owner_norm"].where(
        gdf["owner_norm"].notna(), "UNKNOWN__" + gdf.index.astype(str)
    )

    print("Projecting geometries for adjacency checks...")
    gdf_proj = gdf.to_crs(CRS_PROJECTED)

    print("Building development sites from ownership groups...")
    sites = []
    groups = list(gdf.groupby("owner_norm"))
    n = len(groups)

    for i, (owner, group) in enumerate(groups):
        if i % 5000 == 0:
            print(f"  {i:,}/{n:,} owners processed...", end="\r")

        idx = list(group.index)

        if len(idx) == 1:
            sites.append(make_site(group))
            continue

        proj_geoms = [gdf_proj.loc[j, "geometry"] for j in idx]
        components = connected_components(idx, proj_geoms)

        for component in components:
            sites.append(make_site(group.loc[component]))

    print(f"\n  {n:,} ownership groups -> {len(sites):,} development sites")

    sites_gdf = gpd.GeoDataFrame(sites, crs=gdf.crs)
    sites_gdf = sites_gdf[
        sites_gdf["upzoning_score"].notna() &
        (sites_gdf["area_sqft"] >= 500)
    ].copy()

    sites_gdf["tier"] = (
        pd.qcut(sites_gdf["upzoning_score"], q=10, labels=False, duplicates="drop") + 1
    )

    print(f"  {len(sites_gdf):,} scoreable sites after filtering")

    print(f"Writing {OUT_GEOJSON}...")
    sites_gdf.to_file(OUT_GEOJSON, driver="GeoJSON")
    print(f"Writing {OUT_CSV}...")
    sites_gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    print("\nTop 20 sites by upzoning opportunity:")
    top = (
        sites_gdf.drop(columns="geometry")
        .nlargest(20, "upzoning_score")
        [["addresses", "owner", "n_parcels", "land_av", "imp_av",
          "ILR", "land_value_per_sqft", "upzoning_score"]]
    )
    pd.set_option("display.max_colwidth", 35)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(top.to_string(index=False))

    print("\nScore distribution by tier:")
    print(
        sites_gdf.drop(columns="geometry")
        .groupby("tier")["upzoning_score"]
        .agg(["min", "median", "max", "count"])
        .to_string()
    )


if __name__ == "__main__":
    main()
