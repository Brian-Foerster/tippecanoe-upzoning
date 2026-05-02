import pandas as pd
import numpy as np
from scipy.stats import spearmanr

df = pd.read_csv("data/upzoning_sites.csv")
df = df[df["upzoning_score"].notna() & (df["ILR"] > 0)].copy()

print(f"Sites analysed: {len(df):,}\n")

# --- 1. Distribution of the two raw signals ---
print("=== land_value_per_sqft ($/sqft) ===")
print(df["land_value_per_sqft"].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2))

print("\n=== ILR (imp_av / land_av) ===")
print(df["ILR"].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2))

# --- 2. Variance decomposition ---
eps = 1.0
df["log_lvpsf"] = np.log1p(df["land_value_per_sqft"])
df["log_ilr"]   = np.log(df["ILR"] + eps)
v_l = df["log_lvpsf"].var()
v_i = df["log_ilr"].var()
cov = df[["log_lvpsf", "log_ilr"]].cov().iloc[0, 1]
v_s = v_l + v_i - 2 * cov

print("\n=== Log-space variance decomposition (eps=1.0) ===")
print(f"  Var(log lvpsf):    {v_l:.3f}  ({v_l/v_s:.1%} of score variance)")
print(f"  Var(log ILR+eps):  {v_i:.3f}  ({v_i/v_s:.1%} of score variance)")
print(f"  Corr(lvpsf, ILR):  {df['land_value_per_sqft'].corr(df['ILR']):.3f}")

# --- 3. Epsilon sensitivity ---
print("\n=== Rank correlation vs eps=1.0 (full dataset) ===")
base_rank = (df["land_value_per_sqft"] / (df["ILR"] + 1.0)).rank(ascending=False)
for e in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    alt_rank = (df["land_value_per_sqft"] / (df["ILR"] + e)).rank(ascending=False)
    rho, _ = spearmanr(base_rank, alt_rank)
    print(f"  eps={e:5.1f}:  rho={rho:.4f}")

# --- 4. ILR thresholds — how many sites are actually underbuilt? ---
print("\n=== Share of sites by ILR band ===")
bands = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 999)]
for lo, hi in bands:
    n = ((df["ILR"] >= lo) & (df["ILR"] < hi)).sum()
    print(f"  ILR [{lo:4.1f}, {hi:5.1f}):  {n:6,}  ({n/len(df):.1%})")

# --- 5. Top vs bottom decile breakdown ---
top = df.nlargest(len(df) // 10, "upzoning_score")
bot = df.nsmallest(len(df) // 10, "upzoning_score")

print(f"\n=== Top decile ({len(top):,} sites) ===")
print(f"  median lvpsf:   {top['land_value_per_sqft'].median():.2f}")
print(f"  median ILR:     {top['ILR'].median():.2f}")
print(f"  median score:   {top['upzoning_score'].median():.2f}")

print(f"\n=== Bottom decile ({len(bot):,} sites) ===")
print(f"  median lvpsf:   {bot['land_value_per_sqft'].median():.2f}")
print(f"  median ILR:     {bot['ILR'].median():.2f}")
print(f"  median score:   {bot['upzoning_score'].median():.2f}")

# --- 6. What's actually driving top rankings? ---
# Is it mostly lvpsf variance or ILR variance within top decile?
print("\n=== Within top decile: what separates top from bottom half? ===")
top_hi = top.nlargest(len(top) // 2, "upzoning_score")
top_lo = top.nsmallest(len(top) // 2, "upzoning_score")
print(f"  Top half of top decile:  median lvpsf={top_hi['land_value_per_sqft'].median():.2f}  median ILR={top_hi['ILR'].median():.2f}")
print(f"  Bot half of top decile:  median lvpsf={top_lo['land_value_per_sqft'].median():.2f}  median ILR={top_lo['ILR'].median():.2f}")

# --- 7. Alternative: what if we treat both signals symmetrically? ---
# Normalise each to [0,1] then combine
df["rank_lvpsf"] = df["land_value_per_sqft"].rank(pct=True)
df["rank_ilr"]   = (1 / (df["ILR"] + 1.0)).rank(pct=True)

for w in [0.5, 0.7, 1.0]:
    w2 = 1 - w
    df["alt_score"] = w * df["rank_lvpsf"] + w2 * df["rank_ilr"]
    rho, _ = spearmanr(df["upzoning_score"].rank(), df["alt_score"].rank())
    print(f"\n  Rank-combined (lvpsf weight={w:.0%}, ILR weight={w2:.0%}): rho vs current = {rho:.4f}")
