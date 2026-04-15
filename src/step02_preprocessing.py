#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Climate-Induced Migration Pressure Modeling — India (District-Level)
Step 2: Preprocessing — all four datasets, no merging
"""

import re
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────
CENSUS_PATH = "C:/Users/asus/OneDrive/Documents/climate-migration-prediction/data/raw/census.xlsx"
RAINFALL_PATH = "C:/Users/asus/OneDrive/Documents/climate-migration-prediction/data/raw/rainfall.csv"
MPI_PATH = "C:/Users/asus/OneDrive/Documents/climate-migration-prediction/data/raw/mpi.xlsx"
AGRI_PATH = "C:/Users/asus/OneDrive/Documents/climate-migration-prediction/data/raw/agriculture.csv"

AGRI_YEAR_START = 2000
AGRI_YEAR_END   = 2011


# In[3]:


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names, replace spaces/dashes with underscores."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-—]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def normalize_name(text: str) -> str:
    """
    Normalize a state/district name for join-key matching:
      - lowercase
      - strip leading/trailing whitespace
      - replace '&' with 'and'
      - remove the word 'district'
      - remove all punctuation
      - collapse multiple spaces to one
    """
def normalize_name(text):
    if not isinstance(text, str):
        return np.nan

    text = text.lower().strip()
    text = text.replace("&", "and")

    # handle Indian naming variations
    text = text.replace("twenty four", "24")

    text = re.sub(r"\bdistrict\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else np.nan


def apply_name_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized join-key columns alongside the originals."""
    df["state_key"]    = df["state"].apply(normalize_name)
    df["district_key"] = df["district"].apply(normalize_name)
    return df

def safe_ratio(num, den):
    return num.where(den > 0).div(den.where(den > 0))



# In[10]:


# ─────────────────────────────────────────────
# 1. CENSUS
# ─────────────────────────────────────────────

def preprocess_census(path):
    df = pd.read_excel(path, dtype=str)
    df = standardize_columns(df)

    # Convert numeric columns
    numeric_cols = ["tot_p", "p_lit", "p_sc", "p_st", "mainwork_p", "margwork_p"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ─────────────────────────────
    # STEP 1: Extract STATE names FIRST
    # ─────────────────────────────
    state_lookup = (
        df[df["level"] == "STATE"][["state", "name"]]
        .rename(columns={"name": "state_name"})
    )

    # ─────────────────────────────
    # STEP 2: Filter DISTRICT rows
    # ─────────────────────────────
    df = df[(df["level"] == "DISTRICT") & (df["tru"] == "Total")].copy()

    # ─────────────────────────────
    # STEP 3: Map state codes → names
    # ─────────────────────────────
    df = df.merge(state_lookup, on="state", how="left")

    df["state"] = df["state_name"]   # replace code with name
    df["district"] = df["name"]      # district name

    # ─────────────────────────────
    # STEP 4: Compute features
    # ─────────────────────────────
    df["literacy_rate"] = safe_ratio(df["p_lit"], df["tot_p"])
    df["worker_ratio"] = safe_ratio(df["mainwork_p"] + df["margwork_p"], df["tot_p"])
    df["marginal_worker_rate"] = safe_ratio(df["margwork_p"], df["tot_p"])
    df["sc_share"] = safe_ratio(df["p_sc"], df["tot_p"])
    df["st_share"] = safe_ratio(df["p_st"], df["tot_p"])

    # ─────────────────────────────
    # STEP 5: Final columns
    # ─────────────────────────────
    df = df[[
        "state", "district", "tot_p",
        "literacy_rate", "worker_ratio",
        "marginal_worker_rate", "sc_share", "st_share"
    ]]

    # Clean strings
    df["state"] = df["state"].str.strip()
    df["district"] = df["district"].str.strip()

    df = apply_name_normalization(df)

    # 🔥 FIX: ensure one row per district
    df = df.sort_values(by="tot_p", ascending=False)
    df = df.drop_duplicates(subset=["state_key", "district_key"])

    return df


# In[5]:


# ─────────────────────────────────────────────
# 2. RAINFALL
# ─────────────────────────────────────────────

def drought_freq(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (series < (mean - std)).mean()


def flood_freq(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (series > (mean + std)).mean()


def preprocess_rainfall(path):
    df = pd.read_csv(path)
    df = standardize_columns(df)

    df = df[(df["year"] >= AGRI_YEAR_START) & (df["year"] <= AGRI_YEAR_END)]

    agg = df.groupby(["state", "district"]).agg(
        rainfall_mean=("final_annual", "mean"),
        rainfall_std=("final_annual", "std"),
        monsoon_mean=("final_jjas", "mean"),
        drought_freq=("final_annual", drought_freq),
        flood_freq=("final_annual", flood_freq)
    ).reset_index()

    agg["rainfall_cv"] = safe_ratio(agg["rainfall_std"], agg["rainfall_mean"])

    return apply_name_normalization(agg)


# In[6]:


# ─────────────────────────────────────────────
# 3. AGRICULTURE
# ─────────────────────────────────────────────

def preprocess_agriculture(path):
    df = pd.read_csv(path)
    df = standardize_columns(df)

    df = df.dropna(subset=["area"]).copy()
    df["year_int"] = df["year"].str.extract(r"(\d{4})")[0].astype(float)

    def weighted_yield(g):
        return (g["yield"] * g["area"]).sum() / g["area"].sum()

    result = df.groupby(["state", "district"]).apply(lambda g: pd.Series({
        "avg_yield": weighted_yield(g),
        "yield_cv": g.groupby("year_int")["yield"].mean().std(),
        "area_harvested": g.groupby("year_int")["area"].sum().mean(),
        "production_mean": g.groupby("year_int")["production"].sum().mean(),
        "production_std": g.groupby("year_int")["production"].sum().std(),
        "crop_diversity": g["crop"].nunique()
    })).reset_index()

    return apply_name_normalization(result)



# In[7]:


def preprocess_mpi(path):
    import pandas as pd

    # ─────────────────────────────
    # Load raw Excel (no header)
    # ─────────────────────────────
    raw = pd.read_excel(path, header=None)

    # Build header from row 4 & 5
    header1 = raw.iloc[4].ffill()
    header2 = raw.iloc[5]

    cols = []
    for h1, h2 in zip(header1, header2):
        h1 = str(h1).strip()
        h2 = str(h2).strip()

        if h2 and h2 != "nan":
            cols.append(f"{h1} {h2}")
        else:
            cols.append(h1)

    # Extract actual data
    df = raw.iloc[9:].reset_index(drop=True)
    df.columns = cols

    # Standardize column names
    df = standardize_columns(df)

    # 🔥 FIX 1: remove duplicate columns from Excel
    df = df.loc[:, ~df.columns.duplicated()]

    # Debug (optional)
    print("\n[MPI Columns]:")
    print(df.columns.tolist())

    # ─────────────────────────────
    # Detect state & district columns
    # ─────────────────────────────
    state_col = None
    district_col = None

    for col in df.columns:
        if col == "state":
            state_col = col
        elif col == "district":
            district_col = col

    if state_col is None or district_col is None:
        raise ValueError("Could not detect state/district columns in MPI")

    df = df.rename(columns={
        state_col: "state",
        district_col: "district"
    })

    # Drop invalid rows
    df = df.dropna(subset=["state", "district"])

    # ─────────────────────────────
    # Detect MPI columns dynamically
    # ─────────────────────────────
    for col in df.columns:
        if "mpi" in col and "district" in col:
            df.rename(columns={col: "mpi"}, inplace=True)
        elif "headcount" in col:
            df.rename(columns={col: "headcount_ratio"}, inplace=True)
        elif "intensity" in col:
            df.rename(columns={col: "intensity"}, inplace=True)

    # 🔥 FIX 2: remove duplicates AGAIN after renaming
    df = df.loc[:, ~df.columns.duplicated()]

    # ─────────────────────────────
    # Convert to numeric
    # ─────────────────────────────
    for col in ["mpi", "headcount_ratio", "intensity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only required columns
    df = df[["state", "district", "mpi", "headcount_ratio", "intensity"]]

    # 🔥 FIX 3: reset index to avoid alignment errors
    df = df.reset_index(drop=True)

    # Apply name normalization
    df = apply_name_normalization(df)

    print(f"[MPI] rows: {len(df)} | cols: {df.shape[1]}")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    return df


# In[8]:


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("Preprocessing...")

    df_census = preprocess_census(CENSUS_PATH)
    df_rainfall = preprocess_rainfall(RAINFALL_PATH)
    df_agri = preprocess_agriculture(AGRI_PATH)
    df_mpi = preprocess_mpi(MPI_PATH)

    print("\nShapes:")
    print("Census:", df_census.shape)
    print("Rainfall:", df_rainfall.shape)
    print("Agriculture:", df_agri.shape)
    print("MPI:", df_mpi.shape)

    return df_census, df_rainfall, df_agri, df_mpi


if __name__ == "__main__":
    main()


# In[ ]:




