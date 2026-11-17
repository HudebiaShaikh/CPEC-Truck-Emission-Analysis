"""
Step 1 - Load and clean the raw fleet data (Table S2 of the paper).

Data quality issues found in the raw CSV:
  * Avg_Speed_kmh is text for some trucks, e.g. "20~40" or "25~60" (a range).
  * Engine_Power_hp is missing for 4 trucks (no. 42, 50, 55, 56) (shown as "-" in the paper).
  * Emission_Class uses "-" for trucks older than the Pak-II standard.
"""
import numpy as np
import pandas as pd

from .config import RAW_DATA, SITES

CURRENT_YEAR = 2021  # trucks were tested around 2020-2021; age is measured relative to this


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def _speed_to_number(value):
    """'20~40' -> 30.0 (mid-point of the range); '40' -> 40.0"""
    text = str(value).strip()
    if "~" in text:
        low, high = text.split("~")
        return (float(low) + float(high)) / 2
    return float(text)


def truck_type(axles: int) -> str:
    if axles <= 2:
        return "2-axle (light/medium)"
    if axles <= 4:
        return "3-4 axle (heavy rigid)"
    return "5-6 axle (articulated)"


def data_quality_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Small table used on the 'Dataset' page."""
    speed_ranges = raw["Avg_Speed_kmh"].astype(str).str.contains("~").sum()
    rows = []
    for col in raw.columns:
        rows.append({
            "Column": col,
            "Type (raw)": str(raw[col].dtype),
            "Missing values": int(raw[col].isna().sum()),
            "Unique values": int(raw[col].nunique()),
        })
    rep = pd.DataFrame(rows)
    rep.loc[rep["Column"] == "Avg_Speed_kmh", "Note"] = f"{speed_ranges} values are ranges like '25~60'"
    rep.loc[rep["Column"] == "Emission_Class", "Note"] = "'-' means older than Pak-II standard"
    rep.loc[rep["Column"] == "Engine_Power_hp", "Note"] = "missing in source paper"
    return rep.fillna("")


def clean(raw: pd.DataFrame, sites=None) -> pd.DataFrame:
    """Return a clean, analysis-ready copy of the fleet table."""
    sites = sites or SITES
    df = raw.copy()

    # 1) speed ranges -> mid-point, keep a flag so we know which ones were ranges
    df["Speed_was_range"] = df["Avg_Speed_kmh"].astype(str).str.contains("~")
    df["Avg_Speed_kmh"] = df["Avg_Speed_kmh"].apply(_speed_to_number)

    # 2) emission class: "-" -> "Pre-Pak-II"
    df["Emission_Class"] = df["Emission_Class"].fillna("-").replace("-", "Pre-Pak-II")

    # 3) missing engine power -> median power of trucks with the same number of axles
    #    (axles is the strongest physical indicator of engine size in this fleet)
    df["Power_was_imputed"] = df["Engine_Power_hp"].isna()
    df["Engine_Power_hp"] = df.groupby("Axles")["Engine_Power_hp"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Engine_Power_hp"] = df["Engine_Power_hp"].fillna(df["Engine_Power_hp"].median())

    # 4) engineered features
    df["Vehicle_Age"] = CURRENT_YEAR - df["Model_Year"]
    df["Truck_Type"] = df["Axles"].apply(truck_type)
    # each truck gets the weather of the nearest test site (works for uploaded data too)
    nearest = df["Altitude_m"].map(lambda a: min(sites, key=lambda s: abs(s - a)))
    df["Site"] = nearest.map(lambda s: sites[s]["name"])
    df["Temperature_C"] = nearest.map(lambda s: sites[s]["T"])
    df["Humidity_pct"] = nearest.map(lambda s: sites[s]["RH"])

    df["Truck_No"] = df["Truck_No"].astype(int)
    return df
