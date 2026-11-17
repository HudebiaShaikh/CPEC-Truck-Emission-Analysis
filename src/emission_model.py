"""
Step 2 - Apply the published MLR emission formula (Table S4) to every truck,
then convert the emission factors into monthly emissions and climate impact.
"""
import numpy as np
import pandas as pd

from .config import MLR_COEFFS, POLLUTANTS, GWP_100, VALIDATION_TRUCKS


def emission_factor(pollutant, RH, T, RP, MY, A):
    """EF (g/km) from the paper's regression. Works with numbers or pandas columns."""
    c = MLR_COEFFS[pollutant]
    ef = c["RH"] * RH + c["T"] * T + c["RP"] * RP + c["MY"] * MY + c["A"] * A + c["intercept"]
    return np.maximum(ef, 0)  # an emission can never be negative


def add_emissions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for p in POLLUTANTS:
        out[f"EF_{p}_g_km"] = emission_factor(
            p, out["Humidity_pct"], out["Temperature_C"], out["Engine_Power_hp"],
            out["Model_Year"], out["Axles"],
        )
        # monthly mass = EF (g/km) * distance (km/month) -> kg/month
        out[f"{p}_kg_month"] = out[f"EF_{p}_g_km"] * out["VKT_km_month"] / 1000

    # climate impact in CO2-equivalent using 100-year GWP (HC has no GWP in Table S5)
    out["CO2eq_kg_month"] = sum(out[f"{p}_kg_month"] * g for p, g in GWP_100.items())
    return out


def validate(sites) -> pd.DataFrame:
    """Compare formula estimates against the 3 trucks actually measured (Table S3)."""
    name_to_met = {v["name"]: v for v in sites.values()}
    rows = []
    for t in VALIDATION_TRUCKS:
        met = name_to_met[t["Site"]]
        for p in POLLUTANTS:
            est = float(emission_factor(p, met["RH"], met["T"], t["Engine_Power_hp"], t["Model_Year"], t["Axles"]))
            rows.append({
                "Truck_No": t["Truck_No"], "Site": t["Site"], "Pollutant": p,
                "Measured_g_km": t[p], "Estimated_g_km": round(est, 2),
                "Error_g_km": round(est - t[p], 2),
                "Error_pct": round((est - t[p]) / t[p] * 100, 1),
            })
    return pd.DataFrame(rows)


def validation_summary(val: pd.DataFrame) -> pd.DataFrame:
    """MAPE per pollutant (mean absolute percentage error)."""
    return (val.assign(Abs_pct=val["Error_pct"].abs())
               .groupby("Pollutant", sort=False)["Abs_pct"].mean().round(1)
               .rename("MAPE_%").reset_index())


def site_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"EF_{p}_g_km" for p in POLLUTANTS]
    return df.groupby("Site", sort=False)[cols].mean().round(2)


def route_budget(df: pd.DataFrame, route_km: dict) -> pd.DataFrame:
    """Emission (g) for ONE average truck trip over each altitude band of the KKH."""
    rows = []
    for alt, km in route_km.items():
        sub = df[df["Altitude_m"] == alt]
        if sub.empty:
            continue
        row = {"Altitude_m": alt, "Site": sub["Site"].iloc[0], "Segment_km": km}
        for p in POLLUTANTS:
            row[f"{p}_kg_per_trip"] = round(sub[f"EF_{p}_g_km"].mean() * km / 1000, 2)
        rows.append(row)
    return pd.DataFrame(rows)
