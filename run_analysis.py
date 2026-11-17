"""
Command-line version of the whole analysis (no dashboard).
Run:  python run_analysis.py
It prints every result and saves tables to the outputs/ folder -> use them in your report.
"""
import pandas as pd

from src.config import OUTPUT_DIR, POLLUTANTS, ROUTE_KM, SITES
from src.data_processing import clean, data_quality_report, load_raw
from src.emission_model import add_emissions, route_budget, site_summary, validate, validation_summary
from src.ml_models import cluster_trucks, make_target, train_and_compare, tune_random_forest

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    raw = load_raw()
    print("\n=== 1. RAW DATA ===")
    print(raw.shape, "\n", data_quality_report(raw))

    df = add_emissions(clean(raw, SITES))
    df.to_csv(OUTPUT_DIR / "trucks_with_emissions.csv", index=False)
    print("\n=== 2. MEAN EMISSION FACTOR BY SITE (g/km) ===")
    print(site_summary(df))
    print("\nBy emission class:\n", df.groupby("Emission_Class")[[f"EF_{p}_g_km" for p in POLLUTANTS]].mean().round(2))
    print("\nOne-trip route budget (kg):\n", route_budget(df, ROUTE_KM))

    val = validate(SITES)
    val.to_csv(OUTPUT_DIR / "validation_table_S3.csv", index=False)
    print("\n=== 3. VALIDATION vs TABLE S3 ===")
    print(val)
    print(validation_summary(val))

    print("\n=== 4. MACHINE LEARNING (15% measurement noise) ===")
    for p in POLLUTANTS:
        y = make_target(df, p)
        res, fitted, split = train_and_compare(df, y)
        res.to_csv(OUTPUT_DIR / f"model_comparison_{p}.csv")
        print(f"\n--- Target: {p} ---\n{res}")
    y = make_target(df, "CO2")
    _, _, split = train_and_compare(df, y)
    gs, table = tune_random_forest(split["X_train"], split["y_train"])
    print("\nBest Random Forest params:", gs.best_params_, "CV R2 =", round(gs.best_score_, 3))

    print("\n=== 5. CLUSTERING ===")
    labels, sil, k, _ = cluster_trucks(df)
    print(sil.round(3)); print("Chosen k =", k)
    df["Cluster"] = labels
    print(df.groupby("Cluster")[["Model_Year", "Axles", "Altitude_m", "EF_CO_g_km", "EF_CO2_g_km"]].mean().round(1))
    print(f"\nSaved tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
