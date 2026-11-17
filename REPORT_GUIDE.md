# Report guide (follows the DSA project guidelines)

All numbers below come from `python run_analysis.py` with the default settings
(seed 42, 15 % noise, 20 % test set). If you change anything, re-run and update them.
Screenshots for the report: take them from the app's tabs.

## 1. Introduction & problem formulation  (rubric 1.0)
* CPEC / Karakoram Highway climbs from 540 m (Islamabad) to 4693 m (Khunjerab Pass).
* Problem: no accessible, Pakistan-specific tool for altitude-dependent truck emissions.
* Objectives: estimate CO/HC/NOx/CO₂ for 70 trucks with the published MLR, validate against Table S3, show the effect of altitude, age and truck type, compare ML models, and give an interactive tool.
* ML task: **regression** (predict emission factor, g/km) + **clustering** (K-Means emission profiles).
* Why ML: the published formula needs weather data that fleet operators lack; ML tests whether
  emissions can be predicted from truck and route data alone.
* **Trade-off to mention (required by CEP):** interpretability (Linear Regression) vs accuracy and
  stability (Random Forest); also model complexity vs generalisation on only 70 rows.

## 2. Dataset  (rubric 1.0)
* Source: Table S2, supplement of Shaikh et al. (2023), https://doi.org/10.1016/j.ese.2022.100226
* 70 records, 8 columns. Numerical: Model_Year, Axles, Engine_Power_hp, Avg_Speed_kmh, VKT_km_month,
  Altitude_m. Categorical: Emission_Class (Pak-II / Pre-Pak-II).
* Quality issues: 4 missing engine powers (trucks 42, 50, 55, 56), 6 speeds given as ranges
  ("20~40", "25~60"), "-" in emission class.
* Imbalance: 47 trucks Islamabad, 20 Sost, 3 Khunjerab.
* EDA (Fleet dashboard tab): high-altitude fleet is newer and bigger, so altitude and truck size are confounded.

## 3. Pre-processing & feature engineering  (rubric 1.5)
| Step | Why |
|---|---|
| Speed range → mid-point | needs a number; mid-point is the unbiased value of a range |
| Missing power → median of same-axle trucks | axles predict engine size; median resists outliers |
| "-" → "Pre-Pak-II" | readable category |
| One-hot encode Emission_Class | models need numbers; no false ordering |
| StandardScaler (linear models only) | coefficients become comparable; trees don't need scaling |
| Drop T and RH from ML features | constant per site, so they duplicate Altitude_m (multicollinearity) |
| Stratified 80/20 split by site + 5-fold CV | small data; every split keeps low and high altitude |
| Imputer/scaler inside a sklearn Pipeline | fitted on training folds only → **no data leakage** |

## 4. Emission estimation (published MLR, Table S4)
EF = b₁·RH + b₂·T + b₃·Power + b₄·ModelYear + b₅·Axles + b₀  (g/km)

Mean emission factor by site (g/km):

| Site | CO | HC | NOx | CO₂ |
|---|---|---|---|---|
| Islamabad (540 m) | 12.93 | 14.21 | 10.55 | 458.4 |
| Sost (2800 m) | 15.56 | 17.19 | 12.80 | 635.0 |
| Khunjerab (4693 m) | 22.61 | 24.44 | 18.89 | 944.9 |
| Increase Isb → Khunjerab | +75 % | +72 % | +79 % | +106 % |
| Paper's increase | +104 % | +86 % | +125 % | +134 % |

Check vs paper (Islamabad, Pre-Pak-II): ours CO 12.80 / HC 14.05 / NOx 10.38 vs paper 12.94 / 15.21 / 10.69.
CO₂ rises with axles: 435 (2-axle) → 799 g/km (6-axle). Whole fleet ≈ 352 t CO₂-eq per month (100-yr GWP).

## 5. Validation vs Table S3 (3 measured trucks)
| Pollutant | MAPE |
|---|---|
| CO | 16.0 % |
| HC | 14.2 % |
| NOx | 23.3 % |
| CO₂ | 8.1 % |
Islamabad and Khunjerab trucks are within about ±10 %; the Sost truck is over-estimated (CO +39 %,
NOx +55 %) because the true altitude effect is non-linear (emissions dip at Sost, per the paper).

## 6. ML models & evaluation  (rubric 2.0 + 1.5)
Models: Linear Regression, Ridge (L2-regularised linear), Random Forest (bagged trees),
Gradient Boosting (sequential boosted trees). Metrics: MAE, MSE, RMSE, R², 5-fold CV R².
Target CO₂ (other pollutants in `outputs/model_comparison_*.csv`):

| Model | Train R² | Test R² | CV R² (mean ± std) | Test RMSE (g/km) |
|---|---|---|---|---|
| Linear Regression | 0.891 | 0.650 | 0.790 ± 0.067 | 77.8 |
| Ridge | 0.890 | 0.663 | 0.808 ± 0.057 | 76.4 |
| Random Forest | 0.914 | 0.840 | 0.770 ± 0.077 | 52.5 |
| Gradient Boosting | 0.955 | 0.764 | 0.729 ± 0.102 | 63.9 |

Points to discuss:
* Random Forest wins on the single test set; Ridge wins on cross-validation. With only 14 test trucks
  the test score depends heavily on which trucks land in it, so CV is the fairer comparison.
* Gradient Boosting has the highest train R² and the largest gap → **overfitting**.
* GridSearchCV for Random Forest: best max_depth=5, min_samples_leaf=2, n_estimators=300 (CV R² 0.771).
  Tuning helped very little, which suggests the limit is data size and noise, not hyper-parameters.
* The target is semi-synthetic (formula × noise). Say this clearly. Raising noise lowers every R²
  (irreducible error).
* Poor cases: residual plot in the ML tab lists the worst trucks (e.g. truck 61, 5-axle at Sost).

## 7. Clustering
K-Means on standardised year, axles, power, altitude, CO, NOx, CO₂. Best k = 2 (silhouette 0.57):
old small low-altitude trucks (≈1994, 2.7 axles, 466 g CO₂/km) vs new large high-altitude trucks
(≈2013, 5.7 axles, 784 g CO₂/km).

## 8. Limitations & improvements  (rubric 1.0)
Limitations: only 70 trucks (3 at Khunjerab); site weather assumed; ML target semi-synthetic;
the MLR is linear while the real altitude effect is not (dip at Sost); speed ranges and missing power imputed.
Improvements: get per-truck measurements from the authors; use real weather (PMD / ERA5) for each test day;
add road gradient, payload and fuel quality; collect more high-altitude trucks.

## References
1. Shaikh, A. A., He, T., Deng, F., Luo, Z., Zhao, J., Zhang, Z., & Liu, H. (2023). Altitude-dependent gaseous
   emissions from freight trucks along the China-Pakistan Economic Corridor in Pakistan. Environmental Science
   and Ecotechnology, 14, 100226. https://doi.org/10.1016/j.ese.2022.100226 (with Supplement, Tables S2–S5)
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830.
3. pandas, Plotly and Streamlit documentation.

## 9. AI-tools declaration (required)
State which parts you built with AI help (e.g. code scaffolding, debugging) and that you re-ran and checked
every result yourself.
