"""
Step 3 - Machine Learning.

Why ML when we already have a formula?
  The paper's formula needs temperature and humidity, which a fleet manager usually
  does not have. We ask: "Can an ML model predict a truck's emission factor from
  information you can read off the truck and the route (year, axles, power, speed,
  altitude, emission class)?" This is a REGRESSION problem (target is a number in g/km).

How the target is built (be honest about this in the report!):
  Per-truck measured values are NOT published (only 3 trucks in Table S3). So the
  target = formula estimate x (1 + random measurement noise). The noise level
  (default 15%) represents real on-road measurement scatter (the paper reports
  95% CIs of roughly +/-10-25%). This makes the target "semi-synthetic".
  Consequence: the true relationship is linear, so Linear Regression is EXPECTED to
  do well. Tree models must learn it from only ~56 training rows.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE

NUMERIC_FEATURES = ["Model_Year", "Axles", "Engine_Power_hp", "Avg_Speed_kmh", "VKT_km_month", "Altitude_m"]
CATEGORICAL_FEATURES = ["Emission_Class"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_target(df: pd.DataFrame, pollutant: str, noise_pct: float = 15.0, seed: int = RANDOM_STATE) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = df[f"EF_{pollutant}_g_km"].to_numpy()
    noisy = base * (1 + rng.normal(0, noise_pct / 100, size=len(base)))
    return pd.Series(np.maximum(noisy, 0), index=df.index, name=f"Observed_{pollutant}_g_km")


def _preprocessor(scale: bool) -> ColumnTransformer:
    # Imputer + scaler live INSIDE the pipeline, so they are fitted on training folds
    # only -> no data leakage from the test set.
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES),
    ])


def build_models(rf_n_estimators=300, rf_max_depth=None, gb_learning_rate=0.05, ridge_alpha=1.0):
    return {
        "Linear Regression": Pipeline([("prep", _preprocessor(True)), ("model", LinearRegression())]),
        "Ridge Regression": Pipeline([("prep", _preprocessor(True)), ("model", Ridge(alpha=ridge_alpha))]),
        "Random Forest": Pipeline([("prep", _preprocessor(False)), ("model", RandomForestRegressor(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth, min_samples_leaf=2, random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([("prep", _preprocessor(False)), ("model", GradientBoostingRegressor(
            n_estimators=300, learning_rate=gb_learning_rate, max_depth=2, subsample=0.8, random_state=RANDOM_STATE))]),
    }


def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y_true, y_pred)}


def train_and_compare(df, y, test_size=0.2, models=None, cv_folds=5):
    """Train every model, return (results table, fitted models, split data)."""
    X = df[FEATURES]
    # stratify on site so every split contains low AND high altitude trucks
    strat = (df["Altitude_m"] > 1000).astype(int)  # low vs high altitude (only 3 Khunjerab trucks, so Sost+Khunjerab = high)
    if strat.value_counts().min() < 2:
        strat = None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strat)

    models = models or build_models()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted, preds = [], {}, {}
    for name, pipe in models.items():
        cv_r2 = cross_val_score(pipe, X_tr, y_tr, cv=kf, scoring="r2")
        pipe.fit(X_tr, y_tr)
        tr, te = _metrics(y_tr, pipe.predict(X_tr)), _metrics(y_te, pipe.predict(X_te))
        rows.append({
            "Model": name,
            "Train R2": tr["R2"], "Test R2": te["R2"],
            "CV R2 (mean)": cv_r2.mean(), "CV R2 (std)": cv_r2.std(),
            "Test MAE": te["MAE"], "Test MSE": te["MSE"], "Test RMSE": te["RMSE"],
            "Overfit gap (Train-Test R2)": tr["R2"] - te["R2"],
        })
        fitted[name] = pipe
        preds[name] = pipe.predict(X_te)
    results = pd.DataFrame(rows).set_index("Model").round(3)
    split = {"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te, "pred": preds}
    return results, fitted, split


def tune_random_forest(X_train, y_train, cv_folds=5):
    """Hyper-parameter tuning with GridSearchCV (training data only)."""
    pipe = build_models()["Random Forest"]
    grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 5, None],
        "model__min_samples_leaf": [1, 2, 4],
    }
    gs = GridSearchCV(pipe, grid, cv=KFold(cv_folds, shuffle=True, random_state=RANDOM_STATE), scoring="r2", n_jobs=-1)
    gs.fit(X_train, y_train)
    table = (pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
               .sort_values("rank_test_score").reset_index(drop=True))
    table["params"] = table["params"].astype(str).str.replace("model__", "")
    return gs, table


def feature_names(pipe) -> list:
    return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]


def feature_importance(pipe) -> pd.Series:
    model = pipe.named_steps["model"]
    names = feature_names(pipe)
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:  # linear model on standardised features -> |coef| is comparable
        vals = np.abs(model.coef_)
    return pd.Series(vals, index=names).sort_values(ascending=True)


def cluster_trucks(df, k=None, cols=None):
    """K-Means clustering of trucks into emission profiles. k picked by silhouette score if None."""
    cols = cols or ["Model_Year", "Axles", "Engine_Power_hp", "Altitude_m", "EF_CO_g_km", "EF_NOx_g_km", "EF_CO2_g_km"]
    Xs = StandardScaler().fit_transform(df[cols])
    scores = {}
    for kk in range(2, 7):
        lab = KMeans(n_clusters=kk, n_init=10, random_state=RANDOM_STATE).fit_predict(Xs)
        scores[kk] = silhouette_score(Xs, lab)
    k = k or max(scores, key=scores.get)
    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(Xs)
    return labels, pd.Series(scores, name="Silhouette"), k, cols
