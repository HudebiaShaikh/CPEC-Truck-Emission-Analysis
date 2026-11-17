"""
CPEC Truck Emission Analyzer - interactive web app
Run in VS Code terminal:   streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import GWP_100, POLLUTANTS, ROUTE_KM, SITES
from src.data_processing import clean, data_quality_report, load_raw
from src.emission_model import add_emissions, emission_factor, route_budget, validate, validation_summary
from src.ml_models import (FEATURES, build_models, cluster_trucks, feature_importance, make_target,
                           train_and_compare, tune_random_forest)

st.set_page_config(page_title="CPEC Truck Emission Analyzer", page_icon="🚛", layout="wide")

SITE_ORDER = ["Islamabad", "Sost", "Khunjerab Pass"]
SITE_COLORS = {"Islamabad": "#2a9d8f", "Sost": "#e9c46a", "Khunjerab Pass": "#e76f51"}
POL_COLORS = {"CO": "#457b9d", "HC": "#e9c46a", "NOx": "#e76f51", "CO2": "#2a9d8f"}
REQUIRED_COLS = ["Truck_No", "Model_Year", "Emission_Class", "Axles", "Engine_Power_hp",
                 "Avg_Speed_kmh", "VKT_km_month", "Altitude_m"]

# ------------------------------------------------------------------ sidebar: data + settings
st.sidebar.header("🚛 CPEC Emission Analyzer")
upload = st.sidebar.file_uploader("Use your own truck CSV (optional)", type="csv",
                                  help="Same columns as data/CPEC_Truck_Emissions_70_Trucks.csv")

with st.sidebar.expander("⚙️ Weather at each site", expanded=False):
    sites = {}
    for alt, s in SITES.items():
        t = st.slider(f"{s['name']} temperature (°C)", -10.0, 40.0, float(s["T"]), 0.5, key=f"T{alt}")
        rh = st.slider(f"{s['name']} humidity (%)", 10.0, 90.0, float(s["RH"]), 1.0, key=f"RH{alt}")
        sites[alt] = {"name": s["name"], "T": t, "RH": rh}
    st.caption("Assumed values (the paper does not publish them). Everything recalculates.")


@st.cache_data
def prepare(raw: pd.DataFrame, sites_key):
    sites_ = {alt: {"name": n, "T": t, "RH": rh} for alt, n, t, rh in sites_key}
    return add_emissions(clean(raw, sites_))


if upload is not None:
    raw = pd.read_csv(upload)
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.sidebar.error(f"Missing columns: {', '.join(missing)}. Using the default dataset.")
        raw = load_raw()
    else:
        st.sidebar.success(f"Loaded {len(raw)} trucks from your file")
else:
    raw = load_raw()

sites_key = tuple((a, v["name"], v["T"], v["RH"]) for a, v in sites.items())
df = prepare(raw, sites_key)
EF = {p: f"EF_{p}_g_km" for p in POLLUTANTS}
site_means = df.groupby("Site")[list(EF.values())].mean()

# ------------------------------------------------------------------ header
st.title("🚛 CPEC Truck Emission Analyzer")
st.caption("Islamabad (540 m) → Sost (2800 m) → Khunjerab Pass (4693 m) · emission model: Shaikh et al. (2023)")

k = st.columns(5)
k[0].metric("Trucks analysed", len(df))
k[1].metric("Avg CO₂", f"{df[EF['CO2']].mean():.0f} g/km")
k[2].metric("Avg NOx", f"{df[EF['NOx']].mean():.1f} g/km")
if {"Islamabad", "Khunjerab Pass"} <= set(site_means.index):
    up = (site_means.loc["Khunjerab Pass", EF["CO2"]] / site_means.loc["Islamabad", EF["CO2"]] - 1) * 100
    k[3].metric("CO₂ rise at Khunjerab", f"{up:+.0f} %")
k[4].metric("Fleet CO₂-eq per month", f"{df['CO2eq_kg_month'].sum() / 1000:,.0f} t")

tabs = st.tabs(["🧮 Emission calculator", "📊 Fleet dashboard", "🛣️ Route comparison",
                "✅ Validation", "🤖 ML prediction", "🧩 Truck clusters", "📁 Data"])


# ================================================================== 1. calculator
with tabs[0]:
    trucks = ["Custom truck"] + [f"Truck {n}" for n in df["Truck_No"]]
    pick = st.selectbox("Start from a truck in the dataset, or enter your own", trucks)
    base = df.iloc[trucks.index(pick) - 1] if pick != "Custom truck" else None
    key = pick.replace(" ", "_")

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Truck & conditions")
        my = st.slider("Model year", 1965, 2025, int(base["Model_Year"]) if base is not None else 2010, key=f"my{key}")
        axles = st.select_slider("Axles", [2, 3, 4, 5, 6], int(base["Axles"]) if base is not None else 3, key=f"ax{key}")
        rp = st.slider("Engine power (hp)", 100, 450, int(base["Engine_Power_hp"]) if base is not None else 220, 5,
                       key=f"rp{key}")
        vkt = st.number_input("Distance per month (km)", 500, 20000,
                              int(base["VKT_km_month"]) if base is not None else 6000, 500, key=f"vkt{key}")
        site_names = [s["name"] for s in sites.values()]
        site = st.radio("Site", site_names, horizontal=True,
                        index=site_names.index(base["Site"]) if base is not None else 0, key=f"site{key}")
        met = next(s for s in sites.values() if s["name"] == site)
        t = st.slider("Temperature (°C)", -15.0, 45.0, float(met["T"]), 0.5, key=f"t{key}{site}")
        rh = st.slider("Humidity (%)", 5.0, 95.0, float(met["RH"]), 1.0, key=f"rh{key}{site}")

    with right:
        res = {p: float(emission_factor(p, rh, t, rp, my, axles)) for p in POLLUTANTS}
        st.subheader("Emissions")
        c = st.columns(4)
        for col, p in zip(c, POLLUTANTS):
            col.metric(f"{p} (g/km)", f"{res[p]:.2f}")
            col.caption(f"{res[p] * vkt / 1000:,.1f} kg / month")
        co2eq = sum(res[p] * vkt / 1000 * g for p, g in GWP_100.items())
        st.metric("Climate impact (100-yr GWP)", f"{co2eq:,.0f} kg CO₂-eq / month")

        # same truck at all three sites
        rows = [{"Site": s["name"], "Pollutant": p, "g/km": float(emission_factor(p, s["RH"], s["T"], rp, my, axles))}
                for s in sites.values() for p in POLLUTANTS]
        comp = pd.DataFrame(rows)
        g1, g2 = st.columns(2)
        g1.plotly_chart(px.bar(comp[comp.Pollutant != "CO2"], x="Pollutant", y="g/km", color="Site", barmode="group",
                               color_discrete_map=SITE_COLORS, title="This truck at each site: CO, HC, NOx"),
                        width="stretch")
        g2.plotly_chart(px.bar(comp[comp.Pollutant == "CO2"], x="Site", y="g/km", color="Site",
                               color_discrete_map=SITE_COLORS, title="This truck at each site: CO₂"),
                        width="stretch")
        if base is not None:
            st.caption(f"Truck {int(base['Truck_No'])} in the dataset: {base['Emission_Class']}, "
                       f"{base['Truck_Type']}, tested at {base['Site']}.")


# ================================================================== 2. fleet dashboard
with tabs[1]:
    f = st.columns(4)
    sel_sites = f[0].multiselect("Site", SITE_ORDER, SITE_ORDER)
    sel_cls = f[1].multiselect("Emission class", sorted(df["Emission_Class"].unique()),
                               sorted(df["Emission_Class"].unique()))
    yr = f[2].slider("Model year", int(df.Model_Year.min()), int(df.Model_Year.max()),
                     (int(df.Model_Year.min()), int(df.Model_Year.max())))
    pol = f[3].selectbox("Pollutant", POLLUTANTS, index=3, key="dash_pol")
    d = df[df.Site.isin(sel_sites) & df.Emission_Class.isin(sel_cls) & df.Model_Year.between(*yr)]
    col = EF[pol]
    st.caption(f"{len(d)} trucks selected")

    if d.empty:
        st.info("No trucks match these filters.")
    else:
        a, b = st.columns(2)
        a.plotly_chart(px.box(d, x="Site", y=col, color="Site", points="all", category_orders={"Site": SITE_ORDER},
                              color_discrete_map=SITE_COLORS, hover_data=["Truck_No"],
                              title=f"{pol} by altitude", labels={col: f"{pol} (g/km)"}), width="stretch")
        b.plotly_chart(px.scatter(d, x="Model_Year", y=col, color="Site", size="Axles", symbol="Emission_Class",
                                  category_orders={"Site": SITE_ORDER}, color_discrete_map=SITE_COLORS,
                                  hover_data=["Truck_No", "Engine_Power_hp"],
                                  title=f"{pol} vs model year (bubble size = axles)",
                                  labels={col: f"{pol} (g/km)"}), width="stretch")
        a, b = st.columns(2)
        a.plotly_chart(px.box(d, x="Axles", y=col, color="Emission_Class", title=f"{pol} by number of axles",
                              labels={col: f"{pol} (g/km)"}), width="stretch")
        top = d.nlargest(10, f"{pol}_kg_month")
        b.plotly_chart(px.bar(top, x=f"{pol}_kg_month", y=top["Truck_No"].astype(str).radd("Truck "), color="Site",
                              orientation="h", color_discrete_map=SITE_COLORS,
                              title=f"Top 10 trucks by monthly {pol}",
                              labels={f"{pol}_kg_month": "kg / month", "y": ""}).update_yaxes(autorange="reversed"),
                       width="stretch")
        num = d[["Model_Year", "Axles", "Engine_Power_hp", "Avg_Speed_kmh", "VKT_km_month", "Altitude_m"]
                + list(EF.values())]
        st.plotly_chart(px.imshow(num.corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, aspect="auto", title="Correlation matrix"), width="stretch")


# ================================================================== 3. route comparison
with tabs[2]:
    st.subheader("One trip along the Karakoram Highway")
    seg = []
    km0 = 0
    for alt, km in ROUTE_KM.items():
        seg.append({"km": km0, "Altitude_m": alt, "Site": sites[alt]["name"]})
        km0 += km
        seg.append({"km": km0, "Altitude_m": alt, "Site": sites[alt]["name"]})
    prof = pd.DataFrame(seg)
    fig = px.area(prof, x="km", y="Altitude_m", title="Route altitude profile used in the paper (771 km)",
                  labels={"km": "distance from Islamabad (km)", "Altitude_m": "altitude (m)"})
    fig.update_traces(line_color="#6c757d", fillcolor="rgba(108,117,125,0.2)")
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    sm = site_means.reindex([s for s in SITE_ORDER if s in site_means.index])
    pct = (sm / sm.iloc[0] - 1) * 100
    pct.columns = POLLUTANTS
    a.plotly_chart(px.bar(pct.reset_index().melt(id_vars="Site", var_name="Pollutant", value_name="% vs Islamabad"),
                          x="Site", y="% vs Islamabad", color="Pollutant", barmode="group",
                          color_discrete_map=POL_COLORS, title="Fleet-average change compared with Islamabad"),
                   width="stretch")
    rb = route_budget(df, ROUTE_KM)
    melt = rb.melt(id_vars=["Site"], value_vars=[f"{p}_kg_per_trip" for p in ["CO", "HC", "NOx"]],
                   var_name="Pollutant", value_name="kg")
    melt["Pollutant"] = melt["Pollutant"].str.replace("_kg_per_trip", "")
    b.plotly_chart(px.bar(melt, x="Site", y="kg", color="Pollutant", color_discrete_map=POL_COLORS, barmode="group",
                          title="Average truck, one trip: kg emitted per segment"), width="stretch")
    st.dataframe(rb, hide_index=True, width="stretch")

    st.subheader("Emission per km vs per segment")
    per_km = sm.copy()
    per_km.columns = [f"{p} g/km" for p in POLLUTANTS]
    st.dataframe(per_km.round(2), width="stretch")


# ================================================================== 4. validation
with tabs[3]:
    val = validate(sites)
    summ = validation_summary(val)
    c = st.columns(4)
    for col, (_, r) in zip(c, summ.iterrows()):
        col.metric(f"{r.Pollutant} error (MAPE)", f"{r['MAPE_%']:.1f} %")
    vp = st.radio("Pollutant", POLLUTANTS, horizontal=True, key="val_pol")
    v = val[val.Pollutant == vp]
    fig = go.Figure([go.Bar(name="Measured (paper, Table S3)", x=v.Site, y=v.Measured_g_km, marker_color="#264653"),
                     go.Bar(name="Our formula estimate", x=v.Site, y=v.Estimated_g_km, marker_color="#f4a261")])
    fig.update_layout(barmode="group", title=f"{vp}: measured vs estimated (g/km)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(val, hide_index=True, width="stretch")
    st.caption("Tip: change Sost's temperature in the sidebar and watch the Sost error move.")


# ================================================================== 5. ML
with tabs[4]:
    c = st.columns(4)
    mpol = c[0].selectbox("Predict", POLLUTANTS, index=3, key="ml_pol")
    noise = c[1].slider("Measurement noise (%)", 0, 40, 15, 5,
                        help="Target = formula estimate × (1 + noise). Only 3 trucks have published measurements.")
    test_size = c[2].slider("Test set size", 0.15, 0.40, 0.20, 0.05)
    with c[3].popover("Hyper-parameters"):
        n_est = st.select_slider("RF trees", [50, 100, 200, 300, 500], 300)
        depth = st.select_slider("RF max depth", [2, 3, 5, 8, "None"], "None")
        lr = st.select_slider("GB learning rate", [0.01, 0.05, 0.1, 0.3], 0.05)
        alpha = st.select_slider("Ridge alpha", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0)

    y = make_target(df, mpol, noise)
    models = build_models(n_est, None if depth == "None" else depth, lr, alpha)
    res, fitted, split = train_and_compare(df, y, test_size, models)
    best = res["CV R2 (mean)"].idxmax()

    st.dataframe(res.style.format("{:.3f}").highlight_max(subset=["Test R2", "CV R2 (mean)"], color="#b7e4c7")
                 .highlight_min(subset=["Test RMSE", "Test MAE"], color="#b7e4c7"), width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(px.bar(res.reset_index(), x="Model", y=["Train R2", "Test R2", "CV R2 (mean)"], barmode="group",
                          title="R²: train vs test vs cross-validation"), width="stretch")
    chosen = b.selectbox("Model", list(fitted), index=list(fitted).index(best))
    yt, yp = split["y_test"], split["pred"][chosen]
    fig = px.scatter(x=yt, y=yp, labels={"x": f"actual {mpol} (g/km)", "y": f"predicted {mpol} (g/km)"},
                     title=f"Actual vs predicted, {chosen}")
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="grey"))
    b.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    imp = feature_importance(fitted[chosen])
    a.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", title=f"Feature importance, {chosen}",
                          labels={"x": "importance", "y": ""}), width="stretch")
    resid = pd.DataFrame({"Predicted": yp, "Residual": yt.values - yp, "Site": df.loc[yt.index, "Site"].values,
                          "Truck_No": df.loc[yt.index, "Truck_No"].values})
    fig = px.scatter(resid, x="Predicted", y="Residual", color="Site", hover_data=["Truck_No"],
                     color_discrete_map=SITE_COLORS, title="Residuals on the test set")
    fig.add_hline(y=0, line_dash="dash")
    b.plotly_chart(fig, width="stretch")

    st.subheader("Predict a new truck with ML")
    p = st.columns(6)
    nx = pd.DataFrame([{
        "Model_Year": p[0].number_input("Model year", 1965, 2025, 2015, key="ml_my"),
        "Axles": p[1].selectbox("Axles", [2, 3, 4, 5, 6], 2, key="ml_ax"),
        "Engine_Power_hp": p[2].number_input("Power (hp)", 100, 450, 240, 10, key="ml_rp"),
        "Avg_Speed_kmh": p[3].number_input("Speed (km/h)", 10, 80, 40, 5, key="ml_sp"),
        "VKT_km_month": 6000,
        "Altitude_m": p[4].selectbox("Altitude (m)", list(SITES), 0, key="ml_alt"),
        "Emission_Class": p[5].selectbox("Class", ["Pak-II", "Pre-Pak-II"], key="ml_cls"),
    }])[FEATURES]
    preds = {name: float(m.predict(nx)[0]) for name, m in fitted.items()}
    met = sites[int(nx.Altitude_m[0])]
    preds["Paper formula"] = float(emission_factor(mpol, met["RH"], met["T"], nx.Engine_Power_hp[0],
                                                   nx.Model_Year[0], nx.Axles[0]))
    st.plotly_chart(px.bar(x=list(preds), y=list(preds.values()), color=list(preds),
                           labels={"x": "", "y": f"{mpol} (g/km)"},
                           title=f"Predicted {mpol} for this truck").update_layout(showlegend=False), width="stretch")

    if st.button("Run grid search (Random Forest)"):
        with st.spinner("Trying 18 combinations × 5 folds..."):
            gs, table = tune_random_forest(split["X_train"], split["y_train"])
        st.success(f"Best: {({k.replace('model__', ''): v for k, v in gs.best_params_.items()})}, "
                   f"CV R² = {gs.best_score_:.3f}")
        st.dataframe(table.round(3), hide_index=True, width="stretch")


# ================================================================== 6. clusters
with tabs[5]:
    labels, sil, k_best, cols = cluster_trucks(df)
    kk = st.slider("Number of clusters", 2, 6, int(k_best), help=f"Best by silhouette score: {k_best}")
    labels, _, _, _ = cluster_trucks(df, kk)
    dc = df.assign(Cluster=("Group " + pd.Series(labels + 1, index=df.index).astype(str)))
    a, b = st.columns([2, 1])
    a.plotly_chart(px.scatter(dc, x="Model_Year", y=EF["CO2"], color="Cluster", symbol="Site", size="Axles",
                              hover_data=["Truck_No", "Engine_Power_hp"], title="Truck groups (K-Means)",
                              labels={EF["CO2"]: "CO2 (g/km)"}), width="stretch")
    b.plotly_chart(px.line(x=sil.index, y=sil.values, markers=True, title="Silhouette score",
                           labels={"x": "clusters", "y": "score"}), width="stretch")
    prof = dc.groupby("Cluster")[cols].mean().round(1)
    prof.insert(0, "Trucks", dc.groupby("Cluster").size())
    st.dataframe(prof, width="stretch")


# ================================================================== 7. data
with tabs[6]:
    view = st.radio("Show", ["Clean data with emissions", "Raw data", "Data-quality check"], horizontal=True)
    if view == "Raw data":
        st.dataframe(raw, height=500, hide_index=True, width="stretch")
    elif view == "Data-quality check":
        st.dataframe(data_quality_report(raw), hide_index=True, width="stretch")
        st.write(f"Engine power filled in for **{int(df.Power_was_imputed.sum())}** trucks (median of same-axle "
                 f"trucks). Speed ranges converted to mid-point for **{int(df.Speed_was_range.sum())}** trucks.")
        st.dataframe(df[["Model_Year", "Axles", "Engine_Power_hp", "Avg_Speed_kmh", "VKT_km_month"]]
                     .describe().round(1), width="stretch")
    else:
        show = ["Truck_No", "Site", "Model_Year", "Emission_Class", "Axles", "Engine_Power_hp", "Avg_Speed_kmh",
                "VKT_km_month"] + list(EF.values()) + [f"{p}_kg_month" for p in POLLUTANTS] + ["CO2eq_kg_month"]
        st.dataframe(df[show].round(2), height=500, hide_index=True, width="stretch")
    st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False), "cpec_truck_emissions_results.csv")
