# CPEC Truck Emission Analysis 🚛

An interactive data science web app (Streamlit) that estimates **CO, HC, NOx and CO₂** emissions of
70 freight trucks on the China-Pakistan Economic Corridor, shows how **altitude, truck age and truck
type** affect them, validates the estimates against measured trucks, and compares **ML models**.

Scientific basis: Shaikh et al. (2023), *Altitude-dependent gaseous emissions from freight trucks along
the China-Pakistan Economic Corridor in Pakistan*, Environmental Science and Ecotechnology 14, 100226.

---

## Run it in VS Code (step by step)

**You need:** Python 3.10 or newer ([python.org](https://www.python.org/downloads/), tick
"Add Python to PATH" on Windows) and VS Code with the **Python** extension.

1. Unzip the folder and open it in VS Code: **File → Open Folder → `cpec_truck_emission_app`**.
2. Open the terminal: **Terminal → New Terminal**.
3. Create a virtual environment (one time only):

   **Windows**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
   **Mac / Linux**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   > On Windows, if you get a "running scripts is disabled" error, run
   > `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

4. Install the libraries (one time only):
   ```
   pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```
   streamlit run app.py
   ```
   Your browser opens at **http://localhost:8501**. Press `Ctrl + C` in the terminal to stop it.

Shortcut: in VS Code press **F5** and pick **"Run Streamlit dashboard"** (set up in `.vscode/launch.json`).

To get every number in the terminal (and CSV tables in `outputs/` for your report):
```
python run_analysis.py
```

---

## Project structure

```
cpec_truck_emission_app/
├── app.py                  # Streamlit web app (7 tabs)
├── run_analysis.py         # same analysis without the UI, saves tables to outputs/
├── requirements.txt
├── data/
│   └── CPEC_Truck_Emissions_70_Trucks.csv   # Table S2 of the paper (raw)
├── src/
│   ├── config.py           # paper coefficients (Table S4), GWP (S5), validation trucks (S3), site weather
│   ├── data_processing.py  # cleaning + feature engineering
│   ├── emission_model.py   # applies the MLR formula, monthly kg, CO2-eq, validation
│   └── ml_models.py        # ML pipelines, cross-validation, grid search, K-Means
├── REPORT_GUIDE.md         # results + what to write in each report section
└── .vscode/                # run configurations
```

## What the app does

| Tab | What you can do |
|---|---|
| 🧮 Emission calculator | pick any of the 70 trucks (or enter your own), change year, axles, power, site, weather → get CO, HC, NOx, CO₂ per km, per month and CO₂-eq, and the same truck at all 3 sites |
| 📊 Fleet dashboard | filter by site, emission class, model year → charts by altitude, age, axles, top-10 polluters, correlation |
| 🛣️ Route comparison | Karakoram Highway altitude profile, % increase vs Islamabad, kg per trip segment |
| ✅ Validation | formula vs the 3 measured trucks in the paper (Table S3), error % |
| 🤖 ML prediction | compare Linear, Ridge, Random Forest, Gradient Boosting (R², RMSE, MAE, CV), change hyper-parameters, predict a new truck, run grid search |
| 🧩 Truck clusters | K-Means groups of trucks |
| 📁 Data | raw / clean data, quality check, download results CSV |

Sidebar: upload your own truck CSV (same columns) and change the weather at each site.

## Two assumptions you must be able to explain

1. **Site weather is assumed.** The paper's formula needs temperature (T) and humidity (RH), but the paper
   does not publish them per truck. Defaults: Islamabad 25 °C / 55 %, Sost 15 °C / 45 %, Khunjerab 5 °C / 40 %
   (in `src/config.py`).
2. **The ML target is semi-synthetic.** Only 3 trucks have published measurements, so the ML target is
   *formula estimate × (1 + 15 % random noise)* to imitate on-road measurement scatter. The ML part
   therefore asks: *"can we predict emissions from truck and route data alone, without weather?"*
   Say this openly in the report; the guidelines forbid presenting fabricated results as real.

## Troubleshooting

* `streamlit: command not found` → the venv is not active; run the activate command from step 3, or use
  `python -m streamlit run app.py`.
* `ModuleNotFoundError: src` → run commands from inside the `cpec_truck_emission_app` folder.
* Port busy → `streamlit run app.py --server.port 8502`.
