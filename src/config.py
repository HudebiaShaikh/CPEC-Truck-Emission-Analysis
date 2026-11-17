"""
Project-wide constants.

Everything that comes from the research paper is kept here, so you can point at
one file in the viva and say "these numbers are from Shaikh et al. (2023)".
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "CPEC_Truck_Emissions_70_Trucks.csv"
OUTPUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42  # fixed seed -> every run gives the same results (reproducibility)

POLLUTANTS = ["CO", "HC", "NOx", "CO2"]

# ---------------------------------------------------------------------------
# Table S4 (supplement, data.pdf page 9): Multiple Linear Regression (MLR) of each gas
# EF = b_RH*RH + b_T*T + b_RP*RP + b_MY*MY + b_A*A + intercept      (EF in g/km)
# RH = relative humidity (%), T = ambient temperature (deg C), RP = rated power (hp),
# MY = model year, A = number of axles
# ---------------------------------------------------------------------------
MLR_COEFFS = {
    "CO":  {"RH": -0.0368, "T": -0.2081,  "RP": -0.0008, "MY": -0.0838, "A": 2.4702,  "intercept": 179.7821},
    "HC":  {"RH": -0.0556, "T": -0.2141,  "RP":  0.0002, "MY": -0.0783, "A": 2.4059,  "intercept": 171.3120},
    "NOx": {"RH": -0.0228, "T": -0.1682,  "RP": -0.0004, "MY": -0.0622, "A": 2.1385,  "intercept": 133.5380},
    "CO2": {"RH": -4.4100, "T": -12.2828, "RP":  0.0741, "MY": -2.7862, "A": 78.6754, "intercept": 6308.8218},
}

# ---------------------------------------------------------------------------
# Ambient conditions per test site.
# IMPORTANT ASSUMPTION: the paper does NOT publish the temperature / humidity
# recorded for each truck. These are typical day-time values for each site and
# are an assumption of this project. You can change them live in the dashboard
# sidebar and see how the results move (good for the viva!).
# ---------------------------------------------------------------------------
SITES = {
    540:  {"name": "Islamabad",      "T": 25.0, "RH": 55.0},
    2800: {"name": "Sost",           "T": 15.0, "RH": 45.0},
    4693: {"name": "Khunjerab Pass", "T": 5.0,  "RH": 40.0},
}

# Route length assigned to each altitude band (supplement, page 4)
ROUTE_KM = {540: 425, 2800: 283, 4693: 63}

# Table S5: Global Warming Potential (100-year), relative to CO2
GWP_100 = {"CO2": 1, "CO": 2, "NOx": 28}
GWP_20 = {"CO2": 1, "CO": 6, "NOx": 43}

# Table S3: measured emission factors (g/km) of 3 trucks -> used for validation.
# Matched to truck numbers in Table S2 by axles / model year / power / site.
VALIDATION_TRUCKS = [
    {"Truck_No": 46, "Site": "Islamabad",      "Axles": 6, "Model_Year": 2010, "Engine_Power_hp": 320,
     "CO": 19.98, "HC": 21.89, "NOx": 17.51, "CO2": 650.71},
    {"Truck_No": 65, "Site": "Sost",           "Axles": 6, "Model_Year": 2019, "Engine_Power_hp": 340,
     "CO": 14.61, "HC": 16.68, "NOx": 11.00, "CO2": 683.49},
    {"Truck_No": 70, "Site": "Khunjerab Pass", "Axles": 6, "Model_Year": 2019, "Engine_Power_hp": 340,
     "CO": 23.09, "HC": 23.96, "NOx": 19.78, "CO2": 881.75},
]

# Average emission factors reported in the paper's abstract (Islamabad, g/km)
PAPER_ISLAMABAD_MEANS = {
    "Pre-Pak-II": {"CO": 12.94, "HC": 15.21, "NOx": 10.69},
    "Pak-II":     {"CO": 12.75, "HC": 14.24, "NOx": 10.24},
}
