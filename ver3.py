import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

st.title("Groundwater Pre-Calibration Tool")

st.write(
"""
Conceptual groundwater calibration tool for estimating hydraulic conductivity (Kf).
All parameters use consistent units (meters and days).
"""
)

# -----------------------------
# SIDEBAR PARAMETERS
# -----------------------------

st.sidebar.header("Aquifer Parameters")

R = st.sidebar.number_input(
    "Recharge R (m/day)",
    value=0.0003,
    format="%.6f"
)

L = st.sidebar.number_input(
    "Aquifer Length L (m)",
    value=1000.0
)

h1 = st.sidebar.number_input(
    "Upstream Boundary Head h1 (m)",
    value=52.0
)

h2 = st.sidebar.number_input(
    "Downstream Boundary Head h2 (m)",
    value=49.0
)

# -----------------------------
# OBSERVATION WELLS
# -----------------------------

st.subheader("Observation Wells")

n_wells = st.number_input(
    "Number of wells",
    min_value=1,
    max_value=20,
    value=3
)

data = []

for i in range(n_wells):

    col1, col2 = st.columns(2)

    with col1:
        x = st.number_input(
            f"Distance of Well {i+1} from upstream boundary (m)",
            value=200*(i+1),
            key=f"x{i}"
        )

    with col2:
        head = st.number_input(
            f"Observed Head Well {i+1} (m)",
            value=50.0,
            key=f"h{i}"
        )

    data.append([x, head])

df_obs = pd.DataFrame(data, columns=["Distance","Observed Head"])

# -----------------------------
# Kf RANGE
# -----------------------------

st.sidebar.header("Hydraulic Conductivity")

kf_min = st.sidebar.number_input(
    "Minimum Kf (m/day)",
    value=0.1
)

kf_max = st.sidebar.number_input(
    "Maximum Kf (m/day)",
    value=100.0
)

log_min = np.log10(kf_min)
log_max = np.log10(kf_max)

log_kf = st.sidebar.slider(
    "log10(Kf)",
    float(log_min),
    float(log_max),
    float((log_min+log_max)/2)
)

kf = 10**log_kf

st.sidebar.write("Selected Kf:", round(kf,3),"m/day")

# -----------------------------
# SIMULATION FUNCTION
# -----------------------------

def simulate_heads(kf):

    heads = []

    for x in df_obs["Distance"]:

        h_sq = (
            h1**2
            - ((h1**2 - h2**2)/L) * x
            + (R/kf) * x * (L-x)
        )

        h = np.sqrt(max(h_sq,0))

        heads.append(h)

    return np.array(heads)

# -----------------------------
# RUN SIMULATION
# -----------------------------

sim_heads = simulate_heads(kf)

df_obs["Simulated Head"] = sim_heads
df_obs["Residual"] = df_obs["Simulated Head"] - df_obs["Observed Head"]

rmse = np.sqrt(np.mean(df_obs["Residual"]**2))

st.subheader("Simulation Results")

st.dataframe(df_obs)

st.metric("RMSE", round(rmse,3))


# -----------------------------
# SENSITIVITY ANALYSIS
# -----------------------------

st.subheader("Sensitivity Analysis")

kf_range = np.logspace(log_min,log_max,100)

rmse_curve = []

for k in kf_range:

    sim = simulate_heads(k)
    res = sim - df_obs["Observed Head"].values
    rmse_curve.append(np.sqrt(np.mean(res**2)))

df_curve = pd.DataFrame({
    "Kf":kf_range,
    "RMSE":rmse_curve
})

fig = px.line(
    df_curve,
    x="Kf",
    y="RMSE",
    log_x=True,
    title="Calibration Error vs Hydraulic Conductivity"
)

st.plotly_chart(fig)

# -----------------------------
# EQUATION DISPLAY
# -----------------------------

st.markdown("---")
st.header("Governing Groundwater Equation")

st.latex(r'''
h(x)^2 =
h_1^2 -
\frac{(h_1^2-h_2^2)}{L}x +
\frac{R}{K_f}x(L-x)
''')

st.write(
"""
This equation represents steady-state groundwater flow in an aquifer receiving uniform recharge.
"""
)

# -----------------------------
# VARIABLE DEFINITIONS
# -----------------------------

st.header("Variable Definitions")

definitions = {
"Variable":[
"h(x)",
"h1",
"h2",
"R",
"Kf",
"x",
"L"
],

"Description":[
"Simulated groundwater head at distance x",
"Groundwater head at upstream boundary",
"Groundwater head at downstream boundary",
"Recharge entering the aquifer",
"Hydraulic conductivity of the aquifer",
"Distance from upstream boundary along groundwater flow direction",
"Total groundwater flow length between boundaries"
],

"Units":[
"m",
"m",
"m",
"m/day",
"m/day",
"m",
"m"
]
}

st.table(pd.DataFrame(definitions))

st.write(
"""
Conceptual assumptions:

• steady-state groundwater conditions  
• homogeneous aquifer properties  
• uniform recharge  
• one-dimensional flow along the hydraulic gradient
"""
)

st.write(
"""
This simplified conceptual model helps estimate realistic Kf values before building full numerical groundwater models.
"""
)
