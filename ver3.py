import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

st.title("Groundwater Pre-Calibration Tool")

st.write("Conceptual groundwater calibration tool based on steady-state recharge conditions.")

# -------------------------
# SIDEBAR PARAMETERS
# -------------------------

st.sidebar.header("Aquifer Parameters")

R = st.sidebar.number_input(
    "Recharge R (m/day)",
    value=0.0003,
    format="%.6f"
)

L = st.sidebar.number_input(
    "Aquifer Flow Length L (m)",
    value=1000.0
)

h0 = st.sidebar.number_input(
    "Boundary Head h0 (m)",
    value=50.0
)

# -------------------------
# OBSERVATION WELLS
# -------------------------

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
            f"Distance x{i+1} from boundary (m)",
            value=100*(i+1),
            key=f"x{i}"
        )

    with col2:
        head = st.number_input(
            f"Observed Head h{i+1} (m)",
            value=50.0,
            key=f"h{i}"
        )

    data.append([x, head])

df_obs = pd.DataFrame(data, columns=["Distance","Observed Head"])

# -------------------------
# Kf RANGE
# -------------------------

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

# -------------------------
# SIMULATION FUNCTION
# -------------------------

def simulate_heads(kf):

    heads = []

    for x in df_obs["Distance"]:
        h = np.sqrt(h0**2 + (R/kf) * x * (L-x))
        heads.append(h)

    return np.array(heads)

# -------------------------
# MANUAL SIMULATION
# -------------------------

sim_heads = simulate_heads(kf)

df_obs["Simulated Head"] = sim_heads
df_obs["Residual"] = df_obs["Simulated Head"] - df_obs["Observed Head"]

rmse = np.sqrt(np.mean(df_obs["Residual"]**2))

st.subheader("Simulation Results")

st.dataframe(df_obs)

st.metric("RMSE", round(rmse,3))

# -------------------------
# AUTOMATIC CALIBRATION
# -------------------------

st.subheader("Automatic Kf Calibration")

def objective(log_k):

    k = 10**log_k
    sim = simulate_heads(k)
    residuals = sim - df_obs["Observed Head"].values
    return np.mean(residuals**2)

result = minimize(
    objective,
    x0=np.log10(kf),
    bounds=[(log_min,log_max)]
)

best_kf = 10**result.x[0]

st.write("Estimated Best Kf:", round(best_kf,3),"m/day")

# -------------------------
# SENSITIVITY PLOT
# -------------------------

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

# -------------------------
# EQUATION
# -------------------------

st.markdown("---")
st.header("Governing Equation")

st.latex(r'''
h(x)=\sqrt{h_0^2+\frac{R}{K_f}x(L-x)}
''')

st.write(
"This equation describes steady groundwater flow with uniform recharge in a one-dimensional aquifer."
)

# -------------------------
# VARIABLE DEFINITIONS
# -------------------------

st.header("Variable Definitions")

definitions = {
"Variable":[
"h(x)",
"h0",
"R",
"Kf",
"x",
"L"
],

"Description":[
"Simulated groundwater head at distance x",
"Groundwater head at the upstream boundary",
"Areal recharge entering the aquifer",
"Hydraulic conductivity of the aquifer material",
"Distance from the upstream boundary along the groundwater flow direction",
"Total groundwater flow length between hydraulic boundaries"
],

"Units":[
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
The conceptual model assumes:

• steady-state groundwater flow  
• homogeneous aquifer properties  
• uniform recharge across the aquifer  
• one-dimensional flow along the hydraulic gradient
"""
)
