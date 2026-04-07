import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

st.title("Groundwater Pre-Calibration Tool")

st.write("""
Conceptual groundwater calibration tool for estimating hydraulic conductivity (Kf).
All parameters use consistent units (meters and days).
""")

# ------------------------------------------------
# SIDEBAR PARAMETERS
# ------------------------------------------------

st.sidebar.header("Aquifer Parameters")

R = st.sidebar.number_input(
    "Recharge R (m/day)",
    value=0.0003,
    format="%.6f"
)

L = st.sidebar.number_input(
    "Aquifer Length L (m)",
    value=1000.0,
    min_value=1.0
)

h1 = st.sidebar.number_input(
    "Upstream Boundary Head h1 (m)",
    value=52.0
)

h2 = st.sidebar.number_input(
    "Downstream Boundary Head h2 (m)",
    value=49.0
)

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------

st.subheader("Load Observation Wells")

uploaded_file = st.file_uploader(
    "Upload Excel or CSV file",
    type=["xlsx","csv"]
)

st.info("File must contain columns: 'Distance' and 'Observed Head'")

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df_obs = pd.read_csv(uploaded_file)

    else:
        df_obs = pd.read_excel(uploaded_file)

    st.success("File loaded successfully")
    st.dataframe(df_obs)

else:

    # ------------------------------------------------
    # MANUAL WELL INPUT
    # ------------------------------------------------

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
                f"Distance Well {i+1} (m)",
                min_value=0.0,
                max_value=L,
                value=min(200*(i+1),L),
                key=f"x{i}"
            )

        with col2:
            head = st.number_input(
                f"Observed Head Well {i+1} (m)",
                value=50.0,
                key=f"h{i}"
            )

        data.append([x,head])

    df_obs = pd.DataFrame(
        data,
        columns=["Distance","Observed Head"]
    )

# ------------------------------------------------
# HYDRAULIC CONDUCTIVITY
# ------------------------------------------------

st.sidebar.header("Hydraulic Conductivity")

kf_min = st.sidebar.number_input(
    "Minimum Kf (m/day)",
    value=0.1,
    min_value=1e-6
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

st.sidebar.write("Selected Kf:",round(kf,4),"m/day")

# ------------------------------------------------
# SIMULATION FUNCTION
# ------------------------------------------------

def simulate_heads(kf):

    heads = []

    for x in df_obs["Distance"].values:

        h_sq = (
            h1**2
            - ((h1**2-h2**2)/L)*x
            + (R/kf)*x*(L-x)
        )

        if h_sq < 0:
            h = np.nan
        else:
            h = np.sqrt(h_sq)

        heads.append(h)

    return np.array(heads,dtype=float)

# ------------------------------------------------
# RUN SIMULATION
# ------------------------------------------------

sim_heads = simulate_heads(kf)

df_obs["Simulated Head"] = sim_heads
df_obs["Residual"] = df_obs["Simulated Head"] - df_obs["Observed Head"]

rmse = np.sqrt(np.nanmean(df_obs["Residual"]**2))

st.subheader("Simulation Results")

st.dataframe(df_obs)

st.metric("RMSE",round(rmse,3))

# ------------------------------------------------
# AUTOMATIC CALIBRATION
# ------------------------------------------------

st.subheader("Automatic Kf Calibration")

def objective(log_k):

    log_k = float(log_k)

    k = 10**log_k

    sim = simulate_heads(k)

    residuals = sim - df_obs["Observed Head"].values

    return float(np.nanmean(residuals**2))

result = minimize(
    objective,
    x0=np.log10(kf),
    bounds=[(log_min,log_max)]
)

if result.success:
    best_kf = 10**result.x[0]
else:
    best_kf = np.nan

st.write("Estimated Best Kf:",round(best_kf,4),"m/day")

# ------------------------------------------------
# GROUNDWATER PROFILE
# ------------------------------------------------

st.subheader("Groundwater Profile")

plot_df = df_obs.sort_values("Distance")

fig_profile = px.scatter(
    plot_df,
    x="Distance",
    y="Observed Head",
    title="Observed vs Simulated Heads"
)

fig_profile.add_scatter(
    x=plot_df["Distance"],
    y=plot_df["Simulated Head"],
    mode="markers+lines",
    name="Simulated Head"
)

x_profile = np.linspace(0,L,200)

h_profile = []

for x in x_profile:

    h_sq = (
        h1**2
        - ((h1**2-h2**2)/L)*x
        + (R/kf)*x*(L-x)
    )

    h_profile.append(np.sqrt(max(h_sq,0)))

fig_profile.add_scatter(
    x=x_profile,
    y=h_profile,
    mode="lines",
    name="Model Water Table"
)

st.plotly_chart(fig_profile)

# ------------------------------------------------
# SENSITIVITY ANALYSIS
# ------------------------------------------------

st.subheader("Sensitivity Analysis")

kf_range = np.logspace(log_min,log_max,100)

rmse_curve = []

for k in kf_range:

    sim = simulate_heads(k)
    res = sim - df_obs["Observed Head"].values

    rmse_curve.append(
        np.sqrt(np.nanmean(res**2))
    )

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

# ------------------------------------------------
# MONTE CARLO UNCERTAINTY ANALYSIS
# ------------------------------------------------

st.subheader("Monte Carlo Uncertainty Analysis")

n_sim = st.slider(
    "Number of simulations",
    100,
    5000,
    1000,
    step=100
)

if st.button("Run Monte Carlo Simulation"):

    kf_samples = np.random.uniform(kf_min,kf_max,n_sim)

    rmse_list = []

    for k in kf_samples:

        sim = simulate_heads(k)

        res = sim - df_obs["Observed Head"].values

        rmse_val = np.sqrt(np.nanmean(res**2))

        rmse_list.append(rmse_val)

    df_mc = pd.DataFrame({
        "Kf":kf_samples,
        "RMSE":rmse_list
    })

    st.dataframe(df_mc.head())

    fig_mc = px.histogram(
        df_mc,
        x="Kf",
        nbins=40,
        title="Monte Carlo Distribution of Kf"
    )

    st.plotly_chart(fig_mc)

    best_rmse = df_mc["RMSE"].min()

    threshold = best_rmse*1.2

    accepted = df_mc[df_mc["RMSE"]<=threshold]

    kf_low = accepted["Kf"].min()
    kf_high = accepted["Kf"].max()

    st.write(
        "Estimated Kf confidence range:",
        round(kf_low,3),
        "to",
        round(kf_high,3),
        "m/day"
    )

    fig_scatter = px.scatter(
        df_mc,
        x="Kf",
        y="RMSE",
        title="Monte Carlo Calibration Space"
    )

    st.plotly_chart(fig_scatter)

# ------------------------------------------------
# GOVERNING EQUATION
# ------------------------------------------------

st.markdown("---")

st.header("Governing Groundwater Equation")

st.latex(r'''
h(x)^2 =
h_1^2 -
\frac{(h_1^2-h_2^2)}{L}x +
\frac{R}{K_f}x(L-x)
''')

st.write("Steady-state groundwater flow in a 1D aquifer with uniform recharge.")

# ------------------------------------------------
# VARIABLE DEFINITIONS
# ------------------------------------------------

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
"Hydraulic conductivity",
"Distance from upstream boundary",
"Aquifer flow length"
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

st.write("""
Conceptual assumptions

• steady-state groundwater conditions  
• homogeneous aquifer properties  
• uniform recharge  
• one-dimensional groundwater flow
""")

st.write("""
This conceptual tool helps estimate realistic hydraulic conductivity ranges before building numerical groundwater models such as MODFLOW.
""")