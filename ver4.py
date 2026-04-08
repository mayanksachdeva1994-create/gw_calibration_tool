import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar
 
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
 
st.set_page_config(
    page_title="Groundwater Pre-Calibration Tool",
    page_icon="💧",
    layout="wide"
)
 
st.title("💧 Groundwater Pre-Calibration Tool")
st.markdown(
    """
    Steady-state 1D analytical calibration for **unconfined aquifers** using the
    Dupuit–Forchheimer equation. Estimates hydraulic conductivity **Kf** by
    minimising RMSE against observed well heads.
    
    > ⚠️ **Assumption**: Unconfined aquifer, homogeneous & isotropic, uniform recharge, 1D flow.  
    > For confined aquifers the governing equation is linear in h (not h²) — use a different model.
    """
)
 
st.markdown("---")
 
# ─────────────────────────────────────────────
# SIDEBAR — AQUIFER PARAMETERS
# ─────────────────────────────────────────────
 
st.sidebar.header("⚙️ Aquifer Parameters")
 
R = st.sidebar.number_input(
    "Recharge R (m/day)",
    min_value=0.0,
    value=0.0003,
    format="%.6f",
    help="Typical range: 0.00001–0.005 m/day (0.004–1.8 m/yr). "
         "0.0003 m/day ≈ 11 cm/yr (semi-arid)."
)
 
L = st.sidebar.number_input(
    "Aquifer Length L (m)",
    min_value=1.0,
    value=1000.0,
    help="Distance between upstream and downstream fixed-head boundaries."
)
 
h1 = st.sidebar.number_input(
    "Upstream Head h₁ (m)",
    value=52.0,
    help="Fixed head at the upstream boundary (x = 0)."
)
 
h2 = st.sidebar.number_input(
    "Downstream Head h₂ (m)",
    value=49.0,
    help="Fixed head at the downstream boundary (x = L)."
)
 
# ─────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────
 
errors = []
 
if L <= 0:
    errors.append("Aquifer length L must be > 0.")
if h1 <= 0 or h2 <= 0:
    errors.append("Boundary heads h₁ and h₂ must be positive (unconfined aquifer).")
if h1 < h2:
    errors.append("Upstream head h₁ should be ≥ downstream head h₂ for typical recharge conditions.")
if R < 0:
    errors.append("Recharge R cannot be negative.")
 
if errors:
    for e in errors:
        st.error(e)
    st.stop()
 
# ─────────────────────────────────────────────
# OBSERVATION WELLS
# ─────────────────────────────────────────────
 
st.subheader("🔵 Observation Wells")
 
n_wells = st.number_input(
    "Number of observation wells",
    min_value=1,
    max_value=20,
    value=3,
    help="Enter at least 2 wells for a meaningful calibration."
)
 
data = []
 
# Shadow keys (_sv_h{i}, _sv_x{i}) are written by the synthetic loader BEFORE
# widgets render. The number_input widgets read from them as default values.
# We never write directly to the widget keys (h{i}, x{i}) after render —
# that is what causes StreamlitAPIException.
 
cols_header = st.columns([1, 2, 2])
cols_header[0].markdown("**Well #**")
cols_header[1].markdown("**Distance from upstream (m)**")
cols_header[2].markdown("**Observed Head (m)**")
 
for i in range(int(n_wells)):
    col0, col1, col2 = st.columns([1, 2, 2])
 
    # Read from shadow keys if synthetic loader has populated them
    default_x = float(st.session_state.get(f"_sv_x{i}", min(float(200 * (i + 1)), float(L) * 0.9)))
    default_h = float(st.session_state.get(f"_sv_h{i}", 50.0))
 
    with col0:
        st.markdown(f"<br>Well {i+1}", unsafe_allow_html=True)
    with col1:
        x = st.number_input(
            f"x_{i+1}",
            min_value=0.0,
            max_value=float(L),
            value=default_x,
            key=f"x{i}",
            label_visibility="collapsed"
        )
    with col2:
        head = st.number_input(
            f"h_{i+1}",
            value=default_h,
            key=f"h{i}",
            label_visibility="collapsed"
        )
    data.append([x, head])
 
df_obs = pd.DataFrame(data, columns=["Distance", "Observed Head"])
 
# Validate well positions
if df_obs["Distance"].duplicated().any():
    st.warning("⚠️ Two or more wells share the same position. Results may be misleading.")
 
if (df_obs["Distance"] <= 0).any() or (df_obs["Distance"] >= L).any():
    st.warning("⚠️ Well positions should be strictly between 0 and L (not on boundaries).")
 
# ─────────────────────────────────────────────
# SYNTHETIC OBSERVATION GENERATOR
# ─────────────────────────────────────────────
 
st.subheader("🧪 Synthetic Observation Generator")
 
with st.expander("Generate synthetic well heads from a known Kf (for testing & understanding)", expanded=False):
 
    st.markdown(
        """
        Use this to **test the calibration** or to understand why flat observed heads
        cause the optimizer to prefer high Kf values.  
        
        Enter a *known* (true) Kf → compute what the heads **should** look like at your
        well positions → load them directly into the observation table above.
        
        > 💡 The calibration should then recover this Kf exactly (RMSE → 0).
        """
    )
 
    kf_true = st.number_input(
        "True Kf for synthetic generation (m/day)",
        min_value=1e-6,
        value=5.0,
        format="%.4f",
        help="This is the 'ground truth' Kf you want the calibration to rediscover."
    )
 
    # Preview the synthetic heads before loading
    x_wells = df_obs["Distance"].values
 
    def simulate_heads_generic(kf, x_values, h1_, h2_, L_, R_):
        """Standalone version so it works before df_obs is finalised."""
        h_sq = (
            h1_ ** 2
            - ((h1_ ** 2 - h2_ ** 2) / L_) * x_values
            + (R_ / kf) * x_values * (L_ - x_values)
        )
        return np.sqrt(np.maximum(h_sq, 0))
 
    synthetic_heads = simulate_heads_generic(kf_true, x_wells, h1, h2, L, R)
 
    # Show preview table
    df_preview = pd.DataFrame({
        "Well": [f"Well {i+1}" for i in range(len(x_wells))],
        "Distance (m)": x_wells,
        "Synthetic Head (m)": np.round(synthetic_heads, 4),
        "Linear Baseline (m)": np.round(
            simulate_heads_generic(1e9, x_wells, h1, h2, L, R), 4
        ),
        "Mound above baseline (m)": np.round(
            synthetic_heads - simulate_heads_generic(1e9, x_wells, h1, h2, L, R), 4
        ),
    })
 
    st.dataframe(df_preview, use_container_width=True)
 
    # Check if the mound is detectable — warn user if signal is too weak
    max_mound = df_preview["Mound above baseline (m)"].max()
    if max_mound < 0.01:
        st.warning(
            f"⚠️ Maximum recharge mound is only **{max_mound:.4f} m** above the linear baseline. "
            "This signal is too weak — the calibration will still prefer high Kf. "
            "Try increasing R or decreasing Kf_true."
        )
    else:
        st.success(
            f"✅ Maximum recharge mound = **{max_mound:.4f} m** above baseline. "
            "This signal is strong enough for the calibration to identify Kf."
        )
 
    # ── LOAD BUTTON ──────────────────────────────────────────────────
    # Only active if mound is detectable (> 0.01 m)
    load_disabled = bool(max_mound < 0.01)
 
    load_clicked = st.button(
        "📥 Load synthetic heads into observation wells",
        disabled=load_disabled,
        type="primary",
        help="Disabled when the recharge mound is too small to be informative."
        if load_disabled else
        "Click to overwrite the observed heads above with these synthetic values."
    )
 
    if load_clicked:
        # Write to SHADOW keys only — never to widget keys after render
        for i, h_val in enumerate(synthetic_heads):
            st.session_state[f"_sv_h{i}"] = float(round(h_val, 4))
        st.session_state["loaded_kf_true"] = float(kf_true)
        st.success(
            f"✅ Synthetic heads loaded for Kf_true = {kf_true} m/day. "
            "The observation wells above have been updated — run Auto-Calibration to recover this value."
        )
        st.rerun()
 
    # Show reminder if synthetic data was previously loaded
    if "loaded_kf_true" in st.session_state:
        st.info(
            f"ℹ️ Currently loaded: synthetic heads generated from Kf_true = "
            f"**{st.session_state['loaded_kf_true']} m/day**. "
            "Auto-calibration should recover this value."
        )
 
# ─────────────────────────────────────────────
# Kf RANGE & MANUAL SLIDER
# ─────────────────────────────────────────────
 
st.sidebar.header("🔬 Hydraulic Conductivity")
 
kf_min = st.sidebar.number_input("Minimum Kf (m/day)", value=0.1, min_value=1e-6)
kf_max = st.sidebar.number_input("Maximum Kf (m/day)", value=100.0)
 
if kf_min >= kf_max:
    st.sidebar.error("Kf min must be less than Kf max.")
    st.stop()
 
log_min = np.log10(kf_min)
log_max = np.log10(kf_max)
 
log_kf = st.sidebar.slider(
    "Manual log₁₀(Kf)",
    float(log_min),
    float(log_max),
    float((log_min + log_max) / 2),
    help="Slide to manually explore the effect of Kf on heads."
)
 
kf_manual = 10 ** log_kf
st.sidebar.write(f"**Selected Kf:** {round(kf_manual, 4)} m/day")
 
# ─────────────────────────────────────────────
# SIMULATION FUNCTION
# ─────────────────────────────────────────────
 
def simulate_heads(kf, x_values):
    """
    Dupuit-Forchheimer analytical solution for steady-state unconfined
    flow between two fixed-head boundaries with uniform recharge.
 
    h(x)² = h1² - [(h1²-h2²)/L]·x + (R/Kf)·x·(L-x)
    """
    h_sq = (
        h1 ** 2
        - ((h1 ** 2 - h2 ** 2) / L) * x_values
        + (R / kf) * x_values * (L - x_values)
    )
    # Guard against numerical negatives (non-physical)
    h_sq = np.maximum(h_sq, 0)
    return np.sqrt(h_sq)
 
 
def rmse(kf):
    sim = simulate_heads(kf, df_obs["Distance"].values)
    return np.sqrt(np.mean((sim - df_obs["Observed Head"].values) ** 2))
 
# ─────────────────────────────────────────────
# AUTO-CALIBRATION (scipy)
# ─────────────────────────────────────────────
 
st.subheader("🎯 Auto-Calibration")
 
run_auto = st.button("▶ Run Auto-Calibration (minimise RMSE)", type="primary")
 
kf_best = kf_manual  # default to manual
 
if run_auto:
    result = minimize_scalar(
        rmse,
        bounds=(kf_min, kf_max),
        method="bounded"
    )
 
    if result.success:
        kf_best = result.x
        rmse_best = result.fun
        st.success(
            f"✅ Optimal Kf = **{kf_best:.4f} m/day** | RMSE = **{rmse_best:.4f} m**"
        )
        st.info(
            f"log₁₀(Kf) = {np.log10(kf_best):.3f} | "
            f"Kf falls in: {'gravel/coarse sand' if kf_best > 10 else 'medium sand' if kf_best > 1 else 'fine sand/silt'}"
        )
    else:
        st.error("Auto-calibration did not converge. Check your input parameters.")
        kf_best = kf_manual
else:
    kf_best = kf_manual
    st.info("Using manually selected Kf. Press **Run Auto-Calibration** to find the optimal value.")
 
# ─────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────
 
st.subheader("📋 Simulation Results")
 
sim_heads = simulate_heads(kf_best, df_obs["Distance"].values)
df_results = df_obs.copy()
df_results["Simulated Head (m)"] = np.round(sim_heads, 3)
df_results["Residual (m)"] = np.round(sim_heads - df_obs["Observed Head"].values, 3)
df_results["Abs Residual (m)"] = np.abs(df_results["Residual (m)"])
 
current_rmse = rmse(kf_best)
 
col1, col2, col3 = st.columns(3)
col1.metric("RMSE (m)", round(current_rmse, 4))
col2.metric("Max Abs Residual (m)", round(df_results["Abs Residual (m)"].max(), 4))
col3.metric("Kf Used (m/day)", round(kf_best, 4))
  
st.dataframe(df_results,use_container_width=True)
 
# ─────────────────────────────────────────────
# HEAD PROFILE PLOT (new)
# ─────────────────────────────────────────────
 
st.subheader("📈 Head Profile Along Aquifer")
 
x_cont = np.linspace(0, L, 500)
h_cont = simulate_heads(kf_best, x_cont)
 
fig_profile = go.Figure()
 
# Continuous simulated profile
fig_profile.add_trace(go.Scatter(
    x=x_cont,
    y=h_cont,
    mode="lines",
    name=f"Simulated (Kf={round(kf_best,3)} m/day)",
    line=dict(color="#1f77b4", width=2.5)
))
 
# Boundary heads
fig_profile.add_trace(go.Scatter(
    x=[0, L],
    y=[h1, h2],
    mode="markers",
    name="Boundary Conditions",
    marker=dict(symbol="diamond", size=12, color="green")
))
 
# Observed well heads
fig_profile.add_trace(go.Scatter(
    x=df_obs["Distance"],
    y=df_obs["Observed Head"],
    mode="markers",
    name="Observed Wells",
    marker=dict(symbol="circle", size=10, color="red",
                line=dict(width=1.5, color="darkred"))
))
 
fig_profile.update_layout(
    title="Groundwater Head Profile",
    xaxis_title="Distance from Upstream Boundary (m)",
    yaxis_title="Groundwater Head (m)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified"
)
 
st.plotly_chart(fig_profile, use_container_width=True)
 
# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────
 
st.subheader("🔍 Sensitivity Analysis — RMSE vs Kf")
 
kf_range = np.logspace(log_min, log_max, 200)
rmse_curve = [rmse(k) for k in kf_range]
 
fig_sens = px.line(
    x=kf_range,
    y=rmse_curve,
    log_x=True,
    labels={"x": "Kf (m/day)", "y": "RMSE (m)"},
    title="Calibration Error vs Hydraulic Conductivity"
)
 
# Mark best Kf on sensitivity plot
fig_sens.add_vline(
    x=kf_best,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Best Kf = {round(kf_best,3)}",
    annotation_position="top right"
)
 
st.plotly_chart(fig_sens, use_container_width=True)
 
# ─────────────────────────────────────────────
# WATER TABLE DIVIDE DETECTION (bonus)
# ─────────────────────────────────────────────
 
st.subheader("📐 Water Table Divide")
 
# The divide occurs where dh/dx = 0 — analytically:
# x_divide = L/2 + Kf(h1²-h2²)/(2RL)
# Only physically meaningful if 0 < x_divide < L
 
if R > 0:
    x_divide = L / 2 + kf_best * (h1 ** 2 - h2 ** 2) / (2 * R * L)
    if 0 < x_divide < L:
        h_divide = float(simulate_heads(kf_best, np.array([x_divide]))[0])
        st.success(
            f"A **groundwater divide** exists at x = **{x_divide:.1f} m** "
            f"(head = {h_divide:.2f} m). Flow is towards both boundaries from this point."
        )
    else:
        st.info(
            "No groundwater divide within the aquifer domain. "
            "Flow is unidirectional from upstream to downstream."
        )
else:
    st.info("Recharge R = 0: no divide possible (no recharge driving mound formation).")
 
# ─────────────────────────────────────────────
# GOVERNING EQUATION & VARIABLE DEFINITIONS
# ─────────────────────────────────────────────
 
st.markdown("---")
 
with st.expander("📖 Governing Equation & Variable Definitions", expanded=False):
 
    st.header("Governing Groundwater Equation")
 
    st.latex(r'''
    h(x)^2 =
    h_1^2 -
    \frac{(h_1^2 - h_2^2)}{L}\,x +
    \frac{R}{K_f}\,x(L-x)
    ''')
 
    st.markdown(
        """
        **Derivation basis**: Darcy's Law combined with the continuity equation under the
        Dupuit–Forchheimer assumption (vertical equipotentials). Valid for gentle water-table
        gradients (slope < ~0.1).
        """
    )
 
    definitions = {
        "Variable": ["h(x)", "h₁", "h₂", "R", "Kf", "x", "L"],
        "Description": [
            "Simulated groundwater head at distance x",
            "Fixed head at upstream boundary (x = 0)",
            "Fixed head at downstream boundary (x = L)",
            "Uniform vertical recharge to the aquifer",
            "Horizontal hydraulic conductivity",
            "Distance from upstream boundary",
            "Total length between boundaries",
        ],
        "Units": ["m", "m", "m", "m/day", "m/day", "m", "m"],
        "Typical Range": [
            "—", "—", "—",
            "0.00001–0.005 m/day",
            "0.0001–1000 m/day",
            "0 – L", "—",
        ],
    }
 
    st.table(pd.DataFrame(definitions))
 
    st.markdown(
        """
        **Conceptual assumptions:**
        - Steady-state conditions (no storage change)  
        - Homogeneous, isotropic aquifer  
        - Uniform recharge over the full domain  
        - 1D flow along the hydraulic gradient  
        - Unconfined aquifer (Dupuit–Forchheimer; use linear-h form for confined)
 
        This simplified model estimates realistic Kf values before building full numerical models (e.g., MODFLOW).
        """
    )
