import streamlit as st
from typing import Optional, List, Dict, Any
from pathlib import Path

# ====== PAGE SETUP ======
st.set_page_config(
    page_title="Trajectory Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp { background-color: white; color: black; }
    .css-1d391kg { background-color: #f8f9fa; } /* sidebar (может зависеть от версии) */
    </style>
    """,
    unsafe_allow_html=True
)

# ====== HELPERS ======
def parse_list_input(value: str) -> Optional[List[float]]:
    """ 'a,b,c' -> [a,b,c] ; '' -> None """
    value = value.strip()
    if not value:
        return None
    try:
        return [float(item.strip()) for item in value.split(",")]
    except ValueError:
        st.warning("Невозможно преобразовать список к числам. Используйте запятую.")
        return None

# ====== SIDEBAR UI ======
st.title("Trajectory Simulation and Kalman Filtering")

# -------------------- Data Source --------------------
st.sidebar.header("Data Source")
use_table = st.sidebar.checkbox("Use real data table", value=False)
simulate_mode = "real" if use_table else "synthetic"

if use_table:
    # пример: подстрой под свои таблицы
    # from config import tables
    tables = ["table_1.csv", "table_2.csv", "table_3.csv"]  # placeholder
    table_idx = st.sidebar.selectbox(
        "Select table",
        list(range(len(tables))),
        format_func=lambda i: tables[i],
        index=0
    )
else:
    table_idx = 0

# -------------------- Motion Settings --------------------
with st.sidebar.expander("Motion Settings", expanded=not use_table):
    # (ничего не меняем по составу полей, как просил)
    dimension = st.selectbox("Dimension", [1, 2], index=0)
    trajectory_type = st.selectbox("Trajectory type", ["line", "parabola", "mixed"], index=0)
    velocity = st.number_input("Velocity (m/s)", value=30.0, min_value=0.0, step=1.0)
    angle_deg = st.number_input("Angle (degrees, for 2D)", value=45.0)
    experiment_time = st.number_input("Experiment time (s)", value=100.0, min_value=1.0)

# -------------------- Measurement Settings (+ Outliers) --------------------
with st.sidebar.expander("Measurement Settings", expanded=not use_table):
    dt = st.number_input("Mean dt (s)", value=0.2, min_value=0.0)
    dt_var = st.number_input("Relative dt variation", value=0.2, min_value=0.0)

    measure_stds_str = st.text_input("Measurement stds (comma-separated or single)", "30")
    stds_changepoints_str = st.text_input("Stds changepoints (sec, comma-separated)", "")
    measurements_err = st.multiselect(
        "Measurement error type(s)",
        ["norm", "uniform", "laplace"],
        default=["norm"]
    )

    measure_stds = parse_list_input(measure_stds_str)
    if measure_stds is None:
        try:
            measure_stds = float(measure_stds_str)
        except ValueError:
            st.warning("Measurement stds: укажи число или список через запятую. Использую 30.")
            measure_stds = 30.0
    stds_changepoints = parse_list_input(stds_changepoints_str)

    st.markdown("---")
    st.caption("Outlier Generation (Simulation)")
    add_outliers = st.checkbox("Add outliers?", value=False)
    if add_outliers:
        outlier_types = st.multiselect(
            "Outlier types",
            ["single", "shift", "series"],
            default=[]
        )
        if not outlier_types:
            outlier_types = ["none"]
        outlier_series_length = st.number_input("Outlier series length", value=20, min_value=1, step=1)
        outlier_scale = st.number_input("Outlier scale", value=1000.0, min_value=0.0)
    else:
        outlier_types = ["none"]
        outlier_series_length = 1
        outlier_scale = 0.0

# -------------------- Kalman Filter (Base) --------------------
with st.sidebar.expander("Kalman Filter (Base)", expanded=True):
    p_init = st.number_input("Initial P", value=100.0, min_value=0.0)
    q_init = st.number_input("Initial Q std", value=1.0, min_value=0.0)
    r_init = st.number_input("Initial R std", value=30.0, min_value=0.0)

# -------------------- Adaptation (Q/R) --------------------
with st.sidebar.expander("Adaptation (Q/R)", expanded=False):
    # Можно выбрать несколько адаптаций
    adaptation_choices = st.multiselect(
        "Select adaptation methods",
        [
            "None",
            "IAE",
            "Sage–Husa",
            "EWMA",
            "EM (batch)",
            "MMAE (multi-model)"
        ],
        default=["None"]
    )

    # Общие «перила» для Q/R делают теперь в Constraints → QRBounder,
    # но иногда удобно продублировать быстрые настройки здесь:
    st.caption("(Рекомендуется задать жёсткие пределы Q/R в блоке Constraints → QR bounds)")

    adaptation_params: Dict[str, Any] = {}

    if "IAE" in adaptation_choices:
        st.markdown("**IAE settings**")
        iae_mode = st.radio("IAE mode", ["r", "q", "both"], horizontal=True, key="iae_mode")
        iae_alpha = st.number_input("IAE alpha", value=0.05, min_value=0.0, max_value=1.0, key="iae_alpha")
        adaptation_params["IAE"] = {"mode": iae_mode, "alpha": iae_alpha}

    if "Sage–Husa" in adaptation_choices:
        st.markdown("**Sage–Husa settings**")
        sh_mode = st.radio("Sage–Husa mode", ["r", "q", "both"], horizontal=True, key="sh_mode")
        sh_lambda = st.number_input("Forgetting λ", value=0.05, min_value=0.0, max_value=1.0, key="sh_lambda")
        adaptation_params["Sage–Husa"] = {"mode": sh_mode, "lam": sh_lambda}

    if "EWMA" in adaptation_choices:
        st.markdown("**EWMA settings**")
        ewma_mode = st.radio("EWMA mode", ["r", "q", "both"], horizontal=True, key="ewma_mode")
        ewma_alpha = st.number_input("EWMA alpha", value=0.05, min_value=0.0, max_value=1.0, key="ewma_alpha")
        ewma_gate = st.checkbox("Ignore outliers for R (chi-square gate)?", value=True, key="ewma_gate_on")
        ewma_gate_thr = st.number_input("Gate threshold (χ²)", value=6.0, min_value=0.0, key="ewma_gate_thr")
        accel_idx_str = st.text_input("Q on subset idx (e.g., '2,3') or empty", "", key="ewma_accel_idx")
        accel_idx = parse_list_input(accel_idx_str)
        accel_idx = [int(v) for v in accel_idx] if accel_idx else None
        adaptation_params["EWMA"] = {
            "mode": ewma_mode, "alpha": ewma_alpha,
            "chi2_gate": (ewma_gate_thr if ewma_gate else None),
            "accel_idx": accel_idx
        }

    if "EM (batch)" in adaptation_choices:
        st.markdown("**EM (batch) settings**")
        em_window = st.number_input("EM window size (T)", value=100, min_value=2, step=1, key="em_window")
        em_iters = st.number_input("EM iterations", value=10, min_value=1, step=1, key="em_iters")
        em_period = st.number_input("EM run period (steps)", value=100, min_value=1, step=1, key="em_period")
        em_blend = st.slider("Blend new Q/R with current (0..1)", 0.0, 1.0, 0.5, key="em_blend")
        adaptation_params["EM"] = {
            "window": int(em_window),
            "n_iter": int(em_iters),
            "period": int(em_period),
            "blend": float(em_blend),
        }

    if "MMAE (multi-model)" in adaptation_choices:
        st.markdown("**MMAE settings**")
        mmae_select = st.radio("Parameter selection", ["argmax", "weighted"], horizontal=True, key="mmae_select")
        mmae_models = st.number_input("Number of profiles", value=2, min_value=2, max_value=6, step=1, key="mmae_models")
        mmae_forgetting = st.slider("Forgetting (0..1)", 0.0, 1.0, 0.1, key="mmae_forgetting")
        profiles = []
        for i in range(int(mmae_models)):
            st.caption(f"Profile #{i+1}")
            q_scale = st.number_input(f"Q scale (profile {i+1})", value=1.0, min_value=0.0, key=f"mmae_q_{i}")
            r_scale = st.number_input(f"R scale (profile {i+1})", value=1.0, min_value=0.0, key=f"mmae_r_{i}")
            profiles.append({"Q_scale": q_scale, "R_scale": r_scale})
        adaptation_params["MMAE"] = {
            "select": mmae_select,
            "forgetting": float(mmae_forgetting),
            "profiles": profiles
        }

# -------------------- Robust --------------------
with st.sidebar.expander("Robust (Outlier Handling in Filter)", expanded=False):
    robust_choices = st.multiselect(
        "Select robust methods (applied as a chain)",
        [
            "None",
            "Chi-Square Gate",
            "Huber",
            "Saturation",
            "Student-t"
        ],
        default=["None"]
    )
    robust_params: Dict[str, Any] = {}

    if "Chi-Square Gate" in robust_choices:
        gate_thr = st.number_input("χ² threshold", value=6.0, min_value=0.0, key="gate_thr")
        robust_params["Chi-Square Gate"] = {"threshold": gate_thr}

    if "Huber" in robust_choices:
        huber_c = st.number_input("Huber c", value=2.5, min_value=0.0, key="huber_c")
        robust_params["Huber"] = {"c": huber_c}

    if "Saturation" in robust_choices:
        sat_c = st.number_input("Saturation c (Mahalanobis)", value=4.0, min_value=0.0, key="sat_c")
        robust_params["Saturation"] = {"c": sat_c}

    if "Student-t" in robust_choices:
        dof = st.number_input("Degrees of freedom ν", value=5.0, min_value=0.1, key="stud_t_dof")
        robust_params["Student-t"] = {"dof": dof}

# -------------------- Constraints --------------------
with st.sidebar.expander("Constraints", expanded=False):
    # QR bounds
    use_qr_bounds = st.checkbox("Enable Q/R bounds", value=True, key="qr_on")
    q_min = st.number_input("Q min", value=1e-6, min_value=0.0, key="qr_qmin")
    q_max = st.number_input("Q max", value=1e3, min_value=0.0, key="qr_qmax")
    r_min = st.number_input("R min", value=1e-6, min_value=0.0, key="qr_rmin")
    r_max = st.number_input("R max", value=1e3, min_value=0.0, key="qr_rmax")

    # Equality A x = b
    use_eq = st.checkbox("Equality constraint (A x = b)", value=False, key="eq_on")
    eq_k = st.number_input("k (rows of A)", value=1, min_value=1, step=1, key="eq_k")
    eq_A_str = st.text_input("A (row-wise, comma-separated; ';' rows)", "0,1", key="eq_A")
    eq_b_str = st.text_input("b (comma-separated)", "1.0", key="eq_b")

    # Box lb ≤ x ≤ ub
    use_box = st.checkbox("Box constraint (lb ≤ x ≤ ub)", value=False, key="box_on")
    box_lb_str = st.text_input("lb (comma-separated)", "-1e9,-1e9", key="box_lb")
    box_ub_str = st.text_input("ub (comma-separated)", "1e9,1e9", key="box_ub")

    # Norm ||x[idx]|| ≤ r
    use_norm = st.checkbox("Norm-ball constraint (||x[idx]|| ≤ r)", value=False, key="norm_on")
    norm_idx_str = st.text_input("idx (comma-separated indices)", "0,1", key="norm_idx")
    norm_r = st.number_input("radius r", value=100.0, min_value=0.0, key="norm_r")

# -------------------- Smoothing --------------------
with st.sidebar.expander("Smoothing", expanded=False):
    lag = st.number_input("RTS lag", value=5, min_value=0, step=1)
    use_fixed_point = st.checkbox("Fixed-Point smoothing?", value=False)
    fp_t_idx = st.number_input("Fixed-Point index (t)", value=0, min_value=0, step=1)

# ====== RUN BUTTON ======
if st.sidebar.button("Run Simulation"):
    # Собираем «конфиг» (словарь). Здесь можно заменить на твой Config(...)
    cfg: Dict[str, Any] = dict(
        # source
        use_table=use_table,
        table_idx=table_idx,
        simulate_mode=simulate_mode,
        # motion
        dimension=dimension,
        trajectory_type=trajectory_type,
        velocity=velocity,
        angle_deg=angle_deg,
        experiment_time=experiment_time,
        # measurement (+ outliers)
        dt=dt,
        dt_var=dt_var,
        measure_stds=measure_stds,
        stds_changepoints=stds_changepoints,
        measurements_types_err=measurements_err if len(measurements_err) > 1 else (measurements_err[0] if measurements_err else "norm"),
        add_outliers=add_outliers,
        outlier_types=outlier_types,
        outlier_series_length=outlier_series_length,
        outlier_scale=outlier_scale,
        # base filter
        p_init=p_init,
        q_init=q_init,
        r_init=r_init,
        # adaptation / robust / constraints / smoothing
        adaptation_selected=adaptation_choices,
        adaptation_params=adaptation_params,
        robust_selected=robust_choices,
        robust_params=robust_params,
        constraints=dict(
            qr_bounds=dict(enabled=use_qr_bounds, q_min=q_min, q_max=q_max, r_min=r_min, r_max=r_max),
            equality=dict(enabled=use_eq, A=eq_A_str, b=eq_b_str, k=int(eq_k)),
            box=dict(enabled=use_box, lb=box_lb_str, ub=box_ub_str),
            norm=dict(enabled=use_norm, idx=norm_idx_str, r=norm_r),
        ),
        smoothing=dict(rts_lag=int(lag), fixed_point=use_fixed_point, fp_t_idx=int(fp_t_idx)),
    )

    # ====== PLACEHOLDER PIPELINE ======
    st.subheader("Run preview (UI wiring only)")
    st.json(cfg)

    # Здесь позже:
    # times, true, obs = generate...
    # kf = build_pluggable_filter(cfg)  # создаёшь backend + robust/adaptation/constraints
    # filtered, P, Fs, Qs, ... = run_kf(...)
    # smoothed = run_smoothing(...)
    # viz...

else:
    st.info("Настрой параметры слева и нажми **Run Simulation**.")
