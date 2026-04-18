import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px;
}

/* Inputs */
.stSlider > div > div { background: #21262d; }
div[data-baseweb="select"] > div {
    background-color: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
input[type="number"] {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border-color: #30363d !important;
}

/* Predict button */
div.stButton > button {
    background: linear-gradient(135deg, #da3633, #b91c1c);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 32px;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
    font-family: 'DM Sans', sans-serif;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #b91c1c, #991b1b);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(218,54,51,0.4);
}

/* Cards */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 12px;
}

/* Risk meter */
.risk-high {
    background: linear-gradient(135deg, #1a0a0a, #2d1010);
    border: 1px solid #da3633;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #0a1a0a, #102d10);
    border: 1px solid #2ea043;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.risk-score-high {
    font-size: 64px;
    font-weight: 600;
    color: #f85149;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}
.risk-score-low {
    font-size: 64px;
    font-weight: 600;
    color: #3fb950;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}
.risk-label {
    font-size: 13px;
    color: #8b949e;
    margin-top: 8px;
    letter-spacing: 0.5px;
}

/* Section divider */
.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 24px 0;
}

/* Factor tag */
.factor-tag {
    display: inline-block;
    background: #21262d;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: #c9d1d9;
    margin: 3px;
    font-family: 'DM Mono', monospace;
}

/* Header bar */
.top-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid #21262d;
}
.header-icon { font-size: 28px; }
.header-title {
    font-size: 22px;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.3px;
}
.header-sub {
    font-size: 13px;
    color: #8b949e;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    cols  = joblib.load("columns.pkl")
    return model, cols

model, cols = load_model()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫀 CardioScan AI")
    st.markdown("<p style='font-size:12px;color:#8b949e;margin-top:-8px'>Heart Disease Risk Assessment</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigation", ["🔬 Risk Assessment", "📊 Data Insights", "ℹ️ About"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<p style='font-size:11px;color:#8b949e'>⚠️ For educational purposes only. Not a substitute for medical advice.</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Risk Assessment
# ══════════════════════════════════════════════════════════════════════
if page == "🔬 Risk Assessment":

    st.markdown("""
    <div class="top-header">
        <span class="header-icon">🫀</span>
        <div>
            <div class="header-title">CardioScan AI</div>
            <div class="header-sub">Enter patient vitals to assess cardiovascular risk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown('<div class="card"><div class="card-title">Patient Demographics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 20, 80, 45)
            bp  = st.number_input("Resting BP (mmHg)", 80, 200, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        with c2:
            sex    = st.selectbox("Biological Sex", ["Male", "Female"])
            maxhr  = st.slider("Max Heart Rate", 60, 220, 150)
            fbs    = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">Clinical Indicators</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            cp      = st.selectbox("Chest Pain Type", ["ATA — Atypical Angina", "NAP — Non-Anginal", "ASY — Asymptomatic", "TA — Typical Angina"])
            ecg     = st.selectbox("Resting ECG", ["Normal", "ST Abnormality", "LVH"])
        with c4:
            angina  = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            slope   = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("⚡  Analyze Risk")

    # ── Result panel ───────────────────────────────────────────────────
    with col_right:

        if predict_btn:
            cp_code     = cp.split(" — ")[0]
            ecg_code    = {"Normal": "Normal", "ST Abnormality": "ST", "LVH": "LVH"}[ecg]
            sex_code    = "M" if sex == "Male" else "F"
            angina_code = "Y" if angina == "Yes" else "N"
            fbs_val     = 1 if fbs == "Yes" else 0

            input_dict = {
                "Age": age, "RestingBP": bp, "Cholesterol": chol,
                "FastingBS": fbs_val, "MaxHR": maxhr, "Oldpeak": oldpeak,
                "Sex_F": sex_code == "F", "Sex_M": sex_code == "M",
                "ChestPainType_ATA": cp_code == "ATA",
                "ChestPainType_NAP": cp_code == "NAP",
                "ChestPainType_ASY": cp_code == "ASY",
                "ChestPainType_TA":  cp_code == "TA",
                "RestingECG_Normal": ecg_code == "Normal",
                "RestingECG_ST":     ecg_code == "ST",
                "RestingECG_LVH":    ecg_code == "LVH",
                "ExerciseAngina_N":  angina_code == "N",
                "ExerciseAngina_Y":  angina_code == "Y",
                "ST_Slope_Up":   slope == "Up",
                "ST_Slope_Flat": slope == "Flat",
                "ST_Slope_Down": slope == "Down",
            }

            df_input = pd.DataFrame([input_dict]).reindex(columns=cols, fill_value=0)
            result   = model.predict(df_input)[0]
            prob     = model.predict_proba(df_input)[0][1]
            pct      = int(prob * 100)

            # Risk score card
            if result == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <div class="risk-score-high">{pct}%</div>
                    <div style="font-size:18px;font-weight:600;color:#f85149;margin-top:10px">⚠ High Risk Detected</div>
                    <div class="risk-label">Cardiovascular disease probability</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div class="risk-score-low">{pct}%</div>
                    <div style="font-size:18px;font-weight:600;color:#3fb950;margin-top:10px">✓ Low Risk</div>
                    <div class="risk-label">Cardiovascular disease probability</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # Risk breakdown bar
            st.markdown('<div class="card"><div class="card-title">Risk Breakdown</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 0.6))
            fig.patch.set_facecolor('#161b22')
            ax.set_facecolor('#161b22')
            ax.barh([""], [pct],     color="#f85149" if result == 1 else "#3fb950", height=0.5)
            ax.barh([""], [100-pct], left=[pct], color="#21262d", height=0.5)
            ax.set_xlim(0, 100)
            ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

            # Patient summary
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Patient Summary</div>
                <div style="display:flex;flex-wrap:wrap;gap:4px">
                    <span class="factor-tag">Age {age}</span>
                    <span class="factor-tag">{sex}</span>
                    <span class="factor-tag">BP {bp}</span>
                    <span class="factor-tag">Chol {chol}</span>
                    <span class="factor-tag">MaxHR {maxhr}</span>
                    <span class="factor-tag">Oldpeak {oldpeak}</span>
                    <span class="factor-tag">{cp_code} pain</span>
                    <span class="factor-tag">ECG {ecg_code}</span>
                    <span class="factor-tag">Angina {angina}</span>
                    <span class="factor-tag">Slope {slope}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature importance mini chart
            if hasattr(model, "feature_importances_"):
                st.markdown('<div class="card"><div class="card-title">Top Risk Factors (Model)</div>', unsafe_allow_html=True)
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[-6:]
                top_feats = [cols[i] for i in top_idx]
                top_vals  = [importances[i] for i in top_idx]

                fig2, ax2 = plt.subplots(figsize=(5, 2.5))
                fig2.patch.set_facecolor('#161b22')
                ax2.set_facecolor('#161b22')
                bars = ax2.barh(top_feats, top_vals, color="#da3633", alpha=0.85)
                ax2.tick_params(colors='#8b949e', labelsize=9)
                for spine in ax2.spines.values():
                    spine.set_visible(False)
                ax2.xaxis.set_visible(False)
                for bar, val in zip(bars, top_vals):
                    ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                             f"{val:.3f}", va='center', color='#8b949e', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:48px 24px">
                <div style="font-size:48px;margin-bottom:16px">🫀</div>
                <div style="font-size:16px;font-weight:500;color:#e6edf3">Ready to Analyze</div>
                <div style="font-size:13px;color:#8b949e;margin-top:8px">Fill in patient details and click<br>Analyze Risk to see results</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Insights
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 Data Insights":

    st.markdown("""
    <div class="top-header">
        <span class="header-icon">📊</span>
        <div>
            <div class="header-title">Data Insights</div>
            <div class="header-sub">Explore the heart disease dataset</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = pd.read_csv("heart.csv")

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Patients", df.shape[0])
        k2.metric("Features", df.shape[1] - 1)
        k3.metric("At Risk", int(df["HeartDisease"].sum()))
        k4.metric("Risk Rate", f"{df['HeartDisease'].mean():.1%}")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="card"><div class="card-title">Disease Distribution</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor('#161b22')
            ax.set_facecolor('#161b22')
            counts = df["HeartDisease"].value_counts()
            ax.pie(counts, labels=["At Risk", "Healthy"],
                   colors=["#da3633", "#2ea043"],
                   autopct="%1.1f%%", textprops={"color": "#c9d1d9", "fontsize": 11},
                   wedgeprops={"linewidth": 2, "edgecolor": "#161b22"})
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card"><div class="card-title">Age Distribution by Risk</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor('#161b22')
            ax.set_facecolor('#161b22')
            ax.hist(df[df["HeartDisease"]==0]["Age"], bins=20, color="#2ea043", alpha=0.7, label="Healthy")
            ax.hist(df[df["HeartDisease"]==1]["Age"], bins=20, color="#da3633", alpha=0.7, label="At Risk")
            ax.legend(fontsize=9, facecolor="#21262d", labelcolor="#c9d1d9", edgecolor="#30363d")
            ax.tick_params(colors="#8b949e", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#21262d")
            ax.set_xlabel("Age", color="#8b949e", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # Correlation heatmap
        st.markdown('<div class="card"><div class="card-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    ax=ax, linewidths=0.5, linecolor="#0d1117",
                    annot_kws={"size": 8},
                    cbar_kws={"shrink": 0.8})
        ax.tick_params(colors="#8b949e", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning("heart.csv not found in project folder.")

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — About
# ══════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown("""
    <div class="top-header">
        <span class="header-icon">ℹ️</span>
        <div>
            <div class="header-title">About CardioScan AI</div>
            <div class="header-sub">Project details and model information</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Project Overview</div>
        <p style="color:#c9d1d9;line-height:1.7;font-size:14px">
        CardioScan AI is an end-to-end machine learning web application that predicts
        the risk of cardiovascular disease based on clinical patient data.
        Built using Python, Scikit-learn, and Streamlit.
        </p>
    </div>

    <div class="card">
        <div class="card-title">Tech Stack</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:4px">
            <span class="factor-tag">Python 3.12</span>
            <span class="factor-tag">Scikit-learn</span>
            <span class="factor-tag">Pandas</span>
            <span class="factor-tag">NumPy</span>
            <span class="factor-tag">Streamlit</span>
            <span class="factor-tag">Matplotlib</span>
            <span class="factor-tag">Seaborn</span>
            <span class="factor-tag">Joblib</span>
        </div>
    </div>

    <div class="card">
        <div class="card-title">Dataset</div>
        <p style="color:#c9d1d9;line-height:1.7;font-size:14px">
        UCI Heart Failure Prediction Dataset — 918 patient records,
        11 clinical features. Source: Kaggle.
        </p>
    </div>

    <div class="card">
        <div class="card-title">Disclaimer</div>
        <p style="color:#8b949e;line-height:1.7;font-size:13px">
        This application is built for educational and portfolio purposes only.
        It is not intended to replace professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider.
        </p>
    </div>
    """, unsafe_allow_html=True)
