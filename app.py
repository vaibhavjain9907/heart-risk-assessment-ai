import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Risk Assessment",
    page_icon="",
    layout="wide",
)

def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("heart_pipeline.pkl")

pipeline = load_pipeline()
# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


from utils import compute_risk, sub_scores
# ── Header ────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown('<div class="page-title"> Heart Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Clinical risk stratification · blended AI + medical scoring</div>', unsafe_allow_html=True)
with hcol2:
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer-badge">⚕ Not a diagnosis — consult a physician</div>', unsafe_allow_html=True)

st.divider()

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

# ── LEFT: Inputs ──────────────────────────────────────────────────────────────
with left:
    st.markdown("### Patient Details")

    with st.expander("👤  Demographics & Lifestyle", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1: age    = st.number_input("Age", 1, 100, 50)
        with c2: gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", horizontal=True)
        with c3: bmi    = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)

        c4, c5, c6 = st.columns(3)
        with c4: stress = st.slider("Stress Level", 0, 10, 5)
        with c5: sleep  = st.slider("Sleep Hours",  0, 12, 7)
        with c6: sugar  = st.slider("Sugar Intake",  0, 10, 5)

        c7, c8 = st.columns(2)
        with c7: exercise = st.toggle("Active Exercise Habits", value=False)
        with c8: smoking  = st.toggle("Smoker", value=False)

    with st.expander("  Vitals & Lab Values", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            bp           = st.number_input("Blood Pressure (mmHg)", 50, 300, 120)
            chol         = st.number_input("Cholesterol (mg/dL)", 100, 500, 200)
            triglyceride = st.number_input("Triglycerides (mg/dL)", 50, 600, 150)
        with c2:
            fasting      = st.number_input("Fasting Blood Sugar (mg/dL)", 50, 300, 100)
            crp          = st.number_input("CRP Level (mg/L)", 0.0, 20.0, 2.0, step=0.1)
            homocysteine = st.number_input("Homocysteine (µmol/L)", 0.0, 50.0, 10.0, step=0.1)

        st.markdown('<div class="section-label">Lipid Flags</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: high_bp  = st.toggle("High Blood Pressure", value=False)
        with c2: low_hdl  = st.toggle("Low HDL Cholesterol", value=False)
        with c3: high_ldl = st.toggle("High LDL Cholesterol", value=False)

    with st.expander("  Medical History", expanded=True):
        c1, c2 = st.columns(2)
        with c1: family   = st.toggle("Family History of Heart Disease", value=False)
        with c2: diabetes = st.toggle("Diabetes", value=False)

    patient_name = st.text_input("Patient Name / ID (for history)", placeholder="e.g. Patient 001")
    run = st.button("  Run Assessment", use_container_width=True, type="primary")

# ── RIGHT: Results ────────────────────────────────────────────────────────────
with right:
    st.markdown("### Results")

    if run:
        sample_df = pd.DataFrame([{
            "Age": age, "Gender": int(gender),
            "Blood Pressure": bp, "Cholesterol Level": chol,
            "Exercise Habits": int(exercise), "Smoking": int(smoking),
            "Family Heart Disease": int(family), "Diabetes": int(diabetes),
            "BMI": bmi, "High Blood Pressure": int(high_bp),
            "Low HDL Cholesterol": int(low_hdl), "High LDL Cholesterol": int(high_ldl),
            "Stress Level": stress, "Sleep Hours": sleep, "Sugar Consumption": sugar,
            "Triglyceride Level": triglyceride, "Fasting Blood Sugar": fasting,
            "CRP Level": crp, "Homocysteine Level": homocysteine,
        }])

        model_prob = pipeline.predict_proba(sample_df)[0][1] * 100
        probability, med_score, risk_flags, health_score = compute_risk(
            model_prob, bp, chol, triglyceride, fasting, crp, homocysteine,
            bmi, smoking, diabetes, family, high_bp, low_hdl, high_ldl,
            age, exercise, sleep
        )

        # Tier
        if probability >= 75 or risk_flags >= 5:
            tier, rcss, hcls = "danger", "result-danger", "fast"
            label, sub       = "High Risk", "Multiple critical risk factors. Consult a cardiologist urgently."
            tier_color       = "#E24B4A"
        elif probability >= 40 or risk_flags >= 3:
            tier, rcss, hcls = "risk", "result-risk", "medium"
            label, sub       = "Elevated Risk", "Moderate factors. Lifestyle changes and medical review recommended."
            tier_color       = "#BA7517"
        else:
            tier, rcss, hcls = "healthy", "result-healthy", ""
            label, sub       = "Low Risk", "No significant risk factors. Keep up your healthy habits."
            tier_color       = "#639922"

        txt_cls = f"{tier}-text"

        # ── Result card + animated heart ─────────────────────────────────────
        st.markdown(f"""
        <div class="result-card {rcss}">
          <span class="heart-icon {hcls}"></span>
          <div>
            <div class="result-pct {txt_cls}">{probability:.1f}%</div>
            <div class="result-label {txt_cls}">{label}</div>
            <div class="result-sub">{sub}</div>
            <div class="result-sub" style="margin-top:5px;opacity:0.6;font-size:0.63rem">
              Model: {model_prob:.0f}% &nbsp;·&nbsp; Med score: {med_score}/100 &nbsp;·&nbsp; Flags: {risk_flags}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauge – probability meter ─────────────────────────────────────────
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability, 1),
            number={"suffix": "%", "font": {"size": 26, "family": "DM Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
                "bar":  {"color": tier_color, "thickness": 0.26},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "#EAF3DE"},
                    {"range": [40, 75], "color": "#FAEEDA"},
                    {"range": [75,100], "color": "#FCEBEB"},
                ],
                "threshold": {
                    "line": {"color": tier_color, "width": 3},
                    "thickness": 0.8,
                    "value": probability,
                },
            },
            title={"text": "Probability Meter", "font": {"size": 11, "color": "#888"}},
        ))
        gauge.update_layout(
            height=210, margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#333",
        )
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        # ── Health score donut + risk bars side by side ───────────────────────
        dc, bc = st.columns(2)

        with dc:
            score_color = "#639922" if health_score >= 60 else "#BA7517" if health_score >= 35 else "#E24B4A"
            donut = go.Figure(go.Pie(
                values=[health_score, 100 - health_score],
                hole=0.72,
                marker_colors=[score_color, "#f0ede8"],
                textinfo="none", hoverinfo="skip", showlegend=False,
            ))
            donut.add_annotation(
                text=f"<b>{health_score}</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=28, color=score_color, family="Fraunces"),
            )
            donut.update_layout(
                height=180, margin=dict(t=30, b=0, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Health Score /100", font=dict(size=11, color="#888"), x=0.5),
            )
            st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

        with bc:
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            subs = sub_scores(bp, chol, bmi, stress, sleep, sugar,
                              smoking, diabetes, family, crp, fasting, exercise)
            bar_colors = {
                "Cardiac Risk":   "#E24B4A",
                "Metabolic Risk": "#BA7517",
                "Lifestyle Risk": "#534AB7",
                "Inflammation":   "#D4537E",
            }
            bars_html = ""
            for name, val in subs.items():
                c = bar_colors[name]
                bars_html += f"""
                <div class="risk-bar-row">
                  <div class="risk-bar-label"><span>{name}</span><span>{val}%</span></div>
                  <div class="risk-bar-track">
                    <div class="risk-bar-fill" style="width:{val}%;background:{c}"></div>
                  </div>
                </div>"""
            st.markdown(f'<div style="padding-top:0.25rem">{bars_html}</div>', unsafe_allow_html=True)

        # ── Factor pills ──────────────────────────────────────────────────────
        pills = []
        if smoking:         pills.append((" Smoking",       "bad"))
        if family:          pills.append((" Family Hx",     "bad"))
        if diabetes:        pills.append((" Diabetes",      "bad"))
        if bp > 140:        pills.append((" High BP",       "bad"))
        if chol > 240:      pills.append((" High Chol",    "bad"))
        if bmi > 30:        pills.append((" High BMI",     "bad"))
        if high_bp:         pills.append((" Hypertension", "bad"))
        if low_hdl:         pills.append((" HDL",           "bad"))
        if high_ldl:        pills.append((" LDL",           "bad"))
        if crp > 3:         pills.append((" High CRP",     "bad"))
        if fasting > 126:   pills.append((" High FBS",     "bad"))
        if exercise:        pills.append((" Active",        "good"))
        if 7 <= sleep <= 8: pills.append((" Good Sleep",   "good"))

        if pills:
            ph = "".join(f'<span class="pill pill-{k}">{l}</span>' for l, k in pills)
            st.markdown(f'<div class="pills-wrap">{ph}</div>', unsafe_allow_html=True)

        # ── Save to history ───────────────────────────────────────────────────
        st.session_state.history.append({
            "name":        patient_name or f"Patient {len(st.session_state.history) + 1}",
            "time":        datetime.now().strftime("%H:%M · %d %b"),
            "probability": round(probability, 1),
            "health":      health_score,
            "tier":        tier,
            "flags":       risk_flags,
            "age":         int(age),
            "bp":          int(bp),
            "chol":        int(chol),
        })

    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 1rem;color:#bbb">
          <div style="font-size:3.5rem;margin-bottom:1rem"></div>
          <div style="font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase">
            Fill in patient details and run assessment
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Prediction History ────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    hd1, hd2 = st.columns([2, 1])
    with hd1:
        st.markdown("### Prediction History")
    with hd2:
        if st.button(" Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    hist_left, hist_right = st.columns([1, 1.4], gap="large")
    tier_colors = {"healthy": "#639922", "risk": "#BA7517", "danger": "#E24B4A"}

    with hist_left:
        for rec in reversed(st.session_state.history[-10:]):
            c = tier_colors[rec["tier"]]
            st.markdown(f"""
            <div class="hist-row">
              <div class="hist-dot" style="background:{c}"></div>
              <div>
                <div style="font-size:0.8rem;font-weight:500">{rec['name']}</div>
                <div class="hist-meta">Age {rec['age']} &nbsp;·&nbsp; BP {rec['bp']} &nbsp;·&nbsp; Chol {rec['chol']}</div>
                <div class="hist-meta">{rec['time']} &nbsp;·&nbsp; {rec['flags']} flags</div>
              </div>
              <div class="hist-pct" style="color:{c}">{rec['probability']}%</div>
            </div>
            """, unsafe_allow_html=True)

    with hist_right:
        df_h = pd.DataFrame(st.session_state.history)
        df_h["index"] = range(1, len(df_h) + 1)

        if len(df_h) >= 2:
            dot_colors = [tier_colors[t] for t in df_h["tier"]]
            trend = go.Figure()
            trend.add_hrect(y0=0,   y1=40,  fillcolor="#EAF3DE", opacity=0.3, line_width=0)
            trend.add_hrect(y0=40,  y1=75,  fillcolor="#FAEEDA", opacity=0.3, line_width=0)
            trend.add_hrect(y0=75,  y1=100, fillcolor="#FCEBEB", opacity=0.3, line_width=0)
            trend.add_trace(go.Scatter(
                x=df_h["index"], y=df_h["probability"],
                mode="lines+markers",
                line=dict(color="#bbb", width=1.5, dash="dot"),
                marker=dict(color=dot_colors, size=10, line=dict(color="white", width=2)),
                text=df_h["name"],
                hovertemplate="%{text}<br>Risk: %{y:.1f}%<extra></extra>",
            ))
            trend.update_layout(
                height=220,
                margin=dict(t=35, b=30, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Risk Trend Across Assessments", font=dict(size=11, color="#888"), x=0),
                xaxis=dict(title="Assessment #", showgrid=False, tickvals=df_h["index"].tolist()),
                yaxis=dict(title="Risk %", range=[0, 100], showgrid=True,
                           gridcolor="#eee", gridwidth=0.5),
                showlegend=False,
            )
            st.plotly_chart(trend, use_container_width=True, config={"displayModeBar": False})

        # Summary stats
        mc1, mc2, mc3, mc4 = st.columns(4)
        stats = [
            (f"{df_h['probability'].mean():.1f}%", "Avg Risk"),
            (f"{df_h['probability'].max():.1f}%",  "Peak Risk"),
            (f"{df_h['probability'].min():.1f}%",  "Best Risk"),
            (f"{df_h['health'].mean():.0f}",        "Avg Health"),
        ]
        for col, (val, lbl) in zip([mc1, mc2, mc3, mc4], stats):
            with col:
                st.markdown(f"""
                <div class="metric-mini">
                  <div class="val">{val}</div>
                  <div class="lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        # Distribution bar chart
        if len(df_h) >= 3:
            tier_counts = df_h["tier"].value_counts().reindex(["healthy","risk","danger"], fill_value=0)
            bar_fig = go.Figure(go.Bar(
                x=["Low Risk", "Elevated Risk", "High Risk"],
                y=tier_counts.values,
                marker_color=["#639922", "#BA7517", "#E24B4A"],
                text=tier_counts.values, textposition="outside",
            ))
            bar_fig.update_layout(
                height=180, margin=dict(t=30, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Risk Distribution", font=dict(size=11, color="#888"), x=0),
                yaxis=dict(showgrid=False, showticklabels=False),
                xaxis=dict(showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})
