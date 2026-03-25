import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Heart Failure Predictor",
    page_icon="❤️",
    layout="centered"
)

st.markdown("""
<style>
    .main-title {
        color: #e63946;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #e63946;
    }
    .risk-high {
        background-color: #ffeaea;
        color: #e63946;
        padding: 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
    .risk-low {
        background-color: #e8f7ef;
        color: #2a9d8f;
        padding: 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">❤️ Heart Failure Risk Predictor</h1>', unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/heart_disease_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        return model, scaler, feature_names, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, feature_names, error_msg = load_models()

if error_msg:
    st.error(f"⚠️ **Model Loading Error:** {error_msg}")
    st.info("""
    **To fix this issue:**
    
    1. **Create sample data:**
    ```bash
    python create_sample_data.py
    ```
    
    2. **Train the model:**
    ```bash
    python train_model_simple.py
    ```
    
    3. **Refresh this page**
    """)
    st.warning("Please train the model first to enable predictions.")
    st.stop()

st.markdown("### 📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    age = st.number_input("**Age (years)**", min_value=30, max_value=100, value=65, step=1)
    ejection_fraction = st.number_input("**Ejection Fraction (%)**", min_value=10, max_value=80, value=35, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    serum_creatinine = st.number_input(
        "**Serum Creatinine (mg/dL)**",
        min_value=0.5,
        max_value=10.0,
        value=1.2,
        step=0.1,
        format="%.1f"
    )
    serum_sodium = st.number_input("**Serum Sodium (mEq/L)**", min_value=110, max_value=150, value=137, step=1)
    time = st.number_input("**Follow-up Time (days)**", min_value=0, max_value=300, value=100, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### 📊 Current Parameters")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Age", f"{age} yrs")

with col2:
    status = "⚠️ Low" if ejection_fraction < 40 else "✅ Normal" if ejection_fraction >= 55 else "↔️ Borderline"
    st.metric("EF", f"{ejection_fraction}%", status)

with col3:
    status = "⚠️ High" if serum_creatinine > 1.3 else "✅ Normal"
    st.metric("Creatinine", f"{serum_creatinine:.1f}", status)

with col4:
    status = "⚠️ Low" if serum_sodium < 135 else "✅ Normal" if serum_sodium <= 145 else "⚠️ High"
    st.metric("Sodium", f"{serum_sodium}", status)

with col5:
    st.metric("Follow-up", f"{time} days")

input_data = {
    "age": age,
    "ejection_fraction": ejection_fraction,
    "serum_creatinine": serum_creatinine,
    "serum_sodium": serum_sodium,
    "time": time
}

input_df = pd.DataFrame([input_data])
input_df = input_df[feature_names]
input_scaled = scaler.transform(input_df)


st.markdown("---")
predict_button = st.button("🚀 Predict Risk Score", type="primary", use_container_width=True)

if predict_button:
    y_prob = model.predict_proba(input_scaled)[:, 1]
    threshold = 0.46
    y_pred = int(y_prob[0] >= threshold)

    st.markdown("## 📈 Prediction Results")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=y_prob[0] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#e63946" if y_pred == 1 else "#2a9d8f"},
            'steps': [
                {'range': [0, 30], 'color': "#2a9d8f"},
                {'range': [30, 46], 'color': "#ffba08"},
                {'range': [46, 100], 'color': "#e63946"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
    fig.update_layout(height=300)

    if y_pred == 1:
        st.markdown('<div class="risk-high">🚨 HIGH RISK OF DEATH</div>', unsafe_allow_html=True)
        st.error(f"**Probability: {y_prob[0]:.1%}** - Immediate medical attention recommended!")

        st.warning("""
        **Recommendations:**
        - Immediate medical consultation
        - Consider hospitalization
        - Continuous monitoring
        - Medication review
        """)
    else:
        st.markdown('<div class="risk-low">✅ LOW RISK OF DEATH</div>', unsafe_allow_html=True)
        st.success(f"**Probability: {y_prob[0]:.1%}** - Continue regular monitoring")
        st.info("""
            **Recommendations:**
            - Regular follow-up appointments
            - Lifestyle management
            - Self-monitoring of symptoms
            - Continue current treatment
            """)
    st.markdown("---")
    st.markdown("---")
st.markdown("## 📂 Batch Prediction (Big Dataset)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head(), use_container_width=True)

    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    X = df[feature_names]
    X_scaled = scaler.transform(X)

    df["Risk_Probability"] = model.predict_proba(X_scaled)[:, 1]
    df["Risk_Level"] = df["Risk_Probability"].apply(
        lambda x: "High Risk" if x >= 0.46 else "Low Risk"
    )

    st.subheader("📊 Prediction Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Results",
        csv,
        "heart_failure_predictions.csv",
        "text/csv"
    )
    
    st.markdown("### 🔍 Feature Contributions")
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = abs(model.coef_[0])
    else:
        importance = np.ones(len(feature_names))/len(feature_names)
    
    impact_data = []
    for i, feature in enumerate(feature_names):
        impact = "Increases Risk"
        if feature == "age" and age > 65:   
            impact = "Increases Risk"
        elif feature == "ejection_fraction" and ejection_fraction < 40:
            impact = "Increases Risk"
        elif feature == "serum_creatinine" and serum_creatinine > 1.3:
            impact = "Increases Risk"
        elif feature == "serum_sodium" and serum_sodium < 135:
            impact = "Increases Risk"
        elif feature == "time" and time < 60:
            impact = "Increases Risk"
        else:
            impact = "Decreases/Normal Risk"
        
        impact_data.append({
            "Feature": feature,
            "Importance": float(importance[i]),
            "Value": input_data[feature],
            "Impact": impact
        })
    
    impact_df = pd.DataFrame(impact_data)
    impact_df = impact_df.sort_values(by="Importance", ascending=False)

    st.dataframe(
        impact_df.style.format({
            'Value': '{:.1f}',
            'Importance': '{:.3f}'
        }).background_gradient(subset=['Importance'], cmap='Reds'),
        use_container_width=True
    )

    fig_bar = go.Figure(data=[
        go.Bar(
            x=impact_df['Feature'],
            y=impact_df['Importance'],
            marker_color=['#e63946' if imp == "Increases Risk" else '#2a9d8f' for imp in impact_df['Impact']],
            text=[f"Value: {val:.1f}" for val in impact_df['Value']],
            textposition='auto'
        )
    ])

    fig_bar.update_layout(
        title="Feature Importance",
        xaxis_title="Feature",
        yaxis_title="Importance Score",
        height=400
    )

    st.plotly_chart(fig_bar, use_container_width=True)
with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown("""
    This Heart Failure Risk Predictor estimates the likelihood of death due to heart failure using 5 key clinical parameters:
    - Age
    - Ejection Fraction
    - Serum Creatinine
    - Serum Sodium
    - Follow-up Time
    The model is trained on the Heart Failure Clinical Records Dataset and provides actionable insights based on patient data.
    """)
    st.markdown(
    "&copy; 2026 Heart Failure Risk Predictor | Developed by Jeevanandam D",
    unsafe_allow_html=True
    )

    if model is not None:
        st.success("✅ Model loaded successfully")
    else:
        st.error("⚠️ Model not loaded")

st.markdown("---")
st.caption("❤️ Heart Failure Risk Predictor v2.0 | 5-Feature Model")
