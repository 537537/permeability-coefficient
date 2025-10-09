import streamlit as st
import numpy as np
import joblib
import os

# ========== 页面配置 ==========
st.set_page_config(page_title="Pervious Concrete Permeability Prediction",
                   page_icon="💧",
                   layout="wide")

# ========== 自定义CSS美化 ==========
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #0D47A1;
}
.stButton > button {
    background: linear-gradient(90deg, #1976D2, #0D47A1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 17px;
    font-weight: bold;
    transition: 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0D47A1, #1976D2);
    transform: scale(1.05);
}
div[data-testid="stNumberInput"] > label {
    font-weight: 600;
    color: #1A237E;
}
.result-card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    text-align: center;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# ========== 标题区 ==========
st.markdown("<h1 style='text-align:center;'>💧 Pervious Concrete Permeability Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#1E88E5;'>Enter the following mix parameters to predict the Permeability Coefficient (PEC)</h4>", unsafe_allow_html=True)
st.markdown("---")

# ========== 模型与标准化器路径 ==========
MODEL_PATH = "final_xgboost_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or scaler file is missing! Please check the file paths.")
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # ========== 输入区布局 ==========
    col1, col2, col3 = st.columns(3)

    with col1:
        W_C = st.number_input("W/C (Water-Cement Ratio)", min_value=0.0, value=0.3, step=0.01)
        Dmin = st.number_input("Dmin (Minimum Aggregate Size, mm)", min_value=0.0, value=4.75, step=0.1)
        Porosity = st.number_input("Porosity (%)", min_value=0.0, value=15.0, step=0.1)

    with col2:
        A_C = st.number_input("A/C (Aggregate-Cement Ratio)", min_value=0.0, value=4.0, step=0.1)
        Dmax = st.number_input("Dmax (Maximum Aggregate Size, mm)", min_value=0.0, value=9.5, step=0.1)
        SS = st.number_input("SS (Shape Strength Factor)", min_value=0.0, value=1.0, step=0.1)

    with col3:
        SD = st.number_input("SD (Diameter Strength Factor)", min_value=0.0, value=1.0, step=0.1)
        SH = st.number_input("SH (Height Strength Factor)", min_value=0.0, value=1.0, step=0.1)
        TM = st.number_input("TM (Test Method Code)", min_value=0.0, value=1.0, step=0.1)

    # ========== 预测按钮 ==========
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    predict_button = st.button("🔍 Predict PEC")
    st.markdown("</div>", unsafe_allow_html=True)

    # ========== 预测逻辑 ==========
    if predict_button:
        try:
            input_data = np.array([[W_C, A_C, Dmin, Dmax, Porosity, SS, SD, SH, TM]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            st.markdown(f"""
            <div class="result-card">
                <h2>✅ Predicted Permeability Coefficient (PEC)</h2>
                <h1 style="color:#0D47A1;">{prediction:.4f}</h1>
                <p style="color:gray;">Unit: cm/s or m/s (depending on dataset)</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
