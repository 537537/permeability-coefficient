import streamlit as st
import joblib
import numpy as np

# ========== 页面设置 ==========
st.set_page_config(page_title="Permeability Coefficient Predictor", page_icon="💧", layout="centered")

# ========== 自定义样式 ==========
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .main-title {
        text-align: center;
        font-size: 32px;
        color: #003366;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #004c99;
        color: white;
    }
    .result-box {
        text-align: center;
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ========== 标题 ==========
st.markdown('<p class="main-title">Permeability Coefficient Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict the permeability coefficient (PEC) of pervious concrete using a trained XGBoost model</p>', unsafe_allow_html=True)

# ========== 加载模型和标准化器 ==========
MODEL_PATH = "final_xgboost_model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ========== 输入部分 ==========
st.markdown("### 🧮 Input Parameters")

# 第一行
col1, col2, col3 = st.columns(3)
with col1:
    wc = st.number_input("Water-Cement Ratio (W/C)", min_value=0.0, step=0.01, format="%.2f", placeholder="W/C")
with col2:
    ac = st.number_input("Aggregate-Cement Ratio (A/C)", min_value=0.0, step=0.01, format="%.2f", placeholder="A/C")
with col3:
    dmin = st.number_input("Minimum Aggregate Size (Dmin, mm)", min_value=0.0, step=0.1, format="%.1f", placeholder="Dmin (mm)")

# 第二行
col4, col5, col6 = st.columns(3)
with col4:
    dmax = st.number_input("Maximum Aggregate Size (Dmax, mm)", min_value=0.0, step=0.1, format="%.1f", placeholder="Dmax (mm)")
with col5:
    porosity = st.number_input("Porosity (%)", min_value=0.0, step=0.1, format="%.1f", placeholder="Porosity (%)")
with col6:
    ss = st.number_input("Specimen Shape (SS)", min_value=1, max_value=3, step=1, placeholder="SS")

# 第三行
col7, col8, col9 = st.columns(3)
with col7:
    sd = st.number_input("Specimen Diameter (SD, mm)", min_value=0.0, step=1.0, format="%.0f", placeholder="SD (mm)")
with col8:
    sh = st.number_input("Specimen Height (SH, mm)", min_value=0.0, step=1.0, format="%.0f", placeholder="SH (mm)")
with col9:
    test_method = st.selectbox("Test Method (TM)", ["Constant Head (1)", "Fall Head (2)"])
    test_method_value = 1 if "Constant" in test_method else 2

# ========== 预测 ==========
if st.button("🔮 Predict Permeability Coefficient"):
    try:
        # 组装输入数据
        input_data = np.array([[wc, ac, dmin, dmax, porosity, ss, sd, sh, test_method_value]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # 显示结果
        st.markdown(f"""
        <div class="result-box">
            <h3>Predicted Permeability Coefficient (mm/s)</h3>
            <p style="font-size:26px; color:#0073e6; font-weight:bold;">{prediction:.6f}</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

