import streamlit as st
import numpy as np
import joblib
import os
import shap
import streamlit.components.v1 as components
import st_shap # 导入 st_shap 库

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

# ========== 标题 ==========
st.markdown("<h1 style='text-align:center;'>💧 Pervious Concrete Permeability Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#1E88E5;'>Enter the following parameters to predict the Permeability Coefficient (PEC)</h4>", unsafe_allow_html=True)
st.markdown("---")

# ========== 模型路径 ==========
MODEL_PATH = "final_xgboost_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or scaler file is missing! Please check the file paths.")
else:
    # 使用缓存来加载模型和scaler，避免每次交互都重新加载，提高性能
    @st.cache_resource
    def load_model_and_scaler():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler

    model, scaler = load_model_and_scaler()

    # ========== 输入参数布局 ==========
    # 第一行
    col1, col2, col3 = st.columns(3)
    with col1:
        W_C = st.number_input("Water-Cement Ratio (W/C)", min_value=0.0, value=0.3, step=0.01)
    with col2:
        A_C = st.number_input("Aggregate-Cement Ratio (A/C)", min_value=0.0, value=4.0, step=0.1)
    with col3:
        Dmin = st.number_input("Minimum Aggregate Size (Dmin, mm)", min_value=0.0, value=4.75, step=0.1)

    # 第二行
    col4, col5, col6 = st.columns(3)
    with col4:
        Dmax = st.number_input("Maximum Aggregate Size (Dmax, mm)", min_value=0.0, value=9.5, step=0.1)
    with col5:
        Porosity = st.number_input("Porosity (%)", min_value=0.0, value=15.0, step=0.1)
    with col6:
        shape_option = st.selectbox("Specimen Shape (SS)", ["Cylinder", "Cube"])
        SS = 1 if shape_option == "Cylinder" else 2

    # 第三行
    col7, col8, col9 = st.columns(3)
    with col7:
        SD = st.number_input("Specimen Diameter (SD, mm)", min_value=0.0, value=100.0, step=1.0)
    with col8:
        SH = st.number_input("Specimen Height (SH, mm)", min_value=0.0, value=200.0, step=1.0)
    with col9:
        tm_option = st.selectbox("Test Method (TM)", ["Constant Head", "Fall Head"])
        TM = 1 if tm_option == "Constant Head" else 2

    # ========== 预测按钮 ==========
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    predict_button = st.button("🔍 Predict PEC")
    st.markdown("</div>", unsafe_allow_html=True)

    # ========== 执行预测 ==========
    if predict_button:
        try:
            # 构造输入数据
            input_data = np.array([[W_C, A_C, Dmin, Dmax, Porosity, SS, SD, SH, TM]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            # 显示预测结果
            st.markdown(f"""
            <div class="result-card">
                <h2>✅ Predicted Permeability Coefficient (PEC)</h2>
                <h1 style="color:#0D47A1;">{prediction:.6f} mm/s</h1>
            </div>
            """, unsafe_allow_html=True)

            # ========== SHAP Force Plot (已修复) ==========
            st.markdown("### 🔹 SHAP Force Plot (Feature Contributions)")
            
            # 创建 explainer 并计算 SHAP 值
            # XGBoost模型的explainer建议使用shap.TreeExplainer或shap.Explainer
            explainer = shap.Explainer(model)
            shap_values = explainer(input_scaled)
            
            # 【关键修改】使用 st_shap.shap_explanation 来正确显示交互式 force plot
            st_shap.shap_explanation(shap_values[0])

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
