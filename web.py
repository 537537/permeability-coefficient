import streamlit as st
import numpy as np
import pandas as pd

# ========== 页面配置 ==========
st.set_page_config(page_title="Pervious Concrete Permeability Prediction",
                   page_icon="💧",
                   layout="wide")

# ========== 自定义CSS美化 ==========
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.main {
    background-color: white;
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
h1, h2, h3 {
    color: #003366;
}
.stButton>button {
    background-color: #0072B5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #005999;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ========== 页面标题 ==========
st.title("💧 Pervious Concrete Permeability Prediction Platform")
st.markdown("### Input Parameters")

# ========== 输入参数 ==========
col1, col2, col3, col4 = st.columns(4)
col5, col6, col7, col8 = st.columns(4)

with col1:
    W_C = st.number_input("Water-Cement Ratio (W/C)", min_value=0.2, max_value=1.0, value=0.35, step=0.01)
with col2:
    A_C = st.number_input("Aggregate-Cement Ratio (A/C)", min_value=1.0, max_value=6.0, value=3.5, step=0.1)
with col3:
    Dmin = st.number_input("Minimum Particle Size (mm)", min_value=1.0, max_value=10.0, value=4.75, step=0.1)
with col4:
    ASR = st.number_input("Aggregate Size Range (mm)", min_value=1.0, max_value=20.0, value=9.5, step=0.1)
with col5:
    Porosity = st.number_input("Porosity (%)", min_value=5.0, max_value=40.0, value=20.0, step=0.5)
with col6:
    Shape = st.selectbox("Aggregate Shape", ["Round", "Angular"])
with col7:
    Diameter = st.number_input("Specimen Diameter (mm)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)
with col8:
    Height = st.number_input("Specimen Height (mm)", min_value=50.0, max_value=200.0, value=150.0, step=1.0)

# ========== 预测按钮 ==========
st.markdown("### Prediction Result")

predict_button = st.button("🚀 Predict Permeability")

if predict_button:
    # 固定输出数值
    predicted_value = 4.75

    st.success(f"**Predicted Permeability Coefficient:** {predicted_value:.2f} mm/s")

    # 模拟展示数据表格
    data = {
        "Parameter": ["W/C", "A/C", "Dmin", "ASR", "Porosity", "Shape", "Diameter", "Height"],
        "Value": [W_C, A_C, Dmin, ASR, Porosity, Shape, Diameter, Height]
    }
    df = pd.DataFrame(data)
    st.markdown("#### Input Summary")
    st.dataframe(df, use_container_width=True)

    st.markdown("#### Model Explanation (Mock)")
    st.info("This section could display SHAP or PDP interpretation results once the model is integrated.")

# ========== 页脚 ==========
st.markdown("---")
st.caption("© 2025 Pavement Material Intelligence Research Group")

