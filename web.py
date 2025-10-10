import streamlit as st
import numpy as np
import joblib
import os

# ========== 页面配置 ==========
st.set_page_config(page_title="Permeability Coefficient Predictor", page_icon="💧", layout="centered")

# ========== 恢复原始风格 CSS (Times New Roman + 蓝色按钮) ==========
st.markdown(
    """
    <style>
    * { font-family: 'Times New Roman', serif; }
    
    body {
        background-color: #f7f9fc;
    }

    .title {
        text-align: center;
        color: #0D47A1;
        font-size: 28px;
        margin-bottom: 4px;
    }
    .subtitle {
        text-align: center;
        color: #1565C0;
        font-size: 14px;
        margin-bottom: 18px;
    }

    /* 按钮样式（蓝） */
    div.stButton > button {
        background-color: #007BFF; 
        color: white;
        border-radius: 6px;
        border: 1px solid #0056b3;
        padding: 10px 24px;
        font-size: 15px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #0056b3;
    }

    /* 小提示文本样式（框下简写/单位） */
    .small-note {
        color: #555;
        font-size: 12px;
        margin-top: 4px;
    }

    /* 结果卡片 */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========== 标题 ==========
st.markdown("<div class='title'>Permeability Coefficient Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Use the trained XGBoost model to predict PEC (unit: mm/s)</div>", unsafe_allow_html=True)
st.markdown("---")

# ========== 模型与 scaler 路径 ==========
MODEL_PATH = "final_xgboost_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler file is missing! Please ensure final_xgboost_model.pkl and scaler.pkl are in the app folder.")
else:
    # 加载模型
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Failed to load model/scaler: {e}")
        st.stop()

    # ========== 输入区（3 行，每行 3 列） ==========
    st.markdown("### Input Parameters")

    # 第一行: W/C, A/C, Dmin
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Water-Cement Ratio**")              # 框外全称
        wc = st.number_input("", min_value=0.0, step=0.01, format="%.2f", value=0.30, key="wc")
        st.markdown("<div class='small-note'>(W/C)</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Aggregate-Cement Ratio**")
        ac = st.number_input("", min_value=0.0, step=0.01, format="%.2f", value=4.00, key="ac")
        st.markdown("<div class='small-note'>(A/C)</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("**Minimum Aggregate Size**")
        dmin = st.number_input("", min_value=0.0, step=0.1, format="%.1f", value=4.75, key="dmin")
        st.markdown("<div class='small-note'>(Dmin, mm)</div>", unsafe_allow_html=True)

    # 第二行: Dmax, Porosity, Specimen Shape
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("**Maximum Aggregate Size**")
        dmax = st.number_input("", min_value=0.0, step=0.1, format="%.1f", value=9.50, key="dmax")
        st.markdown("<div class='small-note'>(Dmax, mm)</div>", unsafe_allow_html=True)

    with col5:
        st.markdown("**Porosity**")
        porosity = st.number_input("", min_value=0.0, step=0.1, format="%.1f", value=15.0, key="porosity")
        st.markdown("<div class='small-note'>(Porosity, %)</div>", unsafe_allow_html=True)

    with col6:
        st.markdown("**Specimen Shape**")
        ss_option = st.selectbox("", ["Cylinder", "Cube"], key="ss_option")
        # Specimen Shape 编码：Cylinder -> 1, Cube -> 2
        ss = 1 if ss_option == "Cylinder" else 2
        st.markdown("<div class='small-note'>(SS: Cylinder=1, Cube=2)</div>", unsafe_allow_html=True)

    # 第三行: Specimen Diameter, Specimen Height, Test Method
    col7, col8, col9 = st.columns(3)
    with col7:
        st.markdown("**Specimen Diameter**")
        sd = st.number_input("", min_value=0.0, step=1.0, format="%.0f", value=100.0, key="sd")
        st.markdown("<div class='small-note'>(SD, mm)</div>", unsafe_allow_html=True)

    with col8:
        st.markdown("**Specimen Height**")
        sh = st.number_input("", min_value=0.0, step=1.0, format="%.0f", value=200.0, key="sh")
        st.markdown("<div class='small-note'>(SH, mm)</div>", unsafe_allow_html=True)

    with col9:
        st.markdown("**Test Method**")
        tm_option = st.selectbox("", ["Constant Head (1)", "Fall Head (2)"], key="tm_option")
        tm = 1 if "Constant" in tm_option else 2
        st.markdown("<div class='small-note'>(TM: Constant Head=1, Fall Head=2)</div>", unsafe_allow_html=True)

    # ========== 预测按钮 ==========
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Permeability Coefficient"):
        try:
            # 组装输入 (顺序与模型训练一致)
            input_data = np.array([[wc, ac, dmin, dmax, porosity, ss, sd, sh, tm]])
            # 标准化并预测
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            # 结果展示（单位 mm/s）
            st.markdown(f"""
                <div class="result-card">
                    <h3>Predicted Permeability Coefficient (mm/s)</h3>
                    <p style="color:#0D47A1; font-size:22px; font-weight:700;">{prediction:.6f} mm/s</p>
                </div>
            """, unsafe_allow_html=True)

            # 可选：显示输入摘要，便于核对
            st.write("**Input summary:**", f"W/C={wc}, A/C={ac}, Dmin={dmin} mm, Dmax={dmax} mm, Porosity={porosity}%, SS={ss} ({ss_option}), SD={sd} mm, SH={sh} mm, TM={tm} ({tm_option})")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


