import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ========== 页面配置 ==========
st.set_page_config(page_title='韧性强度预测演示平台', page_icon='🧪', layout='wide')

# ========== 自定义CSS美化 ==========
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f7fbff 0%, #eef6ff 100%);
    color: #0f172a;
}
.card {
    background: white;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 6px 18px rgba(47,84,162,0.08);
}
.big-btn button {
    background: linear-gradient(90deg,#2563eb,#60a5fa) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
}
.result-value {
    font-size: 32px;
    font-weight: 700;
    color: #064e3b;
}
</style>
""", unsafe_allow_html=True)

# ========== 页面结构 ==========
with st.sidebar.container():
    st.header('说明')
    st.write('此页面为韧性强度预测系统的**外观演示版**。')
    st.write('当前未加载真实模型，所有预测结果固定为 **4.75**。')
    st.write('输入框、按钮、图表功能保持原样，仅用于展示界面布局。')

st.title('韧性强度预测 (演示版)')
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('输入参数')

    wb = st.number_input('水胶比 (Water-Binder ratio)', min_value=0.20, max_value=0.80, value=0.35, step=0.01)
    flyash = st.number_input('粉煤灰占胶凝材料比例 (%)', min_value=0.0, max_value=60.0, value=10.0, step=0.5)
    silica = st.number_input('硅灰占胶凝材料比例 (%)', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    slag = st.number_input('矿粉占胶凝材料比例 (%)', min_value=0.0, max_value=60.0, value=20.0, step=0.5)
    far = st.selectbox('纤维长径比 (Aspect Ratio)', options=[1000, 1200, 1500, 2000], index=2)
    fv = st.number_input('纤维掺量 (体积分数 %)', min_value=0.00, max_value=5.00, value=0.50, step=0.05)
    cement = st.number_input('水泥用量 (kg/m³)', min_value=100.0, max_value=1000.0, value=750.0, step=10.0)
    sand = st.number_input('砂用量 (kg/m³)', min_value=0.0, max_value=2000.0, value=600.0, step=10.0)
    sp = st.number_input('减水剂掺量 (%)', min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    predict_btn = st.button('预测', key='predict', help='点击进行韧性强度预测')

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('预测结果')
    result_placeholder = st.empty()
    shap_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# ========== 预测逻辑（固定输出） ==========
if predict_btn:
    Tf = 4.75  # 固定预测结果
    html_result = f"""
    <div class='card' style='text-align:center'>
      <div style='font-size:14px;color:#64748b'>预测韧性强度 (无量纲折压比)</div>
      <div class='result-value'>{Tf:.2f}</div>
      <div style='color:#475569;margin-top:6px;font-size:13px'>演示版固定输出</div>
    </div>
    """
    result_placeholder.markdown(html_result, unsafe_allow_html=True)

    # 模拟SHAP条形图
    fig, ax = plt.subplots(figsize=(6,3))
    fake_features = ['wb','flyash','silica','slag','far','fv','cement','sand','sp']
    fake_values = np.random.uniform(-0.05, 0.05, size=len(fake_features))
    pd.Series(fake_values, index=fake_features).sort_values().plot(kind='barh', ax=ax)
    ax.set_title('SHAP 特征贡献 (演示图)')
    ax.set_xlabel('贡献度')
    st.pyplot(fig)

else:
    with col2:
        st.info('输入参数后点击“预测”可获得演示结果 (固定为4.75)。')

st.markdown('<hr />', unsafe_allow_html=True)
st.markdown('**说明：** 本系统为演示用途，展示模型预测界面布局及交互效果，不执行真实机器学习推理。', unsafe_allow_html=True)
