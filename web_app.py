import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============ 1. 页面配置 ============
st.set_page_config(
    page_title="青少年脊柱弯曲异常运动疗法疗效效果预测与方案推荐",
    layout="centered"
)

# ============ 2. 加载模型 & PCA / 插补 参数 ============

@st.cache_resource
def load_model():
    # 最优随机森林模型（训练阶段已保存）
    model = joblib.load("best_rf_model.pkl")
    return model

@st.cache_resource
def load_imputer():
    # 随机森林插补器（根据 df2.csv 训练并保存）
    imputer = joblib.load("rf_imputer.pkl")
    return imputer

@st.cache_resource
def load_pca_params():
    """
    读取 R 导出的 PCA 参数：
    - pca_set1_*：活动度 PCA
    - pca_set2_*：平衡度 PCA
    约定各 CSV 列名为：
    center: 变量, 中心
    scale:  变量, 标准差
    loadings: 变量, PC1, PC2, ...
    """
    # ---- 活动度 PCA ----
    center1_df = pd.read_csv("pca_set1_center.csv")        # 列：变量, 中心
    scale1_df  = pd.read_csv("pca_set1_scale.csv")         # 列：变量, 标准差
    loadings1_df = pd.read_csv("pca_set1_loadings.csv")    # 列：变量, PC1, PC2, ...

    act_vars = center1_df["变量"].tolist()
    center1 = center1_df["中心"].values.astype(float)
    scale1  = scale1_df["标准差"].values.astype(float)
    rotation1 = loadings1_df.set_index("变量").values.astype(float)

    # ---- 平衡度 PCA ----
    center2_df = pd.read_csv("pca_set2_center.csv")        # 列：变量, 中心
    scale2_df  = pd.read_csv("pca_set2_scale.csv")         # 列：变量, 标准差
    loadings2_df = pd.read_csv("pca_set2_loadings.csv")    # 列：变量, PC1, PC2, ...

    bal_vars = center2_df["变量"].tolist()
    center2 = center2_df["中心"].values.astype(float)
    scale2  = scale2_df["标准差"].values.astype(float)
    rotation2 = loadings2_df.set_index("变量").values.astype(float)

    return {
        "act_vars": act_vars,
        "center1": center1,
        "scale1": scale1,
        "rotation1": rotation1,
        "bal_vars": bal_vars,
        "center2": center2,
        "scale2": scale2,
        "rotation2": rotation2,
    }

model = load_model()
imputer = load_imputer()
pca_params = load_pca_params()

act_vars = pca_params["act_vars"]
center1 = pca_params["center1"]
scale1 = pca_params["scale1"]
rotation1 = pca_params["rotation1"]

bal_vars = pca_params["bal_vars"]
center2 = pca_params["center2"]
scale2 = pca_params["scale2"]
rotation2 = pca_params["rotation2"]

# 训练时 df2 的特征顺序（除去 Result）：
FEATURE_ORDER = [
    "Type", "Gender", "Age", "BMI", "FMR",
    "ST", "SCT", "ATI", "KA",
    "PC1", "PC2", "PC3", "PC4"
]

# ======== 2.1 活动度 / 平衡度 中文标签映射 ========

ACT_LABEL_MAP = {
    "C-LLB": "颈椎左侧向弯曲",
    "C-RLB": "颈椎右侧向弯曲",
    "T-LLB": "胸椎左侧向弯曲",
    "T-RLB": "胸椎右侧向弯曲",
    "L-LLB": "腰椎左侧向弯曲",
    "L-RLB": "腰椎右侧向弯曲",
    "C-FFT": "颈椎前屈",
    "C-BF":  "颈椎后伸",
    "T-FFT": "胸椎前屈",
    "T-BF":  "胸椎后伸",
    "L-FFT": "腰椎前屈",
    "L-BF":  "腰椎后伸",
    "C-LHR": "颈椎左转",
    "C-RHR": "颈椎右转",
    "T-LHR": "胸椎左转",
    "T-RHR": "胸椎右转",
    "L-LHR": "腰椎左转",
    "L-RHR": "腰椎右转",
}

BAL_LABEL_MAP = {
    "HB": "头部平衡",
    "SB": "肩部平衡",
    "PB": "髋骨平衡",
}

# ======== 2.2 ST / SCT 中文选项 → 数值编码（与 df2 映射一致） ========

ST_OPTIONS = {
    "无侧弯": 0,   # 对应 df2 中 "No"
    "Ⅰ度侧弯": 1,  # "1degree"
    "Ⅱ度侧弯": 2,  # "2degree"
    "Ⅲ度侧弯": 3,  # "3degree"
}

SCT_OPTIONS = {
    "无异常": 0,             # "No"
    "脊柱前凸（lordosis）": 1,
    "脊柱后凸（kyphosis）": 2,
    "平背（Flat back）": 3,
}

# ============ 3. PCA 计算函数 ============

def compute_activity_pcs_from_r(act_input: dict) -> np.ndarray:
    """使用 R 导出的活动度 PCA 参数计算主成分"""
    X_vec = np.array([act_input[var] for var in act_vars], dtype=float).reshape(1, -1)
    X_scaled = (X_vec - center1) / scale1
    PCs = np.dot(X_scaled, rotation1)
    return PCs[0]

def compute_balance_pcs_from_r(bal_input: dict) -> np.ndarray:
    """使用 R 导出的平衡度 PCA 参数计算主成分"""
    X_vec = np.array([bal_input[var] for var in bal_vars], dtype=float).reshape(1, -1)
    X_scaled = (X_vec - center2) / scale2
    PCs = np.dot(X_scaled, rotation2)
    return PCs[0]

def make_X(type_code, gender, age, bmi, fmr, st_code, sct_code,
           ati, ka, pc1, pc2, pc3, pc4):
    """严格按 df2 特征顺序构造一行特征"""
    row = {
        "Type": float(type_code),
        "Gender": float(gender),
        "Age": float(age),
        "BMI": float(bmi),
        "FMR": float(fmr),
        "ST": float(st_code),
        "SCT": float(sct_code),
        "ATI": float(ati),
        "KA": float(ka),
        "PC1": float(pc1),
        "PC2": float(pc2),
        "PC3": float(pc3),
        "PC4": float(pc4),
    }
    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

# ============ 4. 页面标题和说明 + 输入框样式 ============

st.title("青少年脊柱弯曲异常运动疗法效果预测与方案推荐系统")

st.markdown("""
<style>
/* 主标题字号 */
h1 {
    font-size: 1.3rem !important;
}

/* 所有子标题（如基本信息、体成分等）字号 */
h3 {
    font-size: 1.0rem !important;
}

/* 默认所有数字输入为白色背景 */
input[type="number"] {
    background-color: rgba(255,255,255,1);
}

/* 只要 value 不是 0.0（即已经填写/非默认值），背景改为淡绿色 30% 透明 */
input[type="number"][value]:not([value="0.0"]) {
    background-color: rgba(144,238,144,0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
这个网页计算器的总体目的是为了辅助选择合适的运动疗法。通过输入患者的基本信息、体成分和脊柱健康数据，它能即时输出适合该患者的运动疗法和有效改善率。其设计理念是将复杂的多维特征输入转化为直观的个体化预测结果，使“黑箱”模型转化为辅助决策工具，从而推动实际运动疗法方案的指定，为个体化干预决策提供科学依据，实现制定个性化康复方案的目标。
""")

st.divider()

# ============ 5. 输入表单 ============

with st.form("patient_form"):
    st.subheader("① 基本信息")
    col1, col2 = st.columns(2)
    with col1:
        gender_cn = st.selectbox("性别", ["男", "女"])
    with col2:
        age = st.number_input(
            "年龄（岁）", min_value=6.0, max_value=25.0,
            value=13.0, step=0.1, format="%.1f"
        )

    st.subheader("② 体成分与脊柱形态指标")
    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input(
            "身高（m）", min_value=1.0, max_value=2.0,
            value=1.6, step=0.1, format="%.1f"
        )
    with col4:
        weight = st.number_input(
            "体重（kg）", min_value=20.0, max_value=120.0,
            value=50.0, step=0.1, format="%.1f"
        )

    col5, col6 = st.columns(2)
    with col5:
        fat_mass = st.number_input(
            "体脂量（kg）", min_value=1.0, max_value=80.0,
            value=10.0, step=0.1, format="%.1f"
        )
    with col6:
        muscle_mass = st.number_input(
            "肌肉量（kg）", min_value=1.0, max_value=80.0,
            value=30.0, step=0.1, format="%.1f"
        )

    col7, col8 = st.columns(2)
    with col7:
        ati = st.number_input(
            "躯干倾斜角 ATI（°）", min_value=0.0, max_value=40.0,
            value=6.0, step=0.1, format="%.1f"
        )
    with col8:
        ka = st.number_input(
            "脊柱后凸角 KA（°）", min_value=0.0, max_value=90.0,
            value=35.0, step=0.1, format="%.1f"
        )

    col9, col10 = st.columns(2)
    with col9:
        st_label = st.selectbox(
            "脊柱侧弯程度 ST",
            list(ST_OPTIONS.keys()),
            index=0
        )
    with col10:
        sct_label = st.selectbox(
            "矢状位弯曲类型 SCT",
            list(SCT_OPTIONS.keys()),
            index=0
        )

    st.subheader("③ 脊柱活动度指标（°）")
    act_inputs = {}
    cols_act = st.columns(3)
    for i, var in enumerate(act_vars):
        label = ACT_LABEL_MAP.get(var, var)
        with cols_act[i % 3]:
            act_inputs[var] = st.number_input(
                f"{label}", value=0.0, step=0.1, format="%.1f",
                key=f"act_{var}"
            )

    st.subheader("④ 脊柱平衡度指标（°）")
    bal_inputs = {}
    cols_bal = st.columns(3)
    for i, var in enumerate(bal_vars):
        label = BAL_LABEL_MAP.get(var, var)
        with cols_bal[i % 3]:
            bal_inputs[var] = st.number_input(
                f"{label}", value=0.0, step=0.1, format="%.1f",
                key=f"bal_{var}"
            )

    submitted = st.form_submit_button("▶ 计算两种疗法的预测结果与推荐方案")

# ============ 6. 预测逻辑 ============

def categorize_bmi(bmi_value: float, gender: int):
    """
    根据性别返回 BMI 分类与颜色
    gender: 0=男, 1=女
    """
    # -----------------------
    # 男性 BMI 分类
    # -----------------------
    if gender == 0:
        if bmi_value < 18.5:
            return "偏低", "#FFA726"   # 橙色
        elif bmi_value < 24.0:
            return "正常", "#66BB6A"   # 绿色
        elif bmi_value < 28.0:
            return "超重", "#EF5350"   # 红色偏浅
        else:
            return "肥胖", "#C62828"   # 深红色

    # -----------------------
    # 女性 BMI 分类
    # -----------------------
    else:
        if bmi_value < 18.0:
            return "偏低", "#FFA726"
        elif bmi_value < 23.5:
            return "正常", "#66BB6A"
        elif bmi_value < 27.0:
            return "超重", "#EF5350"
        else:
            return "肥胖", "#C62828"


if submitted:
    try:
        # 6.1 性别 / ST / SCT 编码
        gender = 0 if gender_cn == "男" else 1
        st_code = ST_OPTIONS[st_label]
        sct_code = SCT_OPTIONS[sct_label]

        # 6.2 计算 BMI 与 FMR
        if height <= 0:
            st.error("身高必须大于 0，请重新输入。")
            st.stop()
        bmi = weight / (height ** 2)  # BMI = 体重(kg) / 身高(m)^2

        if muscle_mass <= 0:
            st.error("肌肉量必须大于 0 才能计算 FMR，请重新输入。")
            st.stop()
        fmr = fat_mass / muscle_mass   # FMR = 体脂量 / 肌肉量

        # 6.2.1 BMI 分类 + 彩色标签
        bmi_cat, bmi_color = categorize_bmi(bmi, gender)
        bmi_tag = (
            f'<span style="background-color:{bmi_color}; '
            f'color:white; padding:2px 8px; border-radius:12px;">'
            f'{bmi:.1f}（{bmi_cat}）</span>'
        )

        # 6.2.2 FMR 彩色标签
        fmr_tag = (
            f'<span style="background-color:#42A5F5; '
            f'color:white; padding:2px 8px; border-radius:12px;">'
            f'{fmr:.1f}</span>'
        )

        # 6.2.3 在页面放一个小表格统一展示 BMI & FMR
        st.markdown(
            f"""
            <div style="margin-top:0.5rem; margin-bottom:0.5rem;">
            <table style="font-size:0.9rem; border-collapse:collapse;">
              <tr>
                <th style="text-align:left; padding:4px 10px; border-bottom:1px solid #ddd;">指标</th>
                <th style="text-align:left; padding:4px 10px; border-bottom:1px solid #ddd;">数值 / 解释</th>
              </tr>
              <tr>
                <td style="padding:4px 10px;">BMI（体重/身高²）</td>
                <td style="padding:4px 10px;">{bmi_tag}</td>
              </tr>
              <tr>
                <td style="padding:4px 10px;">FMR（脂肪肌肉比）</td>
                <td style="padding:4px 10px;">{fmr_tag}</td>
              </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 6.3 计算 PC1–PC4（活动度 + 平衡度 PCA）
        act_pcs = compute_activity_pcs_from_r(act_inputs)
        bal_pcs = compute_balance_pcs_from_r(bal_inputs)

        if act_pcs.size < 3 or bal_pcs.size < 1:
            st.error("PCA 结果维度不足，请检查导出的 PCA 文件是否完整。")
            st.stop()

        pc1, pc2, pc3 = act_pcs[0], act_pcs[1], act_pcs[2]
        pc4 = bal_pcs[0]

        # 6.4 两种疗法分别预测（接入随机森林插补）
        def predict_one(type_code: int):
            # 构造一行原始特征
            X_raw = make_X(
                type_code, gender, age, bmi, fmr,
                st_code, sct_code, ati, ka,
                pc1, pc2, pc3, pc4
            )

            # 使用随机森林插补器补全（尽管没有缺失，也是一种稳健处理）
            X_imp_arr = imputer.transform(X_raw)  # ndarray
            X_imp = pd.DataFrame(X_imp_arr, columns=FEATURE_ORDER)

            # 使用已训练好的随机森林预测模型
            y_pred = int(model.predict(X_imp)[0])
            prob_yes = float(model.predict_proba(X_imp)[0, 1])
            return X_imp, y_pred, prob_yes

        # 0: 单一 SPS 疗法; 1: 联合疗法
        X_sps, y_sps, p_sps = predict_one(0)
        X_combo, y_combo, p_combo = predict_one(1)

        # 6.5 展示结果
        st.divider()
        st.subheader("两种运动疗法的预测结果对比")

        res_df = pd.DataFrame([
            {
                "疗法": "螺旋肌肉链训练法",
                "预测结局": "改善有效" if y_sps == 1 else "改善无效",
                "改善有效概率(%)": round(p_sps * 100, 1),
            },
            {
                "疗法": "螺旋肌肉链训练联合本体感觉神经肌肉促进技术法",
                "预测结局": "改善有效" if y_combo == 1 else "改善无效",
                "改善有效概率(%)": round(p_combo * 100, 1),
            },
        ])
        st.dataframe(res_df, use_container_width=True)

        # 6.6 推荐逻辑
        delta = abs(p_sps - p_combo)
        if delta < 0.02:
            st.warning(
                f"两种疗法预测有效概率非常接近（差值 {delta*100:.1f}%），"
                f"建议结合临床经验综合判断。"
            )
        else:
            if p_sps > p_combo:
                st.success(
                    f"推荐方案：**螺旋肌肉链训练法（SPS）**，"
                    f"预测“改善有效”概率约为 **{p_sps*100:.1f}%**。"
                )
            else:
                st.success(
                    f"推荐方案：**螺旋肌肉链训练联合本体感觉神经肌肉促进技术法（COM）**，"
                    f"预测“改善有效”概率约为 **{p_combo*100:.1f}%**。"
                )

        # 6.7 查看主成分
        with st.expander("查看本次计算得到的主成分（PC1–PC4）："):
            st.write(pd.DataFrame([{
                "PC1": pc1, "PC2": pc2, "PC3": pc3, "PC4": pc4
            }]))

    except Exception as e:
        st.error("预测时出现错误，请检查以下项目：")
        st.write("- best_rf_model.pkl 是否与训练阶段模型一致")
        st.write("- rf_imputer.pkl 是否已按 df2.csv 训练并放在同一目录")
        st.write("- PCA CSV 文件是否与 R 导出文件匹配（变量/中心/标准差/载荷列名是否一致）")
        st.write("- 输入数值是否完整且合理（尤其是身高、体重、体脂、肌肉量）")
        st.exception(e)

