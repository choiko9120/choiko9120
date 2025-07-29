import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import streamlit as st
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Font setting for SHAP plots
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10

# ===================== 모델 불러오기 =====================
@st.cache_resource(show_spinner=False)
def load_model():
    import os
    os.environ["PYTHONHASHSEED"] = "0"  # 안정성 향상
    with open('calibrated_model_0729.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# 기본값 (중앙값 기준) 정의
def get_median_defaults():
    return {
        "Age": 65.0, "Gender": 1.0, "BMI": 22.5, "DM": 0.0, "HTN": 0.0, "Hb": 12.8, "PLT": 240.0,
        "BUN": 14.5, "Cr": 0.85, "Na": 138.0, "K": 4.1, "AST": 28.0, "ALT": 24.0,
        "Total bilirubin": 0.8, "Alb": 3.9, "INR": 1.05, "GGT": 42.0, "T-chol": 160.0,
        "ALP": 85.0, "CEA": 2.4, "CA19-9": 25.0
    }

# 예측 함수
def predict(calibrated_model, input_data):
    feature_names = [
        "Age", "Gender", "BMI", "DM", "HTN", "Hb", "PLT", "BUN", "Cr", "Na", "K", "AST",
        "ALT", "Total bilirubin", "Alb", "INR", "GGT", "T-chol", "ALP", "CEA", "CA19-9"
    ]
    input_df = pd.DataFrame([[input_data[f] for f in feature_names]], columns=feature_names)
    
    # 보정된 확률 예측
    prob = calibrated_model.predict_proba(input_df)[0][1]
    
    # SHAP은 내부 estimator 기반으로 계산
    explainer = shap.Explainer(calibrated_model.estimator)
    shap_values = explainer(input_df)
    
    return prob, shap_values

# ===================== Streamlit App =====================

def main():
    st.title("LiMPC: Predict liver metastasis risk at diagnosis in pancreatic cancer")
    st.sidebar.header("Input Clinical Variables (21 features)")
    st.sidebar.markdown("If any value is missing, please enter the median value.")

    features = {}
    feature_names = [
        "Age", "Gender", "BMI", "DM", "HTN", "Hb", "PLT", "BUN", "Cr", "Na", "K", "AST",
        "ALT", "Total bilirubin", "Alb", "INR", "GGT", "T-chol", "ALP", "CEA", "CA19-9"
    ]

    defaults = get_median_defaults()

    for feature in feature_names:
        label = f"{feature} (median: {defaults[feature]:.2f})"
        if feature == "PLT":
            label += " (*1000)"
        if feature == "Gender":
            label += " (Female:0, Male:1)"
        if feature in ["DM", "HTN"]:
            label += " (No:0, Yes:1)"
        
        if feature in ["Gender", "DM", "HTN"]:
            features[feature] = st.sidebar.selectbox(label, options=[0, 1], index=int(defaults[feature]))
        else:
             features[feature] = st.sidebar.number_input(label, value=float(defaults[feature]), step=0.0001, format="%.6f")

    if st.sidebar.button("Run Prediction"):
        model = load_model()
        prob, shap_values = predict(model, features)

        st.write("input", features)
        threshold_opt = 0.1396685168147087

        st.subheader("Prediction Result")
        if prob > threshold_opt:
            st.markdown(f"""
            <div style='background-color:#ffe6e6;padding:15px;border-radius:10px'>
            <span style='font-size:17px'> <b>High-risk for liver metastasis</b></span><br>
            <b>Predicted probability:</b> {prob:.2%}<br>
            <i>Note: This case may warrant additional clinical evaluation.</i>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color:#e6ffe6;padding:15px;border-radius:10px'>
            <span style='font-size:17px'> <b>Low-risk</b></span><br>
            <b>Predicted probability:</b> {prob:.2%}<br>
            <i>Note: This result may support consideration of surgical planning.</i>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("SHAP Explanation")
        st.markdown("Top 5 contributing features for this prediction:")

        fig, ax = plt.subplots()
        vals = np.abs(shap_values.values[0])
        top_idx = np.argsort(vals)[-5:][::-1]
        ax.barh(np.array(shap_values.feature_names)[top_idx], vals[top_idx], color='steelblue')
        ax.invert_yaxis()
        ax.set_xlabel("|SHAP value| (impact on model output)")
        ax.set_title("SHAP summary (top 5 features)")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
