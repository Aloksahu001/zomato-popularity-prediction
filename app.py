import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.sparse import hstack

#load file
tfidf_rest = joblib.load("tfidf_rest.pkl")
tfidf_cuisine = joblib.load("tfidf_cuisine.pkl")
model = joblib.load("model (2).pkl")
cuisine_options = joblib.load("cuisine_list.pkl")
rest_type_options = joblib.load("rest_type_list.pkl")

# st.markdown("---")
# st.subheader("👨‍💻 About the Developer")
# st.markdown("""
# **Developed by:** Alok Sahu  

# This project uses:
# - Random Forest Classifier
# - TF-IDF Vectorization
# - SHAP Explainability
# - Streamlit for Deployment

# 📊 Model Accuracy: 97%  
# 📈 ROC-AUC Score: 0.99  



st.set_page_config(page_title="Zomato Popularity Predictor", layout="centered")

st.title("🍽️ Zomato Restaurant Popularity Prediction")
st.write("Predict whether a restaurant is likely to be popular (Rating ≥ 4.0)")


# User Inputs

votes = st.number_input("Votes", min_value=0, value=200)
cost = st.number_input("Approx Cost (for two people)", min_value=0.0, value=500.0)

online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
book_table = st.selectbox("Book Table Available?", ["Yes", "No"])

# rest_type = st.text_input("Restaurant Type (e.g. Casual Dining)")

st.subheader("🏪 Restaurant Type")

custom_rest = st.checkbox("✏ Use Custom Restaurant Type")

if custom_rest:
    rest_type = st.text_input("Enter Restaurant Type", value="Casual Dining")
else:
    rest_type = st.selectbox(
        "Select Restaurant Type",
        rest_type_options,
        index=rest_type_options.index("Casual Dining") if "Casual Dining" in rest_type_options else 0
    )
# cuisines = st.text_input("Cuisines (e.g. North Indian, Chinese)")
# cuisines = st.selectbox(
#     "Select Cuisines",
#     cuisine_options,
#     index=cuisine_options.index("North Indian")
# )

# Load cuisine list (recommended method)

st.subheader("🍜 Cuisines Selection")

cuisine_mode = st.radio(
    "Choose Input Method",
    ["Select from List", "Type Custom"],
    horizontal=True
)

if cuisine_mode == "Select from List":
    cuisines = st.selectbox(
        "Select Cuisines",
        cuisine_options,
        index=cuisine_options.index("North Indian") if "North Indian" in cuisine_options else 0
    )
else:
    cuisines = st.text_input(
        "Type Custom Cuisines",
        value="North Indian"
    )

#####
online_order = 1 if online_order == "Yes" else 0
book_table = 1 if book_table == "Yes" else 0


# Prediction
if st.button("Predict Popularity"):

    numeric = np.array([[votes, cost, online_order, book_table]])

    rest_vec = tfidf_rest.transform([rest_type])
    cuisine_vec = tfidf_cuisine.transform([cuisines])

    X_input = hstack([numeric, rest_vec, cuisine_vec])

    prediction = model.predict(X_input)
    probability = model.predict_proba(X_input)[0][1] * 100

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.success("🔥 This restaurant is likely POPULAR!")
    else:
        st.warning("⚠️ This restaurant is NOT likely popular.")

    st.write(f"📊 Confidence Score: {round(probability,2)}%")


    # Probability Graph

    st.subheader("📊 Probability Breakdown")

    fig_prob, ax_prob = plt.subplots()
    ax_prob.bar(["Not Popular", "Popular"],
                [100 - probability, probability])
    ax_prob.set_ylabel("Probability (%)")
    ax_prob.set_title("Prediction Confidence")

    st.pyplot(fig_prob)

    
    # Feature Importance
 
    st.subheader("⭐ Top Influential Features")

    rest_features = tfidf_rest.get_feature_names_out()
    cuisine_features = tfidf_cuisine.get_feature_names_out()

    all_features = (
        ["votes", "approx_cost(for two people)", "online_order", "book_table"]
        + list(rest_features)
        + list(cuisine_features)
    )

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(importance_df["Feature"], importance_df["Importance"])
    ax_imp.invert_yaxis()
    ax_imp.set_title("Top 10 Important Features")

    st.pyplot(fig_imp)

   
    # SHAP Explainability
    
    st.subheader("🧠 Model Explanation (SHAP)")

    try:
        # Convert sparse to dense (safe for single input)
        X_dense = X_input.toarray()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_dense)

        # Extract class 1 (Popular) explanation safely
        shap_val = shap_values.values[0, :, 1]
        base_val = shap_values.base_values[0, 1]
        data_val = X_dense[0]

        explanation = shap.Explanation(
            values=shap_val,
            base_values=base_val,
            data=data_val,
            feature_names=all_features
        )

        fig_shap, ax_shap = plt.subplots()
        shap.plots.waterfall(explanation, show=False)

        st.pyplot(fig_shap)

    except Exception as e:
        st.warning("SHAP explanation temporarily unavailable.")
        st.text(str(e))



# Only ONE About Button


st.markdown("---")

if st.button("👨‍💻 About Developer"):

    st.subheader("👨‍💻 About the Developer")

    st.markdown("""
    **Name:** Alok Sahu  
    **Project:** Zomato Restaurant Popularity Prediction  
    **Model:** Random Forest Classifier  
    **Accuracy:** 97%  
    **ROC-AUC:** 0.99  

    Built using:
    - Python
    - Scikit-learn
    - TF-IDF
    - Streamlit
    - SHAP Explainability
    """)

    st.success("🚀 Built with passion for Machine Learning!")
