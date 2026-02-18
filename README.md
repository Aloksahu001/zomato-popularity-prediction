# 🍽️ Zomato Restaurant Popularity Prediction

A Machine Learning web application that predicts whether a restaurant is popular or not based on customer engagement and restaurant attributes.

Built using:
- Python
- Scikit-Learn
- TF-IDF (NLP Feature Engineering)
- Random Forest Classifier
- Streamlit
- SHAP Explainability

---

## 🚀 Live Demo
🔗 https://zomato-popularity-prediction-kbsr2wrxmappkutyjdokv2a.streamlit.app/


---

## 📊 Problem Statement

Restaurants receive thousands of reviews and engagement data.
This project predicts whether a restaurant will be **Popular (1)** or **Not Popular (0)** based on:

- Votes
- Approx Cost for Two
- Online Order Availability
- Table Booking Availability
- Restaurant Type
- Cuisines

---

## 🧠 Machine Learning Approach

### 🔹 Feature Engineering
- TF-IDF Vectorization on:
  - `rest_type`
  - `cuisines`
- Numeric Features combined with Sparse NLP features
- Final Feature Size: 147+

### 🔹 Model Used
- RandomForestClassifier

### 🔹 Performance

Train Accuracy: **98.75%**  
Test Accuracy: **97.12%**  
Overfitting Gap: **1.62% (Very Stable Model)**

Confusion Matrix:

| Actual \ Predicted | 0 | 1 |
|--------------------|---|---|
| 0 | 5759 | 63 |
| 1 | 175 | 2287 |

---

## 📈 Visualization

- Popular vs Non-Popular Distribution
- Feature Importance Graph
- SHAP Explainability (Why the model predicted this)

---

## 🖥️ Web Application

Built using **Streamlit**

Features:
- Interactive Inputs
- Cuisine Dropdown List
- Editable Fields
- SHAP Explanation
- Feature Importance Graph
- About Developer Section

---

## 🛠️ Installation

```bash
git clone https://github.com/Aloksahu001/zomato-popularity-prediction.git
cd zomato-popularity-prediction
pip install -r requirements.txt
streamlit run app.py
