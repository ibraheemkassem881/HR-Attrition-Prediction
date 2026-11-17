# 📊 HR Employee Attrition Prediction System

An end-to-end Machine Learning and interactive web application built with **Python**, **Scikit-Learn**, and **Streamlit** to predict employee attrition risk and identify key retention drivers.

## 📌 Project Overview
Employee turnover is costly for organizations. This project analyzes HR data (311 employee records) to uncover patterns behind employee departures and provides HR managers with a real-time predictive tool to assess attrition risk and take proactive retention actions.

## 🔍 Key Findings & Feature Importance
Using a **Random Forest Classifier**, feature importance analysis revealed the primary drivers of employee attrition:
1. **Salary (~46.9%):** The single most critical factor influencing an employee's decision to leave.
2. **Employee Satisfaction (~14.7%):** Low job satisfaction significantly increases turnover risk.
3. **Attendance / Days Late Last 30 (~13.7%):** Frequent tardiness acts as an early behavioral warning sign.
4. **Marital Status (~10.9%) & Performance Score (~9.2%):** Secondary contributing factors.

## ⚙️ Technical Approach & Model Optimization
- **Handling Class Imbalance:** The dataset had fewer terminated employees compared to active ones. Initially, the model achieved a high overall accuracy (67%) but suffered from a very low Recall (0.14) for leavers.
- **Solution:** Applied `class_weight='balanced'` in the `RandomForestClassifier`, which more than doubled the model's ability to detect potential leavers (increasing Recall to **0.32**), prioritizing early risk detection for HR intervention.

## 🚀 Interactive Web App (Streamlit)
The trained model is deployed as an interactive dashboard where HR personnel can input employee metrics (Salary, Satisfaction, Tardiness, Performance Score, Marital Status, Gender) and instantly receive:
- **Attrition Risk Probability (%)**
- **Actionable HR Recommendations**

## 🛠️ Tech Stack
- **Language:** Python 3
- **Data Manipulation:** Pandas
- **Machine Learning:** Scikit-Learn (`RandomForestClassifier`)
- **Web Framework / UI:** Streamlit

## 💻 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```