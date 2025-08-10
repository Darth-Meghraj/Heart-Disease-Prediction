🫀 Heart Disease Risk Prediction using Machine Learning

## 📌 Overview
This is my **research internship project** developed at **Amity University Kolkata** under the guidance of **Dr. Anwesha Law**.  
The aim is to predict the **risk of heart disease** using patient medical data and **Machine Learning** techniques.

The project covers the **entire ML pipeline** — from dataset exploration, preprocessing, feature engineering, model training, evaluation, and finally **deployment as a Streamlit web app** for real-time predictions.

---

## 📊 Key Features
- **Data Preprocessing**:
  - Outlier capping for vital signs and lab results
  - Log transformation of skewed biomarkers (`kcm`, `troponin`)
  - Feature scaling for uniform input
  - Label encoding for categorical features
  - Class balancing with **SMOTE**
- **Model Training**:
  - Tested multiple models; **XGBoost** achieved the best results:
    - Accuracy: **93.13%**
    - Precision: **93.14%**
    - Recall: **93.12%**
    - F1-Score: **93.12%**
- **Visualization & EDA**:
  - Distribution plots, boxplots, and correlation analysis
  - Confusion matrix and feature importance plots
- **Deployment**:
  - Interactive **Streamlit** web application for live risk predictions
  - Raw user input processed internally before prediction

🚀 How to Run Locally:
 1. Clone the repository in the vs code terminal:
      git clone https://github.com/yourusername/heart-disease-prediction.git
      cd heart-disease-prediction

2. Install dependencies:
    pip install -r requirements.txt

3. Run the Streamlit app:
    streamlit run Streamlit_webApp.py

🌐 Live Demo
You can try the app online without installation here:
https://heart-disease-prediction-n8zvq6juscrszg3ypppxhf.streamlit.app/

