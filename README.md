# 🇮🇳 BharatPay: Predictive Analytics of Digital Payment Transactions

  **📊 Exploring Digital Payment Trends | 🤖 Predicting UPI Transactions | 📈 Discovering Payment Patterns**

---

## 🌟 Project Overview

**BharatPay** is a Python-based predictive analytics project focused on analysing Indian digital payment transactions, predicting daily UPI transaction volumes, evaluating machine learning models, and discovering payment activity patterns through unsupervised learning.

With the rapid growth of India's digital payment ecosystem, understanding transaction behaviour has become increasingly important. This project uses historical payment statistics to identify trends, explore relationships among payment methods, and develop predictive models for daily UPI transaction activity.

The project combines **Exploratory Data Analysis (EDA), Feature Engineering, Supervised Machine Learning, K-Means Clustering, and Principal Component Analysis (PCA)** to extract meaningful insights from real-world payment data.

## 🎯 Project Objectives

- 📊 Analyse daily UPI transaction trends and transaction values.
- 💳 Compare transaction activity across UPI, IMPS, FASTag, ATM (NFS), and AePS.
- 🔍 Identify relationships and correlations among different payment indicators.
- ⚙️ Perform feature engineering using historical lag variables and rolling statistics.
- 🤖 Develop and compare multiple machine learning regression models.
- 📈 Predict daily UPI transaction volumes using historical transaction patterns.
- 🧩 Discover payment activity patterns using K-Means clustering and PCA.

---

## 📂 Dataset Description

The project uses daily Indian payment-system statistics published by the **National Payments Corporation of India (NPCI)**.

| Dataset Attribute | Details |
|---|---|
| 📄 Dataset | NPCI Daily Payment Statistics |
| 📊 Total Records | 1,987 |
| 📑 Total Columns | 15 |
| 📅 Date Coverage | 1 May 2021 – 8 October 2026 |
| 🎯 Target Variable | UPI, number of transactions, daily |
| 📁 Data Format | Microsoft Excel (.xlsx) |
| 🏛️ Original Publisher | National Payments Corporation of India (NPCI) |
| 🔗 Compilation Source | Measure of India |

### 💳 Payment Systems Covered

The dataset contains daily transaction counts and transaction values for the following payment systems:

- **UPI:** Unified Payments Interface
- **IMPS:** Immediate Payment Service
- **AePS:** Aadhaar Enabled Payment System
- **FASTag (NETC):** National Electronic Toll Collection
- **ATM (NFS):** ATM Network
- **NACH Credit:** National Automated Clearing House Credit
- **NACH Debit:** National Automated Clearing House Debit

📌 **Dataset Reference:** [Measure of India – UPI and the Payment Networks](https://measure.thisindianlife.today/guides/upi/)

*Attribution: NPCI is the original publisher of the payment statistics. Measure of India is the compilation and download source used for this project.*

---

## 🛠️ Technologies & Tools Used

| Category | Technologies |
|---|---|
| 💻 Programming Language | Python |
| 🧮 Data Manipulation | Pandas, NumPy |
| 📊 Data Visualisation | Matplotlib, Seaborn |
| 🤖 Machine Learning | Scikit-learn |
| 📁 Dataset Handling | Microsoft Excel |
| 🧑‍💻 Development | Python IDE / Jupyter-compatible environment |

### 🤖 Machine Learning Algorithms

**📈 Regression Models**

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)

**🧩 Unsupervised Learning**

- K-Means Clustering
- Principal Component Analysis (PCA)

---

## 🔄 Project Workflow

The project follows an end-to-end predictive analytics workflow.

### 1️⃣ Data Collection & Preprocessing

- Imported daily payment statistics from an Excel workbook.
- Converted and validated date fields.
- Checked duplicate records and missing values.
- Sorted observations chronologically.
- Prepared numerical variables for analysis.

### 2️⃣ Exploratory Data Analysis (EDA) 📊

- Analysed daily UPI transaction-count trends.
- Examined daily UPI transaction values.
- Compared average transaction counts across payment methods.
- Generated a correlation heatmap to understand relationships among payment indicators.

### 3️⃣ Feature Engineering ⚙️

Created **50 engineered predictors**, consisting of:

| Feature Category | Features |
|---|---|
| 📅 Calendar Features | 5 |
| ⏳ Historical Lag Features | 42 |
| 📉 Rolling Statistical Features | 3 |
| **Total** | **50** |

**Calendar Features:**
- Year
- Month
- Day of Week
- Day of Year
- Weekend Indicator

**Lag Features:**

Created 1-day, 7-day, and 14-day lag variables for all 14 numerical payment indicators.

**Rolling Statistical Features:**
- 7-day rolling mean
- 14-day rolling mean
- 7-day rolling standard deviation

The prediction target is the day-to-day change in UPI transaction count. Predicted changes are added to the previous day's transaction count to reconstruct the predicted daily volume.

### 4️⃣ Predictive Modelling 🤖

- Applied a chronological 80:20 training and testing split.
- Trained five regression algorithms.
- Compared model performance using MAE, RMSE, and R².
- Evaluated the models against a previous-day baseline.
- Selected the best-performing model based on RMSE.

### 5️⃣ Payment Pattern Clustering 🧩

- Applied log transformation and feature standardisation.
- Evaluated candidate cluster counts using silhouette score and inertia.
- Selected a four-cluster K-Means solution.
- Used PCA to project clustered observations into two dimensions for visualisation.

---

## 🏆 Model Performance & Results

Five regression algorithms were evaluated to identify the most effective model for predicting daily UPI transaction volume.

### 📊 Performance Comparison

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| 🥇 Linear Regression | 13,408,095.84 | 17,904,761.18 | **0.8795** |
| 🥈 Random Forest | 15,099,042.24 | 20,181,798.99 | 0.8469 |
| 🥉 Support Vector Regression | 14,595,619.38 | 20,434,097.90 | 0.8430 |
| Gradient Boosting | 20,236,368.85 | 25,461,164.68 | 0.7563 |
| Decision Tree | 21,992,259.25 | 28,184,899.03 | 0.7014 |

### 🥇 Best Performing Model: Linear Regression

Linear Regression achieved the best overall performance among the five evaluated regression models.

| Evaluation Metric | Result |
|---|---:|
| MAE | 13,408,095.84 |
| RMSE | 17,904,761.18 |
| R² Score | **0.8795 (87.95%)** |

The model achieved an R² score of approximately 87.95%, indicating that it explained a substantial proportion of the variation in the test-set target.

The results also demonstrate that a relatively simple regression algorithm can outperform more complex models for this particular dataset and evaluation period.

📌 *Note: MAE and RMSE are expressed in transaction-count units. Results are based on the chronological test set and do not guarantee future prediction performance.*

---

## 📈 Project Visualisations

The project includes **12 visualisations** covering exploratory analysis, predictive modelling, model evaluation, and clustering.

All visualisations are available in the `outputs/figures/` directory.

### 📊 Exploratory Data Analysis

| No. | Visualisation |
|---|---|
| 01 | Daily UPI Transaction Trend |
| 02 | Daily UPI Transaction Value |
| 03 | Payment Method Comparison |
| 04 | Correlation Heatmap |

### 🤖 Predictive Modelling & Evaluation

| No. | Visualisation |
|---|---|
| 05 | Actual vs Predicted Transactions |
| 06 | Residual Distribution |
| 07 | Actual vs Predicted Scatter Plot |
| 08 | R² Score Comparison |
| 09 | RMSE Comparison |
| 10 | MAE Comparison |

### 🧩 Clustering & Dimensionality Reduction

| No. | Visualisation |
|---|---|
| 11 | K-Means Clusters using PCA |
| 12 | Cluster Distribution |

---

## 💡 Key Learnings

Through this project, I gained practical experience in:

- 📊 Exploratory Data Analysis and data preprocessing.
- ⚙️ Feature engineering for time-dependent data.
- 🤖 Implementing and comparing machine learning regression algorithms.
- 📈 Evaluating predictive models using MAE, RMSE, and R².
- 🧩 Applying K-Means clustering and PCA.
- 🔍 Interpreting model results and communicating data-driven findings.
- 🚀 Developing an end-to-end predictive analytics workflow using Python.

---

## 🔮 Limitations & Future Scope

### ⚠️ Current Limitations

- The dataset does not explicitly include holidays, festivals, policy changes, outages, or macroeconomic indicators.
- Model evaluation uses one chronological train-test split.
- Correlation and clustering results are exploratory and do not establish causal relationships.

### 🚀 Future Enhancements

- Incorporate holiday and festival indicators.
- Explore additional external variables affecting digital payment activity.
- Apply rolling-origin time-series validation.
- Experiment with advanced time-series forecasting techniques.
- Develop an interactive dashboard for monitoring payment trends and predictions.

---

## 🙏 Acknowledgement

I would like to express my sincere gratitude to **Ms. Maneet Kaur, Assistant Professor, Department of Computer Science and Engineering, Lovely Professional University**, for her guidance and academic support throughout this project.

I also acknowledge **NPCI** for publishing the payment-system statistics and **Measure of India** for providing the compiled dataset used in this study.

---

## 👩‍💻 Author

**Mitali**  
🎓 B.Tech – Computer Science and Engineering  
🏛️ Lovely Professional University, Punjab, India

---

  ⭐ **If you find this project interesting, feel free to explore the repository!**

  **🇮🇳 BharatPay – Turning Digital Payment Data into Predictive Insights.**
