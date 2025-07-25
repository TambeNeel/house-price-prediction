# house-price-prediction
This project solves the Kaggle House Prices competition using a blend of Ridge, Lasso, and XGBoost models. It features detailed EDA, skew handling, feature engineering, hyperparameter tuning, and log-transformed target regression. Final predictions are blended and submitted using RMSE on the log scale.

# 🏠 House Prices - Advanced Regression Techniques

A complete machine learning pipeline to predict residential home prices using Ridge, Lasso, and XGBoost. Built for the Kaggle [House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

---

## 📊 Problem Statement
Predict the final sale price of homes in Ames, Iowa using 79 explanatory variables. The competition uses **Root Mean Squared Error (RMSE)** on the **log of SalePrice** as the evaluation metric.

---

## 🚀 Approach & Pipeline Overview

This project follows a structured end-to-end pipeline for predictive modeling:

### 1. 🔍 Exploratory Data Analysis (EDA)
- Visualized `SalePrice` distribution and applied log transform
- Analyzed missing values using heatmaps and barplots
- Checked correlations to identify top predictive features

### 2. 🧹 Data Cleaning & Preprocessing
- Removed outliers based on IQR from key numerical features
- Applied `log1p` to correct skew in numerical columns
- Converted mostly-zero features (e.g., `PoolArea`) to binary flags

### 3. 🏗️ Feature Engineering
- Created features like `TotalSF`, `HouseAge`, `Remodeled`, and `TotalBath`
- Used manual ordinal encoding for quality-related features
- One-hot encoded all remaining categorical variables

### 4. ⚖️ Feature Scaling
- Applied `StandardScaler` to numerical features for linear models

### 5. 🤖 Model Building
- Evaluated with cross-validation (log-RMSE):
  - Linear Regression
  - Ridge Regression (α=10)
  - Lasso Regression (α=0.001)
  - XGBoost (with hyperparameter tuning via GridSearchCV)
- Trained models on full dataset after validation

### 6. 🔀 Blending
- Averaged predictions from Ridge, Lasso, and XGBoost
- Applied `expm1()` to revert log-transformed predictions

### 7. 📤 Submission
- Final output saved as `submission.csv` in the correct Kaggle format

---

## 📝 Files

- `notebooks/House_Price.ipynb`: Complete pipeline from EDA to submission
- `submission/submission.csv`: Final blended predictions
- `requirements.txt`: All Python packages used
- `all_model_predictions.csv`: Individual predictions from Ridge, Lasso, XGBoost

---

## 📈 Evaluation Metric

> Root Mean Squared Error on log-transformed SalePrice

```python
np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
