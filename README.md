# Medical Insurance Cost Prediction

A machine learning project to predict annual medical insurance charges for individuals based on demographic attributes, lifestyle factors, and physical health metrics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [Author](#author)

---

## Project Overview

This project builds an end-to-end ML regression pipeline to predict the medical insurance charges billed to a policyholder based on personal and lifestyle attributes. The dataset contains 1,338 records with features such as age, BMI, smoking status, number of dependents, gender, and region.

The target variable is `charges` — a continuous value representing the annual medical bill in USD. The pipeline covers the full ML workflow: data loading, missing value handling, categorical encoding, outlier treatment, feature scaling, feature selection, model training, and R² evaluation across 7 regression models.

---

## Repository Structure

```
medical-insurance-cost-prediction/
├── medical_insurance.ipynb       # End-to-end ML pipeline notebook
└── insurance.csv                 # Dataset (1,338 records)
```

---

## Dataset

**File:** `insurance.csv`  
**Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
**Total Records:** 1,338  
**Target Variable:** `charges` (annual medical bill in USD)

| Feature | Type | Description |
|---|---|---|
| `age` | Integer | Age of the primary beneficiary (18 – 64) |
| `sex` | Categorical | Gender of the policyholder (male / female) |
| `bmi` | Float | Body Mass Index (15.96 – 53.13) |
| `children` | Integer | Number of dependents covered (0 – 5) |
| `smoker` | Categorical | Whether the individual smokes (yes / no) |
| `region` | Categorical | US residential area (northeast / northwest / southeast / southwest) |
| `charges` | Float | Target — Annual medical bill in USD ($1,121 – $63,770) |

### Summary Statistics

| Feature | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| Age | 39.21 | 14.05 | 18 | 64 |
| BMI | 30.66 | 6.10 | 15.96 | 53.13 |
| Children | 1.09 | 1.21 | 0 | 5 |
| Charges ($) | 13,270 | 12,110 | 1,121 | 63,770 |

---

## ML Pipeline

### 1. Data Loading and Exploration

- Loaded dataset using `pandas`
- Inspected shape, data types, and missing value counts
- Reviewed feature distributions and target variable range

### 2. Missing Value Handling

- **Numerical columns** — imputed using mean via `SimpleImputer(strategy='mean')`
- **Categorical columns** — imputed using mode via `SimpleImputer(strategy='most_frequent')`

### 3. Categorical Encoding

- Applied **Label Encoding** on binary columns (`sex`, `smoker`) using `LabelEncoder`
- Applied **One-Hot Encoding** on multi-class column (`region`) using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity

### 4. Outlier Treatment

- Used **IQR (Interquartile Range)** method for all numerical columns
- Capped values below `Q1 - 1.5 * IQR` to the lower bound
- Capped values above `Q3 + 1.5 * IQR` to the upper bound

### 5. Feature Selection

- Applied `SelectKBest` with `f_regression` to identify the top 10 most relevant features correlated with `charges`
- Computed and sorted a correlation matrix by the target variable

### 6. Feature Scaling

- Applied `StandardScaler` to normalize all feature values to zero mean and unit variance before model training

### 7. Train-Test Split

| Parameter | Value |
|---|---|
| Test size | 30% |
| Train size | 70% |
| Random state | 100 |

---

## Model Results

All 7 regression models were trained on the same preprocessed dataset and evaluated using **R² Score (Coefficient of Determination)**.

| # | Model | Key Parameters |
|---|---|---|
| 1 | Linear Regression | Default |
| 2 | Decision Tree Regressor | Default |
| 3 | Random Forest Regressor | `n_estimators=100` |
| 4 | K-Nearest Neighbors Regressor | `n_neighbors=5` |
| 5 | Support Vector Regression (SVR) | Default (RBF kernel) |
| 6 | Gradient Boosting Regressor | Default |
| 7 | Ridge Regression | `alpha=1.0` |

> R² scores will be updated after running the notebook. R² closer to 1.0 indicates better model performance.

---

## Author

**Bremikha Arunachalam**  
GitHub: [github.com/BremikhaArunachalam](https://github.com/BremikhaArunachalam)
