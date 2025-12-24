# AutoML-Advisor

AutoML-Advisor is an intelligent, dataset-agnostic Streamlit application that performs deep Exploratory Data Analysis (EDA) and provides theory-backed machine learning guidance — before a single model is trained.

Unlike AutoML tools that jump straight to modeling, AutoML-Advisor thinks like a data scientist first.

# What AutoML-Advisor Does

AutoML-Advisor analyzes your dataset step-by-step and answers the most important ML questions:

# 1. Data Understanding

Dataset shape & structure

Numeric vs categorical feature detection

Automatic ML task detection (Regression / Classification)

# 2. Distribution Intelligence (Core Strength)

For each selected variable, the app analyzes:

Central Tendency

Mean, Median, Mode

Spread

Range, Standard Deviation, IQR

Shape

Skewness

Kurtosis

Visual Analysis

Histogram

KDE

Boxplot

Quantiles

Q1, Median, Q3

Percentiles

 Includes beginner-friendly theory explaining why each concept matters.

# 3. Missing Value Intelligence

Instead of blindly filling or dropping missing data, AutoML-Advisor decides why values are missing and suggests the best action:

- Drop columns (too sparse, weak signal)

- Transform features (missing = absence of event)

- Impute using statistical / ML-friendly strategies

- Explains when missing values are actually informative.

# 4. Outlier Intelligence

Automatically detects whether a feature is continuous or categorical

Applies IQR & Z-Score methods only where valid

Detects skewness before suggesting removal

Advises:

cap/remove

transform (log / power)

ignore (tree-based models)

- Prevents one of the most common ML mistakes.

# 5. Feature Selection Guidance

Identifies features that usually hurt model performance:

ID columns

Extremely high-cardinality features

Near-constant features

- Explains bias, noise, and overfitting risks.

# 6. Feature Engineering Suggestions

AutoML-Advisor never leaves this empty.

It dynamically suggests:

Binary features from missingness

Aggregations

Ratios & rates

Encoding strategies (OHE / ordinal)

Scaling / normalization when required

- Teaches feature thinking, not just coding.

# 7. ML Model Recommendations

Based on:

Target type

Distribution

Feature mix

Outliers

Missing data patterns

It recommends Top 5 ML models and explains why each fits the dataset.

Examples:

Random Forest

XGBoost / LightGBM

CatBoost

Logistic / ElasticNet

SVM (when appropriate)

# 8. ML Readiness Score

A final ML Readiness Score (0–100) based on:

Missing value severity

Feature quality

Distribution health

Outlier risk

Clearly answers:

“Can I train a reliable ML model on this data?”

# Design Philosophy

“EDA before ML, understanding before automation.”

AutoML-Advisor:

- Does NOT auto-train black-box models

✅ Teaches why decisions are made

✅ Works with any CSV

✅ Avoids dataset-specific hardcoding

This makes it ideal for:

Students

Researchers

Data analysts

Early-stage ML projects

# Tech Stack

Python

Streamlit

Pandas, NumPy

Matplotlib, Seaborn

SciPy

Scikit-learn (for logic, not pipelines)

# Project Structure
AutoML-Advisor/
├── app/
│   └── app.py
├── requirements.txt
├── README.md

# Author
Pranjal Tyagi
B.Tech CSE (AI & DS)
