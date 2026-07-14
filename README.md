# Intelligent EDA and ML Decision System

This project is a data-centric machine learning support system that performs structured exploratory data analysis, data quality checks, and guided preprocessing decisions, followed by baseline model validation.

The goal is to reduce common mistakes in early ML workflows by making data decisions explicit, explainable, and verifiable.

# Motivation

In real-world machine learning, model performance is often limited by:

poor handling of missing values

inappropriate feature selection

unnoticed data leakage

lack of validation after preprocessing

This system focuses on decision quality before model complexity.

# Core Functionality
Dataset Overview

Row and column statistics

Missing value summary

Automatic task type detection (classification or regression)

Target Analysis

Distribution visualization

Skewness analysis

Guidance on transformation impact

Distribution Analysis (Feature-Level)

For any selected feature:

Variable type detection

Central tendency (mean, median, mode)

Spread (range, IQR, standard deviation)

Shape (skewness, kurtosis)

Visual inspection (histogram, boxplot)

Missing Value Intelligence

Column-wise missing percentage detection

Data-type aware recommendations

Strategy ranking based on skewness, missing rate, and cardinality

Detection of semantic missing values (e.g. "Missing", "Unknown")

Human-Approved Action

Apply recommended imputation strategies

Dataset is updated persistently using session state

Clear feedback when missing values are resolved

Outlier Intelligence

IQR-based and Z-score analysis

Skewness-aware recommendations

Strategy ranking (transform, cap, remove)

Prevents incorrect outlier handling on discrete or ID columns

Feature Selection

Automatically identifies:

ID and index-like columns

Constant features

Extremely high-cardinality features

Human-Approved Action

Drop low-value features

Dataset structure updates dynamically

Feature Predictive Power (Model-Free)

Uses mutual information, correlation, and entropy reduction

No model training required

Ranks features by contribution to the target

Data Leakage Detection

Detects suspiciously high correlation with target

Flags target-derived column names

Highlights post-event information risks

Baseline Model Validation

After preprocessing actions, a baseline model is trained to validate whether the dataset quality has improved.

Algorithm: Linear Regression (baseline)

Purpose: validation, not optimization

Metric: R² (regression) or accuracy (classification)

Displayed outputs:

Model used

Evaluation metric and score

Top contributing features

Validation message indicating whether agent decisions improved the dataset

ML Readiness Score

A dynamic score representing how suitable the dataset is for machine learning.

Factors considered:

Remaining missing values

Feature quality

Target skewness

The score updates automatically after preprocessing actions such as imputation and feature selection.

# Technology Stack

Python

Streamlit

Pandas, NumPy

Scikit-learn

SciPy

Matplotlib, Seaborn

Intended Use

# Understanding real-world EDA workflows

Learning data-driven ML decision making

Preparing datasets before model training

Portfolio project for AI/ML internships

# Project Status

Semi-agent system with human-approved actions implemented

Baseline validation workflow completed

Future work includes cross-validation comparison and dataset export
# Future RoadMap
I will implement this Basic EDA system to advanced EDA and also add multiple agents for work
Future Upgraded Version===  # Autonomous Data Scientist

# Author

Pranjal Tyagi
B.Tech CSE (AI & Data Science)
IIIT Kottayam
B.Tech CSE (AI & DS)
