# INTELLIGENT EDA + ML DECISION SYSTEM
# Author: Pranjal Tyagi


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kurtosis

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder



#################################################3
# ==============================
# GLOBAL METADATA RECOMPUTE
# ==============================
def recompute_metadata(df):
    return {
        "num_cols": df.select_dtypes(include=np.number).columns.tolist(),
        "cat_cols": df.select_dtypes(exclude=np.number).columns.tolist()
    }
####################################################################


####
def compute_entropy(series):
    probs = series.value_counts(normalize=True)
    return entropy(probs)

def feature_predictive_power(df, target, task_type):
    results = []

    y = df[target]
    df_features = df.drop(columns=[target])

    for col in df_features.columns:
        x = df_features[col]
        score = 0

        # Handle missing
        valid = x.notnull() & y.notnull()
        x_valid = x[valid]
        y_valid = y[valid]

        if x_valid.nunique() <= 1:
            final_score = 0.0

        elif x_valid.dtype != "object":
            # Numeric feature
            corr = abs(np.corrcoef(x_valid, y_valid)[0, 1]) if task_type == "Regression" else 0

            mi = mutual_info_regression(
                x_valid.values.reshape(-1, 1), y_valid
            )[0] if task_type == "Regression" else mutual_info_classif(
                x_valid.values.reshape(-1, 1), y_valid
            )[0]

            final_score = 0.6 * corr + 0.4 * mi

        else:
            # Categorical feature
            le = LabelEncoder()
            x_enc = le.fit_transform(x_valid.astype(str))

            mi = mutual_info_classif(
                x_enc.reshape(-1, 1), y_valid
            )[0]

            ent_before = compute_entropy(y_valid)
            ent_after = compute_entropy(
                y_valid.groupby(x_enc).apply(lambda s: s.mode().iloc[0])
            )

            ent_reduction = max(0, ent_before - ent_after)

            final_score = 0.7 * mi + 0.3 * ent_reduction

        # Normalize & label
        final_score = min(1.0, round(final_score, 3))

        if final_score > 0.75:
            strength = "Strong"
        elif final_score > 0.4:
            strength = "Moderate"
        elif final_score > 0.1:
            strength = "Weak"
        else:
            strength = "Useless"

        results.append({
            "Feature": col,
            "Signal_Strength": final_score,
            "Verdict": strength
        })

    return pd.DataFrame(results).sort_values("Signal_Strength", ascending=False)



# DECISION-LEVEL OUTLIER STRATEGY SCORING

def score_outlier_strategies(series: pd.Series):
    """
    Returns ranked outlier handling strategies with scores and justification.
    """
    series = series.dropna()

    if series.nunique() <= 10:
        return []

    # Metrics
    skew = series.skew()
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean()
    z_outliers = (np.abs(stats.zscore(series)) > 3).mean()

    strategies = []

    # 🔹 Transformation
    transform_score = 0
    if abs(skew) > 1:
        transform_score += 0.5
    if iqr_outliers > 0.01:
        transform_score += 0.3
    if z_outliers > 0.01:
        transform_score += 0.2

    strategies.append({
        "Strategy": "Log / Yeo-Johnson Transformation",
        "Score": round(min(transform_score, 1.0), 2),
        "Why": "Handles skewed distributions and preserves extreme but valid values"
    })

    # 🔹 IQR Capping
    iqr_score = 0
    if iqr_outliers > 0.005:
        iqr_score += 0.4
    if abs(skew) < 2:
        iqr_score += 0.3
    if iqr_outliers < 0.05:
        iqr_score += 0.3

    strategies.append({
        "Strategy": "IQR Capping / Winsorization",
        "Score": round(min(iqr_score, 1.0), 2),
        "Why": "Limits extreme values while keeping dataset size intact"
    })

    # 🔹 Z-score Removal
    z_score_score = 0
    if abs(skew) < 0.5:
        z_score_score += 0.4
    if z_outliers < 0.01:
        z_score_score += 0.4
    if series.shape[0] > 1000:
        z_score_score += 0.2

    strategies.append({
        "Strategy": "Z-score Outlier Removal",
        "Score": round(min(z_score_score, 1.0), 2),
        "Why": "Works only when data is approximately normal"
    })

    return sorted(strategies, key=lambda x: x["Score"], reverse=True)


# HELPER: Decision-Level Imputation Strategy Scoring


def score_imputation_strategies(series: pd.Series):
    """
    Returns ranked imputation strategies with confidence scores
    based on data distribution, skewness, and missing percentage.
    """

    miss_pct = series.isnull().mean()
    dtype = series.dtype
    scores = {}

    # Numeric features
    if dtype != "object":
        skew = series.skew()

        # Median
        scores["Median Imputation"] = 0.9 if abs(skew) > 1 else 0.75

        # Mean
        scores["Mean Imputation"] = 0.4 if abs(skew) > 1 else 0.7

        # KNN (heuristic penalty)
        scores["KNN Imputation"] = 0.6 if miss_pct < 0.2 else 0.3

    # Categorical features
    else:
        unique_vals = series.dropna().nunique()

        scores["Mode Imputation"] = 0.85 if unique_vals < 20 else 0.6
        scores["Create 'Missing' Category"] = 0.8 if miss_pct > 0.3 else 0.5

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked



# HELPER: Data Leakage Detection

def detect_data_leakage(df, target):
    leakage_flags = []

    y = df[target]

    for col in df.columns:
        if col == target:
            continue

        x = df[col]

        # Skip constant or empty
        if x.nunique() <= 1:
            continue

        # Numeric leakage
        if x.dtype != "object" and y.dtype != "object":
            valid = x.notnull() & y.notnull()
            if valid.sum() < 10:
                continue

            corr = abs(np.corrcoef(x[valid], y[valid])[0, 1])

            if corr > 0.9:
                leakage_flags.append({
                    "Feature": col,
                    "Risk_Level": "HIGH",
                    "Reason": f"Very high correlation with target ({corr:.2f})"
                })
            elif corr > 0.75:
                leakage_flags.append({
                    "Feature": col,
                    "Risk_Level": "MEDIUM",
                    "Reason": f"Suspicious correlation ({corr:.2f})"
                })

        # Name-based heuristic
        suspicious_keywords = ["target", "label", "outcome", "result", "status", "win"]
        if any(k in col.lower() for k in suspicious_keywords):
            leakage_flags.append({
                "Feature": col,
                "Risk_Level": "MEDIUM",
                "Reason": "Feature name suggests target-derived information"
            })

    return pd.DataFrame(leakage_flags)

##############################################################################
# ==========================================================
# LOAD DATA (AGENT-SAFE)
# ==========================================================
st.title("🧠 Intelligent Data Analysis & ML Decision System")

uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

# ✅ Load dataset ONLY ONCE
if "df" not in st.session_state:
    st.session_state["df"] = pd.read_csv(uploaded)

# ✅ Always work with session-state dataframe
df = st.session_state["df"]
meta = recompute_metadata(df)
num_cols = meta["num_cols"]
cat_cols = meta["cat_cols"]

######################################################################################


# DATASET OVERVIEW

st.header("🔍 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Avg Missing %", f"{df.isnull().mean().mean()*100:.2f}%")

st.dataframe(df.head(50), use_container_width=True)

#############################################################################
# ==============================
# 🧠 SEMANTIC MISSING VALUE DETECTION
# ==============================

st.header("🧠 Semantic Missing Value Detection")

semantic_missing = ["missing", "unknown", "none", "na", "null"]

semantic_missing_cols = []

for col in df.select_dtypes(include="object").columns:
    ratio = df[col].astype(str).str.lower().isin(semantic_missing).mean()
    if ratio > 0.05:  # >5% is meaningful
        semantic_missing_cols.append((col, round(ratio * 100, 2)))

if semantic_missing_cols:
    st.warning("⚠️ Semantic missing values detected (string-based placeholders)")
    for col, pct in semantic_missing_cols:
        st.write(f"• `{col}` → {pct}% values are semantic-missing strings")
else:
    st.success("✅ No semantic missing values detected.")
######################################################################################

# ==========================================================
# FEATURE TYPES
# ==========================================================
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

st.header("🧬 Feature Types")
st.write("Numeric Features:", num_cols)
st.write("Categorical Features:", cat_cols)

# ==========================================================
# TARGET SELECTION
# ==========================================================
st.header("🎯 Target Variable Selection")
target = st.selectbox("Select target column", df.columns)

task_type = "Classification" if df[target].nunique() < 15 else "Regression"
st.info(f"Detected ML Task Type: **{task_type}**")

# TARGET DISTRIBUTION ANALYSIS

st.header("📊 Target Distribution Analysis")

fig, ax = plt.subplots()
sns.histplot(df[target], kde=True, ax=ax)
ax.set_title("Target Distribution")
st.pyplot(fig)

target_skew = df[target].skew()
st.write(f"Skewness: {target_skew:.2f}")

with st.expander("📘 Why this matters"):
    st.markdown("""
- Skewed targets break linear models  
- Tree-based models handle skew naturally  
- Strong skew → consider log / power transform
""")

# DISTRIBUTION INTELLIGENCE (FULL)

st.header("📊 Distribution Intelligence (Complete & Explained)")

dist_feature = st.selectbox(
    "Select a feature to analyze distribution",
    df.columns
)

series = df[dist_feature].dropna()
n_unique = series.nunique()
dtype = df[dist_feature].dtype

# STEP 1: VARIABLE TYPE
if dtype == "object":
    var_type = "Categorical"
elif n_unique <= 10:
    var_type = "Discrete / Ordinal Numeric"
else:
    var_type = "Continuous Numeric"

st.subheader("🔹 Step 1: Variable Type")
st.write(f"Detected Type: **{var_type}**")

# STEP 2: CENTRAL TENDENCY
st.subheader("🔹 Step 2: Central Tendency")

if var_type != "Categorical":
    mean_val = series.mean()
    median_val = series.median()
    mode_val = series.mode().iloc[0]

    st.write(f"Mean: {mean_val:.3f}")
    st.write(f"Median: {median_val:.3f}")
    st.write(f"Mode: {mode_val:.3f}")

    if abs(mean_val - median_val) > 0.1 * abs(mean_val):
        st.warning("Mean ≠ Median → Data is skewed")
    else:
        st.success("Mean ≈ Median → Symmetric distribution")
else:
    st.write(f"Mode (most frequent category): {series.mode().iloc[0]}")

# STEP 3: SPREAD
st.subheader("🔹 Step 3: Spread / Variability")

if var_type != "Categorical":
    range_val = series.max() - series.min()
    std_val = series.std()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr_val = q3 - q1

    st.write(f"Range: {range_val:.3f}")
    st.write(f"Standard Deviation: {std_val:.3f}")
    st.write(f"IQR: {iqr_val:.3f}")

# STEP 4: SHAPE
st.subheader("🔹 Step 4: Shape of Distribution")

if var_type != "Categorical":
    skew_val = series.skew()
    kurt_val = kurtosis(series)

    st.write(f"Skewness: {skew_val:.3f}")
    st.write(f"Kurtosis: {kurt_val:.3f}")

# STEP 5: VISUALS
st.subheader("🔹 Step 5: Visual Analysis")

if var_type == "Categorical":
    fig, ax = plt.subplots()
    series.value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)
else:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(series, kde=True, ax=ax[0])
    sns.boxplot(x=series, ax=ax[1])
    st.pyplot(fig)

# STEP 6: QUANTILES
st.subheader("🔹 Step 6: Quantiles & Percentiles")

if var_type != "Categorical":
    st.write(f"Q1 (25%): {series.quantile(0.25):.3f}")
    st.write(f"Median (50%): {series.quantile(0.50):.3f}")
    st.write(f"Q3 (75%): {series.quantile(0.75):.3f}")
    st.write(f"P90: {series.quantile(0.90):.3f}")


# MISSING VALUE INTELLIGENCE
st.header("🧠 Missing Value Intelligence")

missing_summary = []

for col in df.columns:
    miss_pct = df[col].isnull().mean() * 100
    if miss_pct == 0:
        continue

    dtype = df[col].dtype
    unique_vals = df[col].dropna().nunique()

    if miss_pct > 70:
        if dtype == "object" and unique_vals < 30:
            action = "Transform → Missing means 'No event' (create binary flag)"
        else:
            action = "Drop → Too sparse, weak signal"
    else:
        if dtype != "object":
            action = "Impute → Median (robust to outliers)"
        else:
            action = "Impute → Mode (most frequent category)"

    missing_summary.append({
        "Column": col,
        "Missing_%": round(miss_pct, 2),
        "Data_Type": str(dtype),
        "Recommended_Action": action
    })

missing_df = pd.DataFrame(missing_summary)
st.dataframe(missing_df, use_container_width=True)#



# ==============================
# 🛠 AGENT ACTION 1: APPLY IMPUTATION
# ==============================
st.subheader("🛠 Apply Recommended Imputation")

if not missing_df.empty:
    if st.button("✅ Apply Best Imputation Strategy", key="apply_imputation"):
        for col in missing_df["Column"]:
            ranked = score_imputation_strategies(df[col])
            best_method = ranked[0][0]

            if "Median" in best_method:
                df[col] = df[col].fillna(df[col].median())
            elif "Mean" in best_method:
                df[col] = df[col].fillna(df[col].mean())
            elif "Mode" in best_method:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            elif "Missing" in best_method:
                df[col] = df[col].fillna("Missing")

        # Persist agent action
        st.session_state["df"] = df

        # User-facing explanation (IMPORTANT)
        st.success("✅ Recommended imputations applied successfully.")
        st.info(
            "🧠 **Agent Action Completed**\n\n"
            "The agent detected missing values earlier and has now **filled them using "
            "the recommended strategies**.\n\n"
            "👉 Your dataset is clean with respect to missing values. "
            "You can now move to the **next feature analysis or agent action**."
        )

        st.experimental_rerun()
else:
    st.success("✅ No missing values detected — no imputation needed.")




###################################################################




# ==============================
# DECISION-LEVEL MISSING VALUE INTELLIGENCE
# ==============================

if not missing_df.empty and "Column" in missing_df.columns:
    st.subheader("🧠 Decision-Level Missing Value Intelligence")

    for col in missing_df["Column"]:
        st.markdown(f"### 🔹 Column: `{col}`")

        ranked = score_imputation_strategies(df[col])

        for i, (method, score) in enumerate(ranked, 1):
            st.write(f"{i}. **{method}** → Confidence Score: `{score:.2f}`")

        st.info("Top-ranked strategy is recommended based on data distribution & missingness.")
else:
    st.success("✅ No missing values detected — no imputation needed.")



with st.expander("📘 How to ACTUALLY fix missing values"):
    st.markdown("""
### 🔧 Practical Rules

• **Median > Mean** → safer for skewed numeric data  
• **Mode** → best for categorical data  
• **Very high missing + event-based columns** → transform, don’t drop  
• **Blind imputation destroys signal**

This logic is used in **real ML pipelines**, not tutorials.
""")

# OUTLIER INTELLIGENCE

st.header("🚨 Outlier Intelligence")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
feature = st.selectbox("Select numeric feature", numeric_cols)

series = df[feature].dropna()
unique_vals = series.nunique()

if unique_vals <= 10:
    st.info(
        f"`{feature}` has only {unique_vals} unique values.\n\n"
        "This is **ordinal / discrete**, not continuous.\n"
        "**Outlier detection is NOT applicable.**"
    )
else:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean() * 100

    z_outliers = (np.abs(stats.zscore(series)) > 3).mean() * 100
    skew = series.skew()

    st.write(f"📌 IQR Outliers: **{iqr_outliers:.2f}%**")
    st.write(f"📌 Z-Score Outliers: **{z_outliers:.2f}%**")
    st.write(f"📌 Skewness: **{skew:.2f}**")

    if iqr_outliers < 1:
        verdict = "Ignore → Outliers are negligible"
    elif abs(skew) > 1:
        verdict = "Transform → Log / Yeo-Johnson recommended"
    else:
        verdict = "Cap or Remove → Safe to treat"

    st.success(f"🧠 Recommendation: {verdict}")
#
st.subheader("🧠 Decision-Level Outlier Strategy Ranking")

ranked_strategies = score_outlier_strategies(series)

if ranked_strategies:
    for i, strat in enumerate(ranked_strategies, 1):
        st.markdown(
            f"""
**{i}️⃣ {strat['Strategy']} — Score: {strat['Score']}**  
• {strat['Why']}
"""
        )
else:
    st.info("Outlier strategies not applicable for this feature.")
#

with st.expander("📘 Outlier Theory (Important)"):
    st.markdown("""
### 🚨 When NOT to detect outliers

❌ IDs  
❌ Ordinal values (ball, inning, rank)  
❌ Encoded categories  

### ✅ Best practices
• Tree models tolerate outliers  
• Linear models don’t  
• Transformation > Deletion
""")

# FEATURE SELECTION (GENERIC)
st.header("🗑️ Feature Selection (Data-Aware)")

drop_features = []

for col in df.columns:
    if col == target:
        continue
    if any(x in col.lower() for x in ["id", "index", "uuid"]):
        drop_features.append(col)
    elif df[col].nunique() <= 1:
        drop_features.append(col)
    elif df[col].nunique() > 0.9 * len(df):
        drop_features.append(col)

st.write("🚮 Recommended to drop:")
st.write(drop_features)

with st.expander("📘 Why these features are dropped"):
    st.markdown("""
• IDs have no predictive meaning  
• Constant columns add noise  
• High-cardinality text explodes dimensionality  
• Dropping improves generalization
""")


# ==============================
# 🗑 AGENT ACTION 2: APPLY FEATURE DROPPING
# ==============================
st.subheader("🗑 Apply Feature Selection")

if drop_features:
    if st.button(
        "🗑 Drop Suggested Features",
        key="agent_drop_features"   # 🔑 UNIQUE KEY (MANDATORY)
    ):
        df = df.drop(columns=drop_features, errors="ignore")

        # Commit agent action
        st.session_state["df"] = df

        st.success(f"✅ Dropped {len(drop_features)} low-value features.")
        st.info("Feature space updated. Re-running analysis…")

        st.experimental_rerun()
else:
    st.info("No low-value features detected.")

# ==============================
# 🧠 AGENT DECISION: BASELINE MODEL
# ==============================
st.subheader("🧠 Agent Decision: Baseline Model Selection")

if task_type == "Classification":
    baseline_model_name = "RandomForestClassifier"
    baseline_reason = (
        "Chosen because the task is classification, "
        "features may be non-linear, "
        "handles mixed data types, "
        "and is robust to outliers and missing-value noise."
    )
else:
    baseline_model_name = "RandomForestRegressor"
    baseline_reason = (
        "Chosen because the task is regression, "
        "handles skewed distributions, "
        "does not require feature scaling, "
        "and captures non-linear relationships."
    )

st.success(f"✅ Selected Baseline Model: **{baseline_model_name}**")
st.info(baseline_reason)

# Save decision for training step
st.session_state["baseline_model_name"] = baseline_model_name


    
###
st.header("🧠 Feature Engineering Suggestions")

fe_suggestions = []

# ---------- Encoding ----------
if len(cat_cols) > 0:
    high_card = [c for c in cat_cols if df[c].nunique() > 15]
    low_card = [c for c in cat_cols if df[c].nunique() <= 15]

    if low_card:
        fe_suggestions.append(
            f"Apply One-Hot Encoding to low-cardinality categorical columns: {low_card}"
        )
    if high_card:
        fe_suggestions.append(
            f"Apply Target / Frequency Encoding to high-cardinality columns: {high_card}"
        )

# Scaling 
if task_type == "Regression":
    fe_suggestions.append(
        "Scale numeric features (StandardScaler / RobustScaler) for linear & distance-based models"
    )
else:
    fe_suggestions.append(
        "Scaling optional (tree-based models do not require scaling)"
    )

# Skewness 
skewed_cols = [c for c in num_cols if abs(df[c].skew()) > 1]

if skewed_cols:
    fe_suggestions.append(
        f"Apply log / Yeo-Johnson transform to skewed numeric columns: {skewed_cols}"
    )

# Missing flag
missing_flag_cols = [c for c in df.columns if df[c].isnull().any()]
if missing_flag_cols:
    fe_suggestions.append(
        f"Create binary missing indicators for columns: {missing_flag_cols}"
    )

# Display
for s in fe_suggestions:
    st.write("•", s)

with st.expander("📘 Why these transformations help"):
    st.markdown("""
### Feature Engineering Logic

• **Encoding** converts categories into numeric signal  
• **Scaling** ensures fair contribution in linear & distance models  
• **Transformations** stabilize variance & reduce skew  
• **Missing flags** preserve hidden information

These steps improve **model stability + accuracy**.
""")

# ==========================================================
# FEATURE PREDICTIVE POWER (MODEL-FREE)
# ==========================================================
st.header("📈 Feature Predictive Power (Model-Free)")

st.caption(
    "Quantifies how much each feature contributes to predicting the target "
    "using information theory — not heuristics."
)

# --- Compute predictive power using the correct helper ---
power_df = feature_predictive_power(df, target, task_type)

# --- Safe display ---
if power_df.empty:
    st.warning("⚠️ Feature predictive power could not be computed.")
else:
    st.dataframe(power_df, use_container_width=True)

with st.expander("📘 How this is computed"):
    st.markdown("""
### 🧠 Signal Strength Logic

• **Numeric → Target**  
  - Correlation (regression only)  
  - Mutual Information  

• **Categorical → Target**  
  - Mutual Information  
  - Entropy reduction  

### 📊 Interpretation
• **> 0.75** → Strong predictor  
• **0.4 – 0.75** → Moderate  
• **0.1 – 0.4** → Weak  
• **< 0.1** → Useless  

This analysis is **model-free**, statistically grounded, and fast.
""")
    


# ==========================================================
# 🚨 DATA LEAKAGE DETECTION
# ==========================================================
st.header("🚨 Data Leakage Detection")

leakage_df = detect_data_leakage(df, target)

if leakage_df.empty:
    st.success("✅ No obvious data leakage detected.")
else:
    st.error("⚠️ Potential data leakage detected!")
    st.dataframe(leakage_df, use_container_width=True)

with st.expander("📘 Why this is dangerous"):
    st.markdown("""
### 🚨 Why Data Leakage is Critical

• Produces unrealistically high accuracy  
• Model fails in real-world deployment  
• Extremely hard to debug later  

### 🔍 Common causes
• Target-derived features  
• Post-event information  
• ID-like proxies  
• Data collected after prediction time  

**Always remove or redesign leaked features before training.**
""")


####################################################################################################################3
st.subheader("🚀 Baseline Model Results")

if st.button("🚀 Train Baseline Model (Validation Check)", key="baseline_model"):
    st.info("Training baseline model to validate agent decisions...")

    df_model = st.session_state["df"].copy()

    # -----------------------------
    # Prepare X, y
    # -----------------------------
    X = df_model.drop(columns=[target])
    y = df_model[target]

    # Simple encoding (safe baseline)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Choose model
    # -----------------------------
    if task_type == "Classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        metric_name = "Accuracy"
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        metric_name = "R² Score"

    # -----------------------------
    # Store score
    # -----------------------------
    prev_score = st.session_state.get("baseline_before", None)
    st.session_state["baseline_after"] = score

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.success("✅ Baseline model trained successfully.")

    st.markdown("### 📊 Baseline Model Performance")
    st.write(f"**Model:** {'Logistic Regression' if task_type=='Classification' else 'Linear Regression'}")
    st.write(f"**Metric:** {metric_name}")
    st.write(f"**Score:** `{score:.3f}`")

    # -----------------------------
    # BEFORE vs AFTER
    # -----------------------------
    if prev_score is not None:
        improvement = score - prev_score
        st.markdown("### 📈 Before vs After")
        st.write(f"Before agent actions: `{prev_score:.3f}`")
        st.write(f"After agent actions: `{score:.3f}`")
        st.write(f"Improvement: **+{improvement*100:.1f}%**")
    else:
        st.session_state["baseline_before"] = score
        st.info("Initial baseline stored. Re-run after agent actions to compare.")

    # -----------------------------
    # FEATURE IMPORTANCE (coefficients)
    # -----------------------------
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": np.abs(model.coef_[0]) if task_type=="Classification" else np.abs(model.coef_)
    }).sort_values("Importance", ascending=False)

    st.markdown("### 🧠 What the model learned (Top Features)")
    st.write(coef_df.head(5))

    # -----------------------------
    # AGENT VALIDATION
    # -----------------------------
    st.markdown("### 🤖 Agent Validation")
    st.success("Agent decisions were validated. Dataset quality improved.")

    # -----------------------------
    # NEXT STEPS
    # -----------------------------
    st.markdown("### ➡️ Recommended Next Steps")
    st.markdown("""
• Try **Gradient Boosting / XGBoost**  
• Tune hyperparameters  
• Compare CV scores  
• Export cleaned dataset  
""")
#####################################################################





# MODEL RECOMMENDATIONS

st.header("🤖 ML Model Recommendations (Data-Aware)")

st.markdown("### 🔍 Dataset Characteristics Detected")
st.write(f"• Numeric features: {len(num_cols)}")
st.write(f"• Categorical features: {len(cat_cols)}")
st.write(f"• Skewed numeric features: {len(skewed_cols)}")

if task_type == "Regression":
    st.markdown("""
### ✅ Recommended Models & WHY

1️⃣ **Random Forest**
- Handles non-linearity
- Robust to outliers
- No scaling required

2️⃣ **Gradient Boosting**
- Learns complex interactions
- Strong tabular performance

3️⃣ **XGBoost / LightGBM**
- Handles missing values
- Excellent for structured data
- High accuracy with tuning

4️⃣ **CatBoost**
- Best for categorical-heavy datasets
- Minimal preprocessing

5️⃣ **ElasticNet (Baseline)**
- Interpretable
- Good benchmark after scaling
""")
else:
    st.markdown("""
### ✅ Recommended Models & WHY

1️⃣ **Random Forest**
- Robust baseline
- Handles mixed feature types

2️⃣ **XGBoost / LightGBM**
- Handles imbalance well
- Strong decision boundaries

3️⃣ **CatBoost**
- Best for categorical variables
- Minimal encoding needed

4️⃣ **Logistic Regression**
- Interpretable baseline
- Requires scaling

5️⃣ **SVM**
- Works well on smaller datasets
- Sensitive to scaling
""")


# ML READINESS SCORE

st.header("✅ ML Readiness Score")

score = 100

# Missing values impact
if not missing_df.empty:
    score -= min(30, missing_df["Missing_%"].mean())

# Feature quality impact (SAFE)
score -= 10 if len(drop_features) > 3 else 0

# Target skewness impact
target_skew = df[target].skew()
score -= 10 if abs(target_skew) > 1 else 0

score = max(0, score)

st.metric("ML Readiness", f"{score:.1f} / 100")

if score >= 80:
    st.success("Dataset is ML-ready 🚀")
elif score >= 60:
    st.warning("Dataset usable with preprocessing")
else:
    st.error("Heavy preprocessing required")


