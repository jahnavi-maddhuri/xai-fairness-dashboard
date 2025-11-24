import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
import xgboost as xgb

import shap
shap.initjs()

# Load Data
df = pd.read_csv("adult.csv")
print("Raw shape:", df.shape)

# Replace '?' markers with NaN and drop rows with missing values
df = df.replace("?", np.nan)
df = df.dropna()
print("After dropping missing values:", df.shape)

# Standardize column names
df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]

# Target is 'income': values like ' <=50K' and ' >50K'
df["income"] = df["income"].str.strip()
df["income"] = (df["income"] == ">50K").astype(int)

# Train Test Split
X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode categorical features

categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

print("Categorical columns:", list(categorical_cols))
print("Numeric columns:", list(numeric_cols))

# Model Random Forests
rf_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest accuracy: {acc_rf:.3f}")

# Run Shap Analysis + Save
# 1. Pull preprocessor + RF out of your pipeline
preprocessor = rf_clf.named_steps["preprocessor"]
rf_model     = rf_clf.named_steps["clf"]

# 2. Transform X_train and X_test
X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# If sparse, make dense
if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_test_proc  = X_test_proc.toarray()

# Ensure numeric dtype
X_train_proc = X_train_proc.astype(np.float32)
X_test_proc  = X_test_proc.astype(np.float32)

# 3. Feature names
feature_names = preprocessor.get_feature_names_out()

# 4. TreeExplainer on the RF model
explainer_rf = shap.TreeExplainer(rf_model)

print("X_test_proc shape:", X_test_proc.shape, "dtype:", X_test_proc.dtype)
print("num features:", len(feature_names))

from tqdm import tqdm

# n rows you to explain
n_samples = 500
n_samples = min(n_samples, X_test_proc.shape[0])

# pick random subset
idx = np.random.choice(X_test_proc.shape[0], size=n_samples, replace=False)
X_test_small = X_test_proc[idx]

# progress over batches instead of 1 giant call
batch_size = 100
batches = []

for start in tqdm(range(0, n_samples, batch_size), desc="Computing SHAP in batches"):
    end = min(start + batch_size, n_samples)
    batch = X_test_small[start:end]

    vals = explainer_rf.shap_values(batch)

    if isinstance(vals, list):
        batches.append(vals[1])
    else:
        batches.append(vals[..., 1])

# stack all batches into a single array
shap_values_small = np.vstack(batches)

np.savez(
    "data/shap_results_rf.npz",
    shap_values=shap_values_small,
    X=X_test_small,
    feature_names=feature_names
)
