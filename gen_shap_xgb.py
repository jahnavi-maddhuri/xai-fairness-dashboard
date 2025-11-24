import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

xgb_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False
    ))
])

xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost accuracy: {acc_xgb:.3f}")

## Shap
# 1. Extract preprocessor and XGBoost model from your pipeline
preprocessor = xgb_clf.named_steps["preprocessor"]
xgb_model    = xgb_clf.named_steps["clf"]   # XGBClassifier

# 2. Transform X_test into numeric features (for later SHAP calls)
X_test_proc = preprocessor.transform(X_test)

if hasattr(X_test_proc, "toarray"):
    X_test_proc = X_test_proc.toarray()

X_test_proc = X_test_proc.astype(np.float32)

# 3. Feature names (for plots)
feature_names = preprocessor.get_feature_names_out()

# 4. Build TreeExplainer on the XGBClassifier
explainer_xgb = shap.TreeExplainer(xgb_model)

print("✓ TreeExplainer created.")
print("X_test_proc shape:", X_test_proc.shape, "dtype:", X_test_proc.dtype)
print("num features:", len(feature_names))

from tqdm import tqdm
import numpy as np

# 1. Choose how many samples to explain
n_samples = 500
n_samples = min(n_samples, X_test_proc.shape[0])

# 2. Random subset of the processed test data
rng = np.random.default_rng(42)
idx = rng.choice(X_test_proc.shape[0], size=n_samples, replace=False)
X_xgb_small = X_test_proc[idx]

# 3. Compute SHAP values in batches with a progress bar
batch_size = 100
shap_batches = []

for start in tqdm(range(0, n_samples, batch_size), desc="Computing XGBoost SHAP in batches"):
    end = min(start + batch_size, n_samples)
    batch = X_xgb_small[start:end]

    vals = explainer_xgb.shap_values(batch)

    # For binary XGBoost + TreeExplainer this is usually (batch, n_features)
    # If it ever returns a list (one array per class), take the positive class (index 1)
    if isinstance(vals, list):
        shap_batches.append(vals[1])
    else:
        shap_batches.append(vals)

# 4. Stack all batches into a single array
shap_values_xgb_small = np.vstack(shap_batches)

print("SHAP values shape:", shap_values_xgb_small.shape)
np.savez(
    "data/shap_results_xgb.npz",
    shap_values=shap_values_xgb_small,
    X=X_xgb_small,
    feature_names=feature_names
)

