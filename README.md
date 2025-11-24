# XAI Fairness Dashboard
**Purpose**
Modern machine-learning systems—especially in domains like loan approval, credit risk, and criminal justice cannot rely on black-box predictions alone. High-stakes decisions demand accountability, transparency, and rigorous interpretability to ensure models are not simply amplifying or reproducing historical bias.

The Fairness Explorer Dashboard was built to bridge the gap between model complexity and human understanding. It provides:
- Model-agnostic SHAP visualizations
- Side-by-side comparisons across models (e.g., Random Forest vs. XGBoost)
- Fairness-driven framing: do feature effects differ across models or hyperparameter choices?
- A clean, extensible pipeline for generating, storing, and visualizing SHAP values in a consistent schema for any future models.

The goal is not only to present SHAP values, but to enable auditable interpretability for any predictive system that may impact real people.

## Project Structure
.
├── app.py                       # Streamlit dashboard
├── plotly_plots.py              # Plotly visualizations (RF vs XGB comparisons, etc.)
├── gen_shap_random_forest.py    # SHAP generation script (Random Forest)
├── gen_shap_xgboost.py          # SHAP generation script (XGBoost)
├── data/
│   ├── adult.csv                # UCI Adult dataset (cleaned)
│   ├── shap_results_rf.npz      # Precomputed SHAP results for Random Forest
│   └──shap_results_xgb.npz      # Precomputed SHAP results for XGBoost
├── requirements.txt
└── README.md  (this file)

**Included Data**
The data/ folder already contains:
- `adult.data` Cleaned version of the Adult Income dataset (classification: income ≤50K vs. >50K).
- `shap_results_rf.npz` and `shap_results_xgb.npz` Ready-to-use SHAP outputs for Random Forests and XGBoost respectively.
These files power the default dashboard visualizations

You are encouraged to add:
- new models (e.g., LightGBM, CatBoost, custom models)
- different hyperparameter variations
- fairness-specific models (e.g., reweighing, debiasing runs)
As long as files follow the same schema, they can be compared seamlessly.

## Installation & Setup
1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```
2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
bash
