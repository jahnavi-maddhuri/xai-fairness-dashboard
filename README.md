# XAI Fairness Dashboard

Check out my streamlit app hosted at this link: https://xai-fairness-dashboard.streamlit.app/

**Purpose**

Modern machine-learning systems—especially in domains like loan approval, credit risk, and criminal justice cannot rely on black-box predictions alone. High-stakes decisions demand accountability, transparency, and rigorous interpretability to ensure models are not simply amplifying or reproducing historical bias.

The Fairness Explorer Dashboard was built to bridge the gap between model complexity and human understanding. It provides:
- Model-agnostic SHAP visualizations
- Side-by-side comparisons across models (e.g., Random Forest vs. XGBoost)
- Fairness-driven framing: do feature effects differ across models or hyperparameter choices?
- A clean, extensible pipeline for generating, storing, and visualizing SHAP values in a consistent schema for any future models.

The goal is not only to present SHAP values, but to enable auditable interpretability for any predictive system that may impact real people.

## Project Structure
```
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
```
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
If you are interested in running this app local to add your own models, plots or change the existing app, follow the steps below.

1. Clone the repository
```
git clone <your-repo-url>
cd <your-repo-name>
```
2. Create a virtual environment (recommended)
```
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```
3. Install dependencies
```{bash}
pip install -r requirements.txt
```

## Run Dashboard
```
streamlit run app.py
```
This launches the Fairness Explorer Dashboard in your browser (usually at http://localhost:8501).

If using the precomputed SHAP files in data/, the dashboard loads immediately!

## How to Add New Models
Train your model (e.g., Random Forest, XGBoost, Logistic Regression, etc.).

Use the provided generator script template:
- `gen_shap_rf.py`
- `gen_shap_xgb.py` OR a `custom gen_shap_your_model.py` derived from these examples.

Ensure your SHAP output dictionary uses the same feature_name structure as it's comparisons.

Save your results into `data/your_shap_file.npz`.

Once placed in data/, the dashboard detects the new file and allows comparison.

## Why This Dashboard Matters?
✔️ **Interpretability** Demonstrates transparent SHAP-based explanations for high-stakes ML tasks.

✔️ **Usability** Simple, interactive UI with precomputed SHAP values included for instant exploration.

✔️ **Reproducibility** Code to generate SHAP values is included, Data sources are documented, Schema is standardized

✔️ **Extensibility** Supports new models, new hyperparameters, fairness-tuning experiments

✔️ **Technical clarity** Clear structure, modular SHAP-generation scripts, and robust Plotly visualizations.

### Example Usage
1. User first selects two models to compare. (If additional models are desired, those must be added prior)
<img width="826" height="720" alt="image" src="https://github.com/user-attachments/assets/6ec56235-cb19-45a7-890e-e877c254edc0" />

2. Once user hits "Run comparison analysis" the bar graph below shows.
<img width="788" height="831" alt="image" src="https://github.com/user-attachments/assets/fca49cc3-c3c4-4b30-843f-af9ec9893c13" />

User can interact with the chart for more granular analysis and can reference interpretation guidelines on the dashboard to understand the model better.

3. Then, user can select features to compare accross models for a box visual similar to that shown below.
<img width="808" height="887" alt="image" src="https://github.com/user-attachments/assets/eee1fcfa-5cc4-4991-a491-249d88eb4b2c" />

Users can dynamically choose more features or remove features from the select box. Once again, plot is interactive and interpretation guidelines are included in detail.

### AI Acknowledgment
I used ChatGPT 5.1 throughout this assignment as a coding assistant to help me understand how to generalize SHAP pipelines, design reusable functions, and structure dynamic visualizations. The tool supported my learning process, but all core logic, experimentation, testing, and integration into the final dashboard were done by me. I am including a link to the conversation: [ChatGPT5.1 Conversation Link](https://chatgpt.com/c/6923b24a-bd00-832a-a4ad-c332a8f498ed)
