import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def pretty_feature_name(name: str) -> str:
    # Categorical features: "cat__column_category"
    if name.startswith("cat__"):
        rest = name[len("cat__"):]          # e.g. "marital.status_Married-civ-spouse"
        col, val = rest.split("_", 1)       # "marital.status", "Married-civ-spouse"

        col_clean = (
            col.replace(".", " ")
               .replace("-", " ")
               .title()                     # "Marital Status"
        )
        val_clean = (
            val.replace(".", " ")
               .replace("-", " ")
               .replace("&", " & ")
        )
        return f"{col_clean}: {val_clean}"

    # Numeric features: "num__age", "num__capital.gain", etc.
    if name.startswith("num__"):
        col = name[len("num__"):]           # e.g. "capital.gain"
        mapping = {
            "age": "Age",
            "fnlwgt": "Sampling Weight",
            "education.num": "Education",
            "capital.gain": "Capital Gain",
            "capital.loss": "Capital Loss",
            "hours.per.week": "Hours per Week",
        }
        if col in mapping:
            return mapping[col]
        return col.replace(".", " ").replace("_", " ").title()

    # Fallback
    return name

def plot_top_features_two_models(
    shap_values_1,
    shap_values_2,
    feature_names,
    pretty_names=None,
    model1_name="Random Forest",
    model2_name="XGBoost",
    top_n=5,
):
    """
    Compare top-N global feature importances (mean |SHAP|) for two models
    in a single long Plotly figure, stacked vertically.

    Parameters
    ----------
    shap_values_1 : array-like of shape (n_samples, n_features)
    shap_values_2 : array-like of shape (n_samples, n_features)
    feature_names : list/array of original feature names (length n_features)
    pretty_names  : optional list/array of display names (same length as feature_names)
    model1_name   : str, label for first model
    model2_name   : str, label for second model
    top_n         : int, number of top features to show per model
    """
    feature_names = np.array(feature_names)

    if pretty_names is None:
        display_names = feature_names
    else:
        display_names = np.array(pretty_names)

    # 1. Global importance = mean absolute SHAP per feature
    imp1 = np.abs(shap_values_1).mean(axis=0)
    imp2 = np.abs(shap_values_2).mean(axis=0)

    # 2. Get top-N indices for each model
    top_idx_1 = np.argsort(imp1)[-top_n:][::-1]
    top_idx_2 = np.argsort(imp2)[-top_n:][::-1]

    names_1 = display_names[top_idx_1]
    vals_1  = imp1[top_idx_1]

    names_2 = display_names[top_idx_2]
    vals_2  = imp2[top_idx_2]

    # Reverse for plotting so most important appears at top
    names_1_plot = names_1[::-1]
    vals_1_plot  = vals_1[::-1]

    names_2_plot = names_2[::-1]
    vals_2_plot  = vals_2[::-1]

    # 3. Build one long figure (two rows, one column)
    fig = make_subplots(
        rows=2,
        cols=1,
        # shared_x=False,
        vertical_spacing=0.25,
        subplot_titles=(
            f"{model1_name}: Top {top_n} Features",
            f"{model2_name}: Top {top_n} Features",
        ),
    )

    # Model 1 (green)
    fig.add_trace(
        go.Bar(
            x=vals_1_plot,
            y=names_1_plot,
            orientation="h",
            name=model1_name,
            marker_color="#2E7D32",  # green
        ),
        row=1,
        col=1,
    )

    # Model 2 (blue)
    fig.add_trace(
        go.Bar(
            x=vals_2_plot,
            y=names_2_plot,
            orientation="h",
            name=model2_name,
            marker_color="#1565C0",  # blue
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(automargin=True, row=1, col=1)
    fig.update_yaxes(automargin=True, row=2, col=1)

    fig.update_layout(
        height=650,
        width=900,
        showlegend=False,
        xaxis_title="Mean |SHAP value|",
        xaxis2_title="Mean |SHAP value|",
        margin=dict(l=160, r=40, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black")
    )

    fig.update_xaxes(tickfont=dict(color="black")) # , titlefont=dict(color="black")
    fig.update_yaxes(tickfont=dict(color="black"))

    return fig

def plot_shap_multi_feature_comparison(
    model_results: dict,
    features,              # list of feature names (str) OR indices (int), or a single one
    models: list = None,   # list of model names to compare; defaults to all in model_results
    class_idx: int = None, # for multi-class SHAP arrays of shape (n_samples, n_features, n_classes)
    title: str = None
):
    """
    Plot horizontal boxplots comparing SHAP values for multiple features across models.

    - Grouped by feature (Y axis)
    - Colored by model (green, blue for RF / XGB)
    """

    # Normalize features to a list
    if isinstance(features, (str, int)):
        features = [features]

    # Default to all models if none provided
    if models is None:
        models = list(model_results.keys())

    # Use the first model as reference for feature names
    ref_model = models[0]
    feat_names = list(model_results[ref_model]['feature_names'])

    # Resolve feature indices and pretty labels
    selected_features = []  # list of (feature_idx, pretty_label)
    for f in features:
        if isinstance(f, str):
            try:
                idx = feat_names.index(f)
            except ValueError:
                raise ValueError(
                    f"Feature '{f}' not found in feature_names for model '{ref_model}'."
                )
            raw_name = feat_names[idx]
        else:
            idx = int(f)
            raw_name = feat_names[idx]

        pretty_label = pretty_feature_name(raw_name)
        selected_features.append((idx, pretty_label))

    # Build long-form dataframe: one row per (sample, feature, model)
    records = []

    for model_name in models:
        shap_vals = model_results[model_name]['shap_values']

        for feature_idx, pretty_label in selected_features:
            # Handle SHAP array shape
            if shap_vals.ndim == 2:
                values = shap_vals[:, feature_idx]
            elif shap_vals.ndim == 3:
                if class_idx is None:
                    raise ValueError(
                        f"shap_values for model '{model_name}' have shape {shap_vals.shape}; "
                        "please provide class_idx for multi-class SHAP values."
                    )
                values = shap_vals[:, feature_idx, class_idx]
            else:
                raise ValueError(
                    f"Unexpected shap_values shape {shap_vals.shape} for model '{model_name}'."
                )

            for v in values:
                records.append(
                    {
                        "Model": model_name,
                        "Feature": pretty_label,
                        "SHAP": float(v),
                    }
                )

    df = pd.DataFrame(records)

    # Color map (your specified colors for these two)
    color_map = {
        "Random Forest": "green",
        "XGBoost": "blue",
    }

    if title is None:
        if len(selected_features) == 1:
            title = f"SHAP distribution for {selected_features[0][1]} across models"
        else:
            feat_list = ", ".join([pl for _, pl in selected_features])
            title = f"SHAP Distribution Comparison Across Models for Selected Features"

    fig = px.box(
        df,
        x="SHAP",
        y="Feature",
        color="Model",
        orientation="h",
        color_discrete_map=color_map,
        points=False,  # change to "all" or "outliers" if you want jittered points
    )

    fig.update_layout(
        title=title,
        height=200 + 80 * len(selected_features),
        xaxis_title="SHAP value",
        yaxis_title="Feature",
        boxmode="group",
        template="plotly_white",
        legend_title_text="Model",
    )

    # Order features in the same order as requested (top to bottom)
    fig.update_yaxes(categoryorder="array", categoryarray=[pl for _, pl in selected_features])
    return fig