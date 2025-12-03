import streamlit as st
import numpy as np
from plotly_plots import *

# Page config
st.set_page_config(
    page_title="Model Fairness Explorer",
    layout="centered",
)

# CSS style for application
# White theme, visible subheaders/info boxes
st.markdown(
    """
    <style>

        /* ---------------------------------------------------------- */
        /* GLOBAL APP BACKGROUND + TEXT COLOR                        */
        /* ---------------------------------------------------------- */
        .stApp {
            background-color: white !important;
            color: black !important;
        }

        html, body, [class*="css"] {
            color: black !important;
        }

        /* Headings (all levels) */
        h1, h2, h3, h4, h5, h6 {
            color: black !important;
        }

        /* Streamlit subheader markdown */
        .stMarkdown h3 {
            color: black !important;
        }

        /* ---------------------------------------------------------- */
        /* DROPDOWNS                                                  */
        /* ---------------------------------------------------------- */
        div[data-baseweb="select"] > div {
            background-color: white !important;
            color: black !important;
        }

        div[data-baseweb="select"] span {
            color: black !important;
        }

        /* ---------------------------------------------------------- */
        /* INFO / ALERT BOXES                                         */
        /* ---------------------------------------------------------- */
        .stAlert {
            background-color: #E8F4FD !important;   /* Light info-box blue */
            border-left: 4px solid #2196F3 !important;
            color: black !important;
        }

        /* Force black text inside info/warning/error/success */
        .stAlert, .stAlert p, .stAlert div, .stAlert span {
            color: black !important;
        }

        /* ---------------------------------------------------------- */
        /* HORIZONTAL RULE (---)                                      */
        /* ---------------------------------------------------------- */
        hr {
            border: 1px solid black !important;
        }
        /* Make Streamlit buttons green with white text */
        div.stButton > button {
            background-color: #2E7D32 !important;   /* green */
            color: white !important;                 /* white text */
            border-radius: 6px !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
        }

        /* Darker green on hover */
        div.stButton > button:hover {
            background-color: #1B5E20 !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        /* Force BLACK TEXT for all alert/feedback boxes:
           st.error, st.warning, st.info, st.success */
        .stAlert, .stAlert p, .stAlert div, .stAlert span {
            color: black !important;
        }

        /* Optional: customize the background of st.error specifically */
        .stAlert[data-baseweb="notification"][kind="error"] {
            background-color: #fdecea !important;  /* light red */
            border-left: 4px solid #d32f2f !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with "Fairness" in green
st.markdown(
    """
    <h1 style="text-align:center; color:black;">
        Model <span style="color:#2E7D32;">Fairness</span> Explorer
    </h1>
    <style>
    .stSelectbox div[data-baseweb="select"] > div {
    border: 1px solid #000000 !important;        /* black border */
    box-shadow: 0 0 0 1px #000000 !important;    /* black outline */
    border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
    Use this app to compare how different models behave from a fairness and interpretability
    perspective. Get started by selecting two models below! Comparing a model to itself is allowed.

    Each model is trained to predict whether or not a user would get a loan, using adult income data.
    """
)

st.markdown("---") # Divider line
st.subheader("Select two models to compare")

model_options = ["Please Select a Model", "Random Forest", "XGBoost"]

col1, col2 = st.columns(2)

with col1:
    model_left = st.selectbox(
        "Left model",
        model_options,
        index=0,
        key="model_left",
    )

with col2:
    model_right = st.selectbox(
        "Right model",
        model_options,
        index=0 if len(model_options) > 1 else 0,
        key="model_right",
    )

st.write("### Current Selection")
st.write(f"- **Left model:** {model_left}")
st.write(f"- **Right model:** {model_right}")
# Load Model Results/SHAP values
model_results = {
    'Random Forest':{'res_file':'shap_results_rf.npz'},
    'XGBoost': {'res_file':'shap_results_xgb.npz'}
}

for model in model_results.keys():
    fname = model_results[model]['res_file']
    data = np.load(f'data/{fname}', allow_pickle=True)
    model_results[model]['shap_values'] = shap_values = data["shap_values"]
    model_results[model]['X_small'] = data["X"]
    model_results[model]['feature_names'] = data["feature_names"]

if "run_compare_pressed" not in st.session_state:
    st.session_state.run_compare_pressed = False

def handle_run_compare():
    if model_left == "Please Select a Model" or model_right == "Please Select a Model":
        st.session_state.run_compare_pressed = False
        st.error("Please select a model in both dropdowns before running the comparison.")
    else:
        st.session_state.run_compare_pressed = True

st.button("Run comparison analysis", on_click=handle_run_compare)

if st.session_state.run_compare_pressed:
    st.markdown("---") # Divider Line
    st.subheader("SHAP Comparison: Feature Importance")
    st.write(f"Comparing **{model_left}** vs **{model_right}**")

    pretty_names = [pretty_feature_name(f) for f in model_results[model_left]['feature_names']]

    fig = plot_top_features_two_models(
        model_results[model_left]['shap_values'],
        model_results[model_right]['shap_values'],
        feature_names=model_results[model_left]['feature_names'],
        pretty_names=pretty_names,
        model1_name=model_left,
        model2_name=model_right,
        top_n=7,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
    """
    **How to interpret this chart?**

    The bar chart above highlights the features that each model considers the most influential.
    Notice how different model selections rank features differently, even though 
    models are trained on identical datasets. This illustrates that different algorithms can focus 
    on different signals in the data.

    The x-axis represents the **average absolute SHAP value**, which measures how strongly a 
    feature affects the model’s predictions on average. Importantly, this reflects **magnitude 
    of impact** only, not whether the feature pushes a prediction up or down.

    **Fairness Evaluation**

    As you look at which features appear most influential, consider whether these are fair 
    characteristics to use when deciding whether to approve a loan. Are any features potentially 
    sensitive, uncomfortable, or misaligned with your understanding of fair decision-making?

    This comparison helps reveal where a model may rely on attributes that, while predictive, 
    may not align with ethical or regulatory expectations of fairness.
    """
    )

if "feature_compare_pressed" not in st.session_state:
    st.session_state.feature_compare_pressed = False

def handle_feature_compare():
    if model_left == "Please Select a Model" or model_right == "Please Select a Model":
        st.session_state.feature_compare_pressed = False
        st.error("Please select a model in both dropdowns before running any comparison.")
    else:
        st.session_state.feature_compare_pressed = True

st.markdown("---") # Divider Line
st.subheader("SHAP Feature Comparison Across Models")

# st.write(f"Comparing **{model_left}** vs **{model_right}**")
raw_feature_names = model_results['Random Forest']['feature_names']
pretty_names = [pretty_feature_name(f) for f in raw_feature_names]
pretty_to_raw = dict(zip(pretty_names, raw_feature_names))

st.markdown("""
<style>
    .stMultiSelect div[data-baseweb="select"] > div {
    border: 1px solid #000000 !important;        /* black border */
    box-shadow: 0 0 0 1px #000000 !important;    /* black outline */
    border-radius: 10px !important;
    }
    /* --- SELECTED TAG (CHIP) COLOR --- */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #99ccff !important;  /* blue */
        color: white !important;
        border-radius: 6px !important;
    }

</style>
""", unsafe_allow_html=True)

selected_pretty = st.multiselect(
        "Select features to compare",
        options=pretty_names,
        default=['Marital Status: Married AF spouse',
                 'Marital Status: Married civ spouse',
                 'Race: Black'
                 ],
        key="feature_multiselect",
        )

st.button("Run feature comparison", on_click=handle_feature_compare)
if st.session_state.feature_compare_pressed:    
    selected_raw = [pretty_to_raw[p] for p in selected_pretty]
    features_to_plot = [
        "cat__marital.status_Married-civ-spouse",
        "cat__marital.status_Married-AF-spouse",
        "num__age",
    ]

    fig2 = plot_shap_multi_feature_comparison(
        model_results=model_results,
        features=selected_raw,
        models=[model_left, model_right],  # optional
        # class_idx=1  # only if your SHAP arrays are 3D
    )

    st.plotly_chart(fig2, use_container_width=True)
    
    st.info("""
**How to Interpret SHAP Feature Comparison Across Models**

This plot compares how each model weighs the same features when predicting loan approval.

- **Positive SHAP values** increase the odds of receiving a loan, while **Negative values** decrease those odds.
- Larger absolute SHAP values mean the feature has a stronger influence on the model’s decision.
- Big differences between models for the same feature indicate that one model relies on that feature more heavily, which can reveal model-specific bias.
- If protected characteristics (such as race or sex) show strong or asymmetric SHAP patterns, this may signal potential fairness concerns and should prompt further investigation.

This view helps highlight which features drive each model’s behavior and where their decisions may diverge in ways relevant to fairness.
"""
)
