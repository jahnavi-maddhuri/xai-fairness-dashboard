import streamlit as st

# Page config
st.set_page_config(
    page_title="Model Fairness Explorer",
    layout="centered",
)

# Title + intro
st.title("Model Fairness Explorer")
st.write(
    """
    Welcome to the Model Fairness Explorer app.

    Use the selector below to choose which model you want to explore.
    """
)

# Model selector
model_choice = st.selectbox(
    "Select a model to work with:",
    ["Random Forest", "XGBoost"],
    index=0,
)

# Simple feedback / placeholder
st.markdown("---")
st.subheader("Current Selection")

if model_choice == "Random Forest":
    st.write(
        """
        You have selected **Random Forest**.

        This model uses an ensemble of decision trees with bagging to make predictions.
        """
    )
else:
    st.write(
        """
        You have selected **XGBoost**.

        This model uses gradient boosting over decision trees to optimize prediction accuracy.
        """
    )

st.info("Next step: hook this selection into your SHAP visualizations and dashboards.")
