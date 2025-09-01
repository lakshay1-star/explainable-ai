import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

def get_shap_explainer(model: BaseEstimator, X: pd.DataFrame) -> tuple:
    """
    Creates a SHAP explainer and calculates SHAP values for a given model.

    Args:
        model: The trained machine learning model (e.g., XGBoost, RandomForest).
        X (pd.DataFrame): The input data for which to generate explanations.

    Returns:
        A tuple containing the SHAP explainer and the calculated SHAP values.
    """
    # Use TreeExplainer for tree-based models for better performance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values

def get_lime_explainer(X_train: pd.DataFrame, class_names: list) -> lime.lime_tabular.LimeTabularExplainer:
    """
    Creates and configures a LIME Tabular Explainer.

    Args:
        X_train (pd.DataFrame): The training data used to build the explainer.
        class_names (list): The names of the classes for the prediction (e.g., ['No Churn', 'Churn']).

    Returns:
        A configured LIME Tabular Explainer instance.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode='classification'
    )
    return explainer