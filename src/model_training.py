import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from typing import Tuple

def train_xgb_classifier(X: pd.DataFrame, y: pd.Series) -> Tuple[XGBClassifier, pd.DataFrame, pd.Series]:
    """
    Trains an XGBoost classifier and evaluates its performance.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.

    Returns:
        A tuple containing the trained model, test features, and test target.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_class = (preds_proba > 0.5).astype(int)
    
    print("--- XGBoost Model Performance ---")
    print(f"AUC Score: {roc_auc_score(y_test, preds_proba):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, preds_class):.4f}\n")
    
    return model, X_test, y_test

def train_random_forest_classifier(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    """
    Trains a RandomForest classifier and prints a classification report.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.

    Returns:
        A tuple containing the trained model, test features, and test target.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    print("--- Random Forest Model Performance ---")
    print(classification_report(y_test, predictions))
    
    return model, X_test, y_test