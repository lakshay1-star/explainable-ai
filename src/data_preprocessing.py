import pandas as pd
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    """
    Loads data from a specified CSV file path.

    Args:
        path (str): The file path to the CSV data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        return pd.DataFrame()

def preprocess_churn_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the churn dataset by handling categorical features.

    Args:
        df (pd.DataFrame): The raw churn DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the feature matrix (X)
                                         and the target vector (y).
    """
    df_processed = df.copy()
    
    # One-hot encode categorical features from the churn dataset
    df_processed = pd.get_dummies(df_processed, columns=['contract', 'payment_method'], drop_first=True)
    
    # Define features and target
    X = df_processed.drop(columns=['customer_id', 'churn'], axis=1)
    y = df_processed['churn']
    
    return X, y