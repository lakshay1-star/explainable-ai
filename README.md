# Explainable AI (XAI) for Model Transparency in Credit and Churn Prediction

This project provides a practical demonstration of Explainable AI (XAI) techniques to interpret and build trust in machine learning models. Using synthetic datasets for customer churn and credit scoring, it showcases how to dissect "black-box" models like XGBoost and Random Forest to understand their decision-making processes.

## 🚀 Features

- **Real-World Use Cases:** Demonstrates XAI techniques on two common business problems: customer churn prediction and credit risk assessment.
- **State-of-the-Art XAI Libraries:**
    - **SHAP (SHapley Additive Explanations):** Utilizes `shap` to create visualizations for global and local feature importance.
    - **LIME (Local Interpretable Model-agnostic Explanations):** Implements `lime` to explain individual model predictions on a case-by-case basis.
- **Modular & Reusable Code:** The project is structured with a `src/` directory containing clean, reusable functions for data loading, preprocessing, model training, and explanation generation.
- **Multiple Models:** Includes examples for explaining both `XGBoost` and `RandomForestClassifier` models.
- **End-to-End Notebooks:** Provides detailed Jupyter notebooks that walk through the entire process from data loading to generating and interpreting explanations.

<!-- ## 📂 Project Structure

A well-organized and scalable project structure.

explainable-ai-xai/
├── .gitignore                # Standard file to ignore Python artifacts
├── README.md                 # You are here!
├── requirements.txt          # Project dependencies
├── data/
│   ├── churn_synthetic.csv   # Synthetic customer churn data
│   └── credit_data.csv       # Synthetic credit scoring data
├── notebooks/
│   ├── 1_credit_risk_explanation.ipynb  # Demo on credit data
│   └── 2_churn_prediction_explanation.ipynb # Demo on churn data
├── results/
│   ├── credit_shap_summary.png # Example SHAP output for credit model
│   └── churn_lime_explanation.txt  # Example LIME output for churn model
└── src/
├── init.py
├── data_preprocessing.py   # Data loading and cleaning functions
├── explainability.py       # SHAP and LIME explanation utilities
└── model_training.py       # Model training and evaluation pipelines -->


## ⚡ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd explainable-ai-xai
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run the Demos

The core of this project is in the Jupyter notebooks, which are designed to be run sequentially.

1.  **Start the Jupyter Notebook server:**
    ```bash
    jupyter notebook
    ```

2.  **Navigate to the `notebooks/` directory and run the notebooks:**
    - `1_credit_risk_explanation.ipynb`: A comprehensive workflow using SHAP and LIME on the synthetic credit scoring dataset.
    - `2_churn_prediction_explanation.ipynb`: A focused demonstration explaining churn predictions.

## 💡 Project Motivation

As machine learning models become increasingly integrated into critical business decisions, transparency is no longer a "nice-to-have" but a necessity. This project was built to bridge the gap between model performance and model interpretability. As part of my journey in **AI for real-world impact**, I wanted to create a practical, hands-on repository that highlights the importance of understanding the *why* behind a model's predictions, a core theme in my professional and academic work.

---
📝 **Author:** Lakshay Naresh Ramchandani – MS Data Science, University of Pennsylvania