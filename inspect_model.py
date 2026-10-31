"""
================================================================================
Model Inspection Utility: inspect_model.py
================================================================================
Academic Project: Diabetes Prediction System
Purpose:
  Explains and reveals the internal contents of 'model.pkl' in human-readable
  format for advisors, professors, and college viva voce examination.
================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import export_text


def inspect_saved_model(model_path: str = "model.pkl"):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found! Run 'python train_model.py' first.")
        return

    print("=" * 80)
    print("MODEL INSPECTION REPORT: WHAT IS INSIDE 'model.pkl'?")
    print("=" * 80)
    print(f"File: {model_path} (Binary Serialized Object via joblib / pickle)")
    print("This file stores the complete trained pipeline, including imputation medians,")
    print("standard scaling parameters, and the weights of all three algorithms.")
    print("=" * 80)

    # 1. Load the model
    pipeline = joblib.load(model_path)
    print("\n[1] TOP-LEVEL OBJECT:")
    print(f"  Class: {type(pipeline).__module__}.{type(pipeline).__name__}")
    print(f"  Pipeline Steps: {list(pipeline.named_steps.keys())}")

    # 2. Inspect Preprocessor
    print("\n" + "-" * 80)
    print("[2] STEP 1: PREPROCESSOR (ColumnTransformer)")
    print("-" * 80)
    preprocessor = pipeline.named_steps['preprocessor']
    for trans_name, trans_obj, cols in preprocessor.transformers_:
        print(f"  * Transformer: '{trans_name}'")
        print(f"    - Applied to columns: {cols}")
        if hasattr(trans_obj, 'statistics_'):
            print(f"    - Learned Median Values: {dict(zip(cols, trans_obj.statistics_.round(2)))}")

    # 3. Inspect Scaler
    print("\n" + "-" * 80)
    print("[3] STEP 2: FEATURE SCALER (StandardScaler)")
    print("-" * 80)
    scaler = pipeline.named_steps['scaler']
    transformed_cols = [
        'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'Pregnancies', 'DiabetesPedigreeFunction', 'Age'
    ]
    print(f"  Learned Feature Means: {dict(zip(transformed_cols, scaler.mean_.round(2)))}")
    print(f"  Learned Feature Scales (Std Dev): {dict(zip(transformed_cols, scaler.scale_.round(2)))}")

    # 4. Inspect Ensemble Classifier
    print("\n" + "-" * 80)
    print("[4] STEP 3: ENSEMBLE CLASSIFIER (VotingClassifier - Soft Voting)")
    print("-" * 80)
    ensemble = pipeline.named_steps['classifier']
    print(f"  Voting Strategy: {ensemble.voting.upper()} VOTING")
    print(f"  Estimators Combined: {list(ensemble.named_estimators_.keys())}")

    # Estimator 1: Logistic Regression
    lr = ensemble.named_estimators_['logistic_regression']
    print("\n  >> ESTIMATOR 1: Logistic Regression")
    print(f"     * Hyperparameters: max_iter={lr.max_iter}, solver='{lr.solver}', C={lr.C}")
    print(f"     * Intercept (Bias): {lr.intercept_[0]:.4f}")
    print("     * Learned Feature Coefficients (Weights):")
    lr_weights = pd.DataFrame({
        'Feature': transformed_cols,
        'Coefficient (Weight)': lr.coef_[0].round(4)
    }).sort_values(by='Coefficient (Weight)', ascending=False)
    for _, row in lr_weights.iterrows():
        print(f"       - {row['Feature']:<25}: {row['Coefficient (Weight)']:>+.4f}")

    # Estimator 2: Decision Tree
    dt = ensemble.named_estimators_['decision_tree']
    print("\n  >> ESTIMATOR 2: Decision Tree Classifier")
    print(f"     * Tree Depth: {dt.get_depth()} (Max Depth: {dt.max_depth})")
    print(f"     * Number of Decision Leaves: {dt.get_n_leaves()}")
    print("     * Top Decision Rules in the Tree:")
    tree_rules = export_text(dt, feature_names=transformed_cols, max_depth=2)
    print("\n" + "\n".join("       " + line for line in tree_rules.splitlines()[:12]))

    # Estimator 3: Random Forest
    rf = ensemble.named_estimators_['random_forest']
    print("\n  >> ESTIMATOR 3: Random Forest Classifier")
    print(f"     * Number of Trees in Forest: {len(rf.estimators_)}")
    print(f"     * Tree Max Depth: {rf.max_depth}, Min Samples Leaf: {rf.min_samples_leaf}")
    print("     * Ensemble Feature Importances (Gini Importance):")
    rf_imp = pd.DataFrame({
        'Feature': transformed_cols,
        'Importance (%)': (rf.feature_importances_ * 100).round(2)
    }).sort_values(by='Importance (%)', ascending=False)
    for _, row in rf_imp.iterrows():
        print(f"       - {row['Feature']:<25}: {row['Importance (%)']:>6.2f}%")

    print("\n" + "=" * 80)
    print("SUMMARY FOR VIVA / ADVISOR PRESENTATION:")
    print("  1. 'model.pkl' is a compiled binary object, not plain text code.")
    print("  2. It bundles the imputation parameters + scaling parameters + all 3 models.")
    print("  3. Most influential features for prediction: Glucose, BMI, and Age.")
    print("=" * 80)


if __name__ == "__main__":
    inspect_saved_model()
