
import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def load_and_inspect_data(csv_path: str = "diabetes.csv") -> pd.DataFrame:
    """Loads the dataset and prints basic inspection summary."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at '{csv_path}'. Please ensure diabetes.csv exists.")

    df = pd.read_csv(csv_path)
    print("=" * 70)
    print("STAGE 1: DATA INSPECTION & PREPARATION")
    print("=" * 70)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target Distribution:\n{df['Outcome'].value_counts().to_string()}")
    
    # Identify biologically impossible zeros (missing medical measurements)
    zero_counts = (df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] == 0).sum()
    print("\nMissing/Zero Value Counts in Medical Features:")
    for col, count in zero_counts.items():
        print(f"  - {col}: {count} zeros ({count / len(df) * 100:.1f}%)")
    
    return df


def build_pipeline():
    """
    Constructs the complete machine learning pipeline:
    1. Preprocessing: Median imputation for biologically missing zero values.
    2. Scaling: Standard scaling across all 8 features.
    3. Classification: Soft VotingClassifier combining 3 diverse algorithms.
    """
    feature_cols = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]

    # Columns where 0 represents missing clinical measurement
    zero_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    # Columns where 0 is physiologically valid (or does not occur)
    passthrough_cols = ['Pregnancies', 'DiabetesPedigreeFunction', 'Age']

    # Step 1: Preprocessing transformer
    # Impute missing clinical zeros with the median computed strictly on training data
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_zero', SimpleImputer(missing_values=0, strategy='median'), zero_missing_cols),
            ('pass_features', 'passthrough', passthrough_cols),
        ]
    )

    # Step 2: Individual Base Classifiers
    # Algorithm 1: Logistic Regression (linear decision boundary baseline)
    clf_lr = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Algorithm 2: Decision Tree Classifier (non-linear, rule-based)
    clf_dt = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    )

    # Algorithm 3: Random Forest Classifier (ensemble of bagged decision trees)
    clf_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    )

    # Step 3: Ensemble Voting Classifier (Soft Voting)
    # Combines predicted class probabilities from all three estimators
    voting_ensemble = VotingClassifier(
        estimators=[
            ('logistic_regression', clf_lr),
            ('decision_tree', clf_dt),
            ('random_forest', clf_rf)
        ],
        voting='soft'
    )

    # Step 4: Full Pipeline wrapping preprocessor, scaler, and ensemble
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', voting_ensemble)
    ])

    return full_pipeline, feature_cols


def train_and_evaluate(csv_path: str = "diabetes.csv", model_output: str = "model.pkl", metrics_output: str = "metrics.json"):
    """
    Full training workflow:
    - Loads data
    - Performs stratified train/test split
    - Fits pipeline on training data ONLY (prevents leakage)
    - Evaluates final model on held-out test data
    - Saves model.pkl and metrics.json
    """
    df = load_and_inspect_data(csv_path)

    feature_cols = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]
    X = df[feature_cols]
    y = df['Outcome']

    # Stratified split ensures equal class proportions in train & test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\nTrain Set Size: {len(X_train)} samples ({y_train.sum()} diabetic, {len(y_train) - y_train.sum()} non-diabetic)")
    print(f"Test Set Size:  {len(X_test)} samples ({y_test.sum()} diabetic, {len(y_test) - y_test.sum()} non-diabetic)")

    pipeline, _ = build_pipeline()

    print("\n" + "=" * 70)
    print("STAGE 2: TRAINING COMBINED VOTING CLASSIFIER")
    print("=" * 70)
    print("Fitting preprocessing, scaling, and ensemble on training data...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    # Evaluate individual estimators on test data for academic viva comparison
    print("\n" + "=" * 70)
    print("STAGE 3: INDIVIDUAL ALGORITHM EVALUATION (HELD-OUT TEST SET)")
    print("=" * 70)
    
    # We can inspect the fitted estimators within the trained pipeline
    fitted_preprocessor = Pipeline([
        ('preprocessor', pipeline.named_steps['preprocessor']),
        ('scaler', pipeline.named_steps['scaler'])
    ])
    X_test_transformed = fitted_preprocessor.transform(X_test)

    individual_metrics = {}
    for name, clf in pipeline.named_steps['classifier'].named_estimators_.items():
        y_ind_pred = clf.predict(X_test_transformed)
        acc = accuracy_score(y_test, y_ind_pred)
        prec = precision_score(y_test, y_ind_pred, zero_division=0)
        rec = recall_score(y_test, y_ind_pred, zero_division=0)
        f1 = f1_score(y_test, y_ind_pred, zero_division=0)
        individual_metrics[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4)
        }
        print(f"[{name.replace('_', ' ').title()}]")
        print(f"  Accuracy:  {acc * 100:.2f}% | Precision: {prec * 100:.2f}% | Recall: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%")

    print("\n" + "=" * 70)
    print("STAGE 4: FINAL ENSEMBLE EVALUATION (VOTING CLASSIFIER)")
    print("=" * 70)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Combined VotingClassifier Test Accuracy:  {acc * 100:.2f}%")
    print(f"Combined VotingClassifier Test Precision: {prec * 100:.2f}%")
    print(f"Combined VotingClassifier Test Recall:    {rec * 100:.2f}%")
    print(f"Combined VotingClassifier Test F1-Score:  {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {cm[0][0]}")
    print(f"  False Positives (FP): {cm[0][1]}")
    print(f"  False Negatives (FN): {cm[1][0]}")
    print(f"  True Positives (TP):  {cm[1][1]}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

    # Store evaluation metrics to accompany model.pkl
    evaluation_record = {
        'model_name': 'VotingClassifier (Soft Voting)',
        'algorithms_combined': [
            'Logistic Regression',
            'Decision Tree Classifier',
            'Random Forest Classifier'
        ],
        'test_set_size': int(len(X_test)),
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'confusion_matrix': {
            'tn': int(cm[0][0]),
            'fp': int(cm[0][1]),
            'fn': int(cm[1][0]),
            'tp': int(cm[1][1])
        },
        'individual_metrics': individual_metrics,
        'feature_names': feature_cols
    }

    # Save metrics JSON
    with open(metrics_output, 'w', encoding='utf-8') as f:
        json.dump(evaluation_record, f, indent=4)
    print(f"Evaluation metrics saved to '{metrics_output}'.")

    # Save fitted model pipeline
    joblib.dump(pipeline, model_output)
    print(f"Fitted pipeline successfully saved to '{model_output}'.")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
