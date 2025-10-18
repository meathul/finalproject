"""Machine Learning Modeling Script
Step 4 tasks:
 - Load engineered CSV (aligned_features.csv)
 - Train/test split
 - Train RandomForest & XGBoost classifiers
 - Evaluate with accuracy, precision, recall, F1
 - Plot feature importance & SHAP values
 - Optional: KMeans clustering for high-risk grouping

Usage:
  python modeling.py --data data/processed/aligned_features.csv --hazard-threshold 1

Assumptions:
 - hazard_label is numeric (higher means higher risk). For binary classification, we can threshold.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    return df


def prepare_labels(df, hazard_threshold=None):
    # If already binary, use as is. Otherwise threshold.
    y_raw = df['hazard_label'].values
    if hazard_threshold is not None:
        y = (y_raw >= hazard_threshold).astype(int)
    else:
        # Attempt to infer binary
        unique_vals = np.unique(y_raw)
        if len(unique_vals) == 2:
            y = y_raw.astype(int)
        else:
            # Fallback: median threshold
            thresh = np.nanmedian(y_raw)
            y = (y_raw >= thresh).astype(int)
    return y


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"\n{name} Performance:")
    print(f" Accuracy: {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall: {rec:.4f}")
    print(f" F1-score: {f1:.4f}")
    print("\nDetailed report:\n", classification_report(y_test, preds, zero_division=0))
    return {'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def plot_feature_importance(model, feature_names, title, out_path):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"Model {title} does not have feature_importances_. Skipping plot.")
        return
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], palette='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved feature importance plot to {out_path}")


def shap_analysis(model, X_train, feature_names, out_prefix):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_shap_summary.png")
        plt.close()
        print(f"Saved SHAP summary plot to {out_prefix}_shap_summary.png")
        # Bar plot
        plt.figure()
        shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_shap_bar.png")
        plt.close()
        print(f"Saved SHAP bar plot to {out_prefix}_shap_bar.png")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")


def kmeans_clustering(X, out_path, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X)
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette='deep', s=10)
    plt.title('KMeans Clusters (first two features)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved clustering plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/aligned_features.csv')
    parser.add_argument('--hazard-threshold', type=float, default=None, help='Threshold to binarize hazard label')
    parser.add_argument('--no-xgb', action='store_true', help='Skip XGBoost if issues arise')
    parser.add_argument('--clusters', type=int, default=0, help='Number of KMeans clusters (0 to skip)')
    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    df = load_data(csv_path)

    feature_cols = ['elevation', 'slope', 'ndvi', 'rainfall']
    X = df[feature_cols].values
    y = prepare_labels(df, args.hazard_threshold)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    results = []

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced')
    rf.fit(X_train, y_train)
    results.append(evaluate_model('RandomForest', rf, X_test, y_test))
    plot_feature_importance(rf, feature_cols, 'RandomForest Feature Importance', 'rf_feature_importance.png')
    shap_analysis(rf, X_train, feature_cols, 'rf')

    # XGBoost
    if not args.no_xgb:
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
            results.append(evaluate_model('XGBoost', xgb_model, X_test, y_test))
            plot_feature_importance(xgb_model, feature_cols, 'XGBoost Feature Importance', 'xgb_feature_importance.png')
            shap_analysis(xgb_model, X_train, feature_cols, 'xgb')
        except Exception as e:
            print(f"XGBoost training failed: {e}")

    # Optional clustering
    if args.clusters and args.clusters > 0:
        kmeans_clustering(X, 'kmeans_clusters.png', n_clusters=args.clusters)

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_results_summary.csv', index=False)
    print("Saved model performance summary to model_results_summary.csv")


if __name__ == '__main__':
    main()
