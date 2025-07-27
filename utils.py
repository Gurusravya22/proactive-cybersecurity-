import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
import os
from matplotlib.lines import Line2D

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Get predictions for ROC and PR curves
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC curve values
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # PR curve values
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall, precision)
    
    return report, roc_auc, auprc, fpr, tpr, precision, recall

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plots - 2 plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot ROC curves
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title(f'ROC Curves - {dataset_name}', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Plot PR curves
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_title(f'Precision-Recall Curves - {dataset_name}', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
   
def run_experiments(dataset_name, X_train, X_test, y_train, y_test):
    results = []
    curve_data = []

    # Dictionary to store model colors for consistent plotting
    model_colors = {
        'Random Forest': 'blue',
        'AdaBoost': 'red',
        'XGBoost': 'green',
        'LightGBM': 'purple'
    }

    # Random Forest
    print(f"Training Random Forest for {dataset_name}...")
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, param_grid={'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}, cv=5, scoring='f1_weighted')
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    rf_report, rf_roc_auc, rf_auprc, rf_fpr, rf_tpr, rf_precision, rf_recall = evaluate_model(best_rf, X_test, y_test)
    results.append(('Random Forest', rf_report, rf_roc_auc, rf_auprc))
    curve_data.append(('Random Forest', rf_fpr, rf_tpr, rf_precision, rf_recall, model_colors['Random Forest']))

    # AdaBoost
    print(f"Training AdaBoost for {dataset_name}...")
    ada = AdaBoostClassifier(random_state=42)
    ada_grid = GridSearchCV(ada, param_grid={'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1.0]}, cv=5, scoring='f1_weighted')
    ada_grid.fit(X_train, y_train)
    best_ada = ada_grid.best_estimator_
    ada_report, ada_roc_auc, ada_auprc, ada_fpr, ada_tpr, ada_precision, ada_recall = evaluate_model(best_ada, X_test, y_test)
    results.append(('AdaBoost', ada_report, ada_roc_auc, ada_auprc))
    curve_data.append(('AdaBoost', ada_fpr, ada_tpr, ada_precision, ada_recall, model_colors['AdaBoost']))

    # XGBoost
    print(f"Training XGBoost for {dataset_name}...")
    xgb = XGBClassifier(random_state=42)
    xgb_grid = GridSearchCV(xgb, param_grid={'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}, cv=5, scoring='f1_weighted')
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    xgb_report, xgb_roc_auc, xgb_auprc, xgb_fpr, xgb_tpr, xgb_precision, xgb_recall = evaluate_model(best_xgb, X_test, y_test)
    results.append(('XGBoost', xgb_report, xgb_roc_auc, xgb_auprc))
    curve_data.append(('XGBoost', xgb_fpr, xgb_tpr, xgb_precision, xgb_recall, model_colors['XGBoost']))

    # LightGBM
    print(f"Training LightGBM for {dataset_name}...")
    lgb = LGBMClassifier(random_state=42)
    lgb_grid = GridSearchCV(lgb, param_grid={'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}, cv=5, scoring='f1_weighted')
    lgb_grid.fit(X_train, y_train)
    best_lgb = lgb_grid.best_estimator_
    lgb_report, lgb_roc_auc, lgb_auprc, lgb_fpr, lgb_tpr, lgb_precision, lgb_recall = evaluate_model(best_lgb, X_test, y_test)
    results.append(('LightGBM', lgb_report, lgb_roc_auc, lgb_auprc))
    curve_data.append(('LightGBM', lgb_fpr, lgb_tpr, lgb_precision, lgb_recall, model_colors['LightGBM']))

    # Create the results table
    table = []
    for model_name, report, roc_auc, auprc in results:
        table.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Accuracy (%)': round(report['accuracy'] * 100, 2),
            'Precision (%)': round(report['macro avg']['precision'] * 100, 2),
            'Recall (%)': round(report['macro avg']['recall'] * 100, 2),
            'F1-Score (%)': round(report['macro avg']['f1-score'] * 100, 2),
            'ROC-AUC': round(roc_auc, 4),
            'AUPRC': round(auprc, 4)
        })
    
    # Plot the curves
    plot_curves(dataset_name, curve_data)
    
    return pd.DataFrame(table), curve_data

# Main function to run the full experiment
def main():
    # Load and preprocess the datasets
    file_paths = {
        "CICIDS 2017": r"C:\sravss\major pro\ids 2018\02-14-2018.csv",
        "CICIDS 2018": r"C:\sravss\major pro\ids 2018\02-16-2018.csv",
        "UNSW-NB15": r"C:\sravss\major pro\proactive\archive (3)\NUSW-NB15_features.csv",
        "ISCX-IDS 2012": r"C:\sravss\major pro\proactive\archive (1)\DrDoS_DNS_data_1_per.csv"
    }

    datasets = {}
    for dataset_name, path in file_paths.items():
        if os.path.exists(path):
            print(f"Loading dataset: {dataset_name}")
            # Specify encoding as 'ISO-8859-1' and set low_memory=False
            datasets[dataset_name] = pd.read_csv(path, encoding='ISO-8859-1', low_memory=False)
        else:
            print(f"File not found: {path}")

    # Preprocess the data
    results = []
    for dataset_name, data in datasets.items():
        if 'label' not in data.columns:
            print(f"Dataset {dataset_name} does not contain 'label' column. Skipping...")
            continue
        
        print(f"\nProcessing dataset: {dataset_name}")
        X, y = data.drop('label', axis=1), data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        df_results, _ = run_experiments(dataset_name, X_train, X_test, y_train, y_test)
        results.append(df_results)

    # Combine results
    if results:
        final_results = pd.concat(results, ignore_index=True)
        
        # Display a nicely formatted table with percentage metrics
        print("\n=== MODEL PERFORMANCE COMPARISON ===\n")
        
        # Group by dataset and create comparison tables
        for dataset, group in final_results.groupby('Dataset'):
            print(f"\nDataset: {dataset}")
            comparison_table = group[['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC-AUC', 'AUPRC']].reset_index(drop=True)
            
            # Format the output to align columns nicely
            print(comparison_table.to_string(index=False))
            print("\n" + "-" * 80)
            
        # Save results to CSV
        final_results.to_csv("model_comparison_results.csv", index=False)
        print("\nResults saved to model_comparison_results.csv")
    else:
        print("No results to display. Check if the datasets contain the 'label' column.")

if __name__ == "__main__":
    main()