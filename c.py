import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
import os
import gc
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    try:
        # Only compute these if applicable (binary classification)
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auprc = auc(recall, precision)
        
        # Get ROC curve data for plotting
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data = (fpr, tpr)
    except (IndexError, ValueError):
        # Multi-class or other issue
        roc_auc = None
        auprc = None
        roc_data = None
        
    return report, roc_auc, auprc, roc_data

def plot_roc_curves(roc_data_dict, dataset_name):
    """
    Plot ROC curves for all models on a single graph
    
    Parameters:
    roc_data_dict: Dictionary with model names as keys and (fpr, tpr) tuples as values
    dataset_name: Name of the dataset for title
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, (fpr, tpr) in roc_data_dict.items():
        if fpr is not None and tpr is not None:
            plt.plot(fpr, tpr, lw=2, label=f'{model_name}')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {dataset_name}')
    plt.legend(loc="lower right")
    
    # Save the figure
    filename = f"{dataset_name.replace(' ', '_')}_roc_curves.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {filename}")
    plt.close()

def plot_auc_comparison(results_df, dataset_name):
    """
    Create a bar chart comparing AUC values for different models
    
    Parameters:
    results_df: DataFrame with model comparison results
    dataset_name: Name of the dataset for filtering and title
    """
    # Filter results for this dataset
    df = results_df[results_df['Dataset'] == dataset_name]
    
    # Check if ROC-AUC column exists
    if 'ROC-AUC' not in df.columns:
        print(f"No ROC-AUC data available for {dataset_name}")
        return
    
    # Sort by ROC-AUC value
    df = df.sort_values('ROC-AUC', ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = plt.bar(df['Model'], df['ROC-AUC'], color='skyblue')
    
    # Add value labels on top of each bar
    for i, v in enumerate(df['ROC-AUC']):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.ylim(0, 1.1)
    plt.xlabel('Model')
    plt.ylabel('AUROC Score')
    plt.title(f'AUROC Comparison for {dataset_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    filename = f"{dataset_name.replace(' ', '_')}_auroc_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"AUROC comparison saved to {filename}")
    plt.close()

def run_experiments(dataset_name, X_train, X_test, y_train, y_test):
    results = []
    roc_data_dict = {}  # Dictionary to store ROC curve data for plotting
    
    # Use simpler parameters for GridSearchCV to make it faster
    n_cv = 2  # Reduced from 3
    
    # Random Forest
    print(f"Training Random Forest on {dataset_name}...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_params = {'n_estimators': [100], 'max_depth': [10]}  # Simplified
    rf_grid = GridSearchCV(rf, param_grid=rf_params, cv=n_cv, scoring='f1_weighted')
    try:
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_
        rf_report, rf_roc_auc, rf_auprc, rf_roc_data = evaluate_model(best_rf, X_test, y_test)
        results.append(('Random Forest', rf_report, rf_roc_auc, rf_auprc))
        roc_data_dict['Random Forest'] = rf_roc_data
        print("  Random Forest training completed successfully")
    except Exception as e:
        print(f"  Error in Random Forest: {str(e)}")
    
    del rf_grid
    gc.collect()

    # AdaBoost
    print(f"Training AdaBoost on {dataset_name}...")
    ada = AdaBoostClassifier(random_state=42)
    ada_params = {'n_estimators': [50], 'learning_rate': [0.1]}  # Simplified
    ada_grid = GridSearchCV(ada, param_grid=ada_params, cv=n_cv, scoring='f1_weighted')
    try:
        ada_grid.fit(X_train, y_train)
        best_ada = ada_grid.best_estimator_
        ada_report, ada_roc_auc, ada_auprc, ada_roc_data = evaluate_model(best_ada, X_test, y_test)
        results.append(('AdaBoost', ada_report, ada_roc_auc, ada_auprc))
        roc_data_dict['AdaBoost'] = ada_roc_data
        print("  AdaBoost training completed successfully")
    except Exception as e:
        print(f"  Error in AdaBoost: {str(e)}")
    
    del ada_grid
    gc.collect()

    # XGBoost
    print(f"Training XGBoost on {dataset_name}...")
    xgb = XGBClassifier(random_state=42)
    xgb_params = {'n_estimators': [100], 'max_depth': [3]}  # Simplified
    xgb_grid = GridSearchCV(xgb, param_grid=xgb_params, cv=n_cv, scoring='f1_weighted')
    try:
        xgb_grid.fit(X_train, y_train)
        best_xgb = xgb_grid.best_estimator_
        xgb_report, xgb_roc_auc, xgb_auprc, xgb_roc_data = evaluate_model(best_xgb, X_test, y_test)
        results.append(('XGBoost', xgb_report, xgb_roc_auc, xgb_auprc))
        roc_data_dict['XGBoost'] = xgb_roc_data
        print("  XGBoost training completed successfully")
    except Exception as e:
        print(f"  Error in XGBoost: {str(e)}")
    
    del xgb_grid
    gc.collect()

    # LightGBM
    print(f"Training LightGBM on {dataset_name}...")
    lgb = LGBMClassifier(random_state=42)
    lgb_params = {'n_estimators': [100], 'max_depth': [3]}  # Simplified
    lgb_grid = GridSearchCV(lgb, param_grid=lgb_params, cv=n_cv, scoring='f1_weighted')
    try:
        lgb_grid.fit(X_train, y_train)
        best_lgb = lgb_grid.best_estimator_
        lgb_report, lgb_roc_auc, lgb_auprc, lgb_roc_data = evaluate_model(best_lgb, X_test, y_test)
        results.append(('LightGBM', lgb_report, lgb_roc_auc, lgb_auprc))
        roc_data_dict['LightGBM'] = lgb_roc_data
        print("  LightGBM training completed successfully")
    except Exception as e:
        print(f"  Error in LightGBM: {str(e)}")
    
    del lgb_grid
    gc.collect()

    # Create the results table
    if not results:
        print(f"No successful models for {dataset_name}")
        return None
        
    table = []
    for model_name, report, roc_auc, auprc in results:
        row = {
            'Dataset': dataset_name,
            'Model': model_name,
            'Accuracy': report['accuracy'],
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score']
        }
        
        # Add ROC-AUC and AUPRC if available
        if roc_auc is not None:
            row['ROC-AUC'] = roc_auc
        if auprc is not None:
            row['AUPRC'] = auprc
            
        table.append(row)
    
    # Create a DataFrame from the results
    results_df = pd.DataFrame(table)
    
    # Plot ROC curves
    if any(data is not None for data in roc_data_dict.values()):
        plot_roc_curves(roc_data_dict, dataset_name)
        # Plot AUC comparison
        plot_auc_comparison(results_df, dataset_name)
    
    return results_df

def preprocess_dataset(df, dataset_name):
    """Preprocess the dataset based on its specific requirements"""
    print(f"Preprocessing {dataset_name}...")
    
    # Handle specific preprocessing for each dataset
    if dataset_name == "CICIDS 2017" or dataset_name == "CICIDS 2018":
        # Drop the Timestamp column which is causing errors
        if 'Timestamp' in df.columns:
            print(f"  Dropping Timestamp column")
            df = df.drop('Timestamp', axis=1)
        
        # Handle label encoding for CICIDS datasets
        print(f"  Converting Label to numeric format")
        # Ensure Label is treated as string first
        df['Label'] = df['Label'].astype(str)
        # Map specific attack types if found in typical CICIDS datasets
        attack_map = {
            'BENIGN': 0,
            'DoS': 1, 'DoS Hulk': 1, 'DoS GoldenEye': 1, 'DoS slowloris': 1, 'DoS Slowhttptest': 1,
            'DDoS': 2,
            'PortScan': 3,
            'Bot': 4,
            'Infiltration': 5,
            'Web Attack': 6, 'Web Attack – Brute Force': 6, 'Web Attack – XSS': 6, 'Web Attack – Sql Injection': 6,
            'SSH-Patator': 7,
            'FTP-Patator': 8,
            'Heartbleed': 9
        }
        
        # Apply mapping where possible, otherwise use label encoding
        label_col = 'Label'
        known_labels = set(attack_map.keys())
        unknown_labels = set(df[label_col].unique()) - known_labels
        
        if unknown_labels:
            print(f"  Found unknown labels: {unknown_labels}")
            # Use label encoding for all if unknown labels exist
            df[label_col], _ = pd.factorize(df[label_col])
        else:
            # Use predefined mapping
            df[label_col] = df[label_col].map(attack_map)
    
    elif dataset_name == "UNSW-NB15":
        # UNSW-NB15 specific preprocessing
        pass
    
    elif dataset_name == "ISCX-IDS 2012":
        # ISCX-IDS 2012 specific preprocessing - handle spaces in column names
        # Strip spaces from column names
        df.columns = [col.strip() for col in df.columns]
        
    # Common preprocessing for all datasets
    # 1. Convert object columns to category to save memory
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Label':  # Don't convert Label if it's still a string
            df[col] = df[col].astype('category')
    
    # 2. Handle infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 3. Handle NaN values
    print(f"  Shape before dropping NA: {df.shape}")
    df = df.dropna()
    print(f"  Shape after dropping NA: {df.shape}")
    
    return df

# Load and preprocess the datasets
file_paths = {
    "CICIDS 2017": r"C:\sravss\major pro\ids 2018\02-14-2018.csv",
    "CICIDS 2018": r"C:\sravss\major pro\ids 2018\02-16-2018.csv",
    "UNSW-NB15": r"C:\sravss\major pro\proactive\archive (3)\NUSW-NB15_features.csv",
    "ISCX-IDS 2012": r"C:\sravss\major pro\proactive\archive (1)\DrDoS_DNS_data_1_per.csv"
}

# Define the label column name for each dataset
label_columns = {
    "CICIDS 2017": "Label",
    "CICIDS 2018": "Label",
    "UNSW-NB15": "attack_cat",
    "ISCX-IDS 2012": "Label"  # Fixed: removed the space before 'Label'
}

# Process each dataset individually to save memory
all_results = []

for dataset_name, path in file_paths.items():
    print(f"\nProcessing dataset: {dataset_name}")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
        
    try:
        # Use chunking to read large files
        print(f"Reading file: {path}")
        # First check if it's a large file
        file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        
        # Get the expected label column name for this dataset
        label_col = label_columns.get(dataset_name, "label")
        print(f"Looking for label column: {label_col}")
        
        # Special handling for CICIDS datasets which have header issues
        if dataset_name in ["CICIDS 2017", "CICIDS 2018"]:
            # First check the header to detect if there are issues
            try:
                # Sample first few rows
                sample = pd.read_csv(path, nrows=5, encoding='ISO-8859-1', low_memory=False, on_bad_lines='skip')
                print(f"Sample data columns: {sample.columns.tolist()}")
                
                # Check if we need to skip rows (header might be repeated in data)
                if 'Dst Port' in sample.columns[0]:
                    print("Header detected in first column - skipping first row")
                    
                    # Try reading with header=None and set header manually
                    full_data = pd.read_csv(path, encoding='ISO-8859-1', low_memory=False, 
                                         on_bad_lines='skip', nrows=500000)  # Limit rows for testing
                    
                    # Clean the data
                    data = preprocess_dataset(full_data, dataset_name)
                    print(f"Processed data shape: {data.shape}")
                else:
                    # Normal read
                    full_data = pd.read_csv(path, encoding='ISO-8859-1', low_memory=False, 
                                         on_bad_lines='skip', nrows=500000)  # Limit rows for testing
                    
                    # Clean the data
                    data = preprocess_dataset(full_data, dataset_name)
                    print(f"Processed data shape: {data.shape}")
            except Exception as e:
                print(f"Error in sampling: {str(e)}")
                continue
        else:
            # For other datasets, use standard chunking approach
            if file_size > 500:
                print("Large file detected. Using chunked reading...")
                try:
                    # Read in chunks
                    chunks = []
                    chunk_size = 100000
                    for chunk_idx, chunk in enumerate(pd.read_csv(path, encoding='ISO-8859-1', 
                                                             chunksize=chunk_size, low_memory=False, 
                                                             on_bad_lines='skip')):
                        if chunk_idx > 4:  # Limit to 5 chunks for testing
                            break
                        chunks.append(chunk)
                        
                    if not chunks:
                        print("No data chunks read")
                        continue
                        
                    # Combine chunks
                    data = pd.concat(chunks, ignore_index=True)
                    print(f"Data shape from chunks: {data.shape}")
                    
                    # Clean column names by stripping whitespaces before column check
                    data.columns = data.columns.str.strip()
                    
                    # Check if label column exists with different capitalization
                    actual_label_col = None
                    for col in data.columns:
                        if col.lower() == label_col.lower():
                            actual_label_col = col
                            print(f"Found label column: {actual_label_col}")
                            break
                    
                    if not actual_label_col:
                        print(f"Dataset {dataset_name} does not contain '{label_col}' column. Columns: {data.columns.tolist()}")
                        continue
                        
                    # Use the actual column name found
                    label_col = actual_label_col
                    
                    # Clean the data
                    data = preprocess_dataset(data, dataset_name)
                    print(f"Processed data shape: {data.shape}")
                    
                    # Clean up chunks to free memory
                    del chunks
                    gc.collect()
                except Exception as e:
                    print(f"Error in chunked reading: {str(e)}")
                    continue
            else:
                try:
                    # Read the entire file for smaller datasets
                    data = pd.read_csv(path, encoding='ISO-8859-1', low_memory=False, on_bad_lines='skip')
                    print(f"Data shape: {data.shape}")
                    
                    # Check column names
                    print(f"Available columns: {data.columns.tolist()}")
                    
                    # Strip whitespace from column names
                    data.columns = data.columns.str.strip()
                    
                    # Check if label column exists with different capitalization
                    actual_label_col = None
                    for col in data.columns:
                        if col.lower() == label_col.lower():
                            actual_label_col = col
                            print(f"Found label column: {actual_label_col}")
                            break
                    
                    if not actual_label_col:
                        print(f"Dataset {dataset_name} does not contain '{label_col}' column or any variation of it. Skipping...")
                        continue
                        
                    # Use the actual column name found
                    label_col = actual_label_col
                    
                    # Clean the data
                    data = preprocess_dataset(data, dataset_name)
                    print(f"Processed data shape: {data.shape}")
                except Exception as e:
                    print(f"Error reading file: {str(e)}")
                    continue
        
        # Prepare data for modeling
        try:
            # Strip spaces from column names again just to be sure
            data.columns = data.columns.str.strip()
            
            # Now try to drop the label column
            X = data.drop(label_col, axis=1)
            y = data[label_col]
        except KeyError as e:
            print(f"Error: Column '{label_col}' not found in DataFrame.")
            print(f"Available columns: {data.columns.tolist()}")
            continue
        
        # Convert categorical columns to numeric for models
        for col in X.select_dtypes(include=['category']).columns:
            X[col] = X[col].cat.codes
            
        # Handle any remaining non-numeric columns
        for col in X.select_dtypes(include=['object']).columns:
            X[col], _ = pd.factorize(X[col])
        
        # Convert X to float32 to save memory
        X = X.astype('float32')
        
        # Clean up original dataframe to free memory
        del data
        gc.collect()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Clean up to free memory
        del X, y
        gc.collect()
        
        # Run experiments
        result_df = run_experiments(dataset_name, X_train, X_test, y_train, y_test)
        if result_df is not None:
            all_results.append(result_df)
        
        # Clean up to free memory
        del X_train, X_test, y_train, y_test
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Combine results
if all_results:
    final_results = pd.concat(all_results, ignore_index=True)
    print("\nFinal Results:")
    print(final_results)
    
    # Plot overall AUROC comparison across all datasets
    plt.figure(figsize=(12, 8))
    
    # Group by Dataset and Model, calculate mean of ROC-AUC
    if 'ROC-AUC' in final_results.columns:
        pivot_data = final_results.pivot(index='Model', columns='Dataset', values='ROC-AUC')
        
        # Plot heatmap of AUROC scores
        plt.figure(figsize=(14, 8))
        ax = plt.axes()
        im = ax.imshow(pivot_data.values, cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='AUROC Score')
        
        # Add x and y labels
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticklabels(pivot_data.index)
        
        # Rotate the x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations in the cells
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = pivot_data.values[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.7 else "black")
        
        plt.title('AUROC Scores Across Datasets and Models')
        plt.tight_layout()
        plt.savefig("all_datasets_auroc_comparison.png", dpi=300, bbox_inches='tight')
        print("Overall AUROC comparison saved to all_datasets_auroc_comparison.png")
        plt.close()
    
    # Save results to CSV
    final_results.to_csv("model_comparison_results.csv", index=False)
    print("\nResults saved to 'model_comparison_results.csv'")
else:
    print("No results to display. Check if the datasets contain the expected label columns.")