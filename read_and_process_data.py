import pandas as pd
import os

def preprocess_data_in_chunks(dataset_folder, target_column='label', chunk_size=100000):
   
    all_data = []
    
    # Recursively walk through all files in the folder
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.csv'):  # Adjust this if files are not CSV
                file_path = os.path.join(root, file)
                print(f"Reading {file_path}")
                try:
                    # Read the CSV file in chunks
                    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
                    for chunk in chunk_iter:
                        all_data.append(chunk)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    if not all_data:
        raise FileNotFoundError("No CSV files found in the dataset folder.")
    
    # Combine all data into a single DataFrame
    data = pd.concat(all_data, ignore_index=True)
    
    # Ensure the target column exists
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")
    
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Remove rows where the target column has NaN values
    data_cleaned = data.dropna(subset=[target_column])

    # Drop rows with NaN in the features (X) columns as well
    data_cleaned = data_cleaned.dropna(subset=X.columns)

    # Re-assign cleaned data to X and y
    X = data_cleaned.drop(columns=[target_column])
    y = data_cleaned[target_column]

    # Final alignment check to ensure matching number of samples
    if len(X) != len(y):
        raise ValueError(f"After cleaning, mismatched number of rows: X = {len(X)}, y = {len(y)}")

    print(f"After processing - X shape: {X.shape}, y shape: {y.shape}")

    # Split into train-test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Data preprocessing complete.")
    return X_train, X_test, y_train, y_test
