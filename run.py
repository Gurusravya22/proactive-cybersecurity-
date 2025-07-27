import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def optimize_memory_usage(df):
    """Optimize memory usage for numeric columns."""
    for col in df.select_dtypes(include=["int", "float"]).columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df

def preprocess_data_advanced(dataset_file, target_column="label", test_size=0.2, random_state=42):
    """Advanced preprocessing pipeline for structured datasets."""
    # Load the dataset
    df = pd.read_parquet(dataset_file)
    
    # Check for target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing in the dataset.")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Optimize memory usage
    X = optimize_memory_usage(X)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Numeric preprocessing: Impute missing values and scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical preprocessing: Impute missing values and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # Combine preprocessors into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    # Apply preprocessing
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Convert preprocessed data to a dataframe for ease of use
    feature_names = (
        numeric_features +
        list(preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features))
    )
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=test_size, random_state=random_state
    )
    
    return (X_train, y_train), (X_test, y_test)

# Usage example
dataset_file = r"C:\sravss\major pro\major cyber\KDDTrain.txt"

if not os.path.exists(dataset_file):
    print(f"Dataset file does not exist: {dataset_file}")
else:
    try:
        (X_train, y_train), (X_test, y_test) = preprocess_data_advanced(dataset_file)
        print("Data preprocessed successfully.")
        print(f"Training set size: {X_train.shape[0]} rows, {X_train.shape[1]} features")
        print(f"Testing set size: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    except PermissionError as e:
        print(f"Permission denied: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
