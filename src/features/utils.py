import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Iterate through all columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not isinstance(col_type, pd.CategoricalDtype):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int' or str(col_type)[:4] == 'uint':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # Avoid float16 due to torch/numpy compat issues sometimes
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        print(f"Decreased by {decrease:.1f}%")
    return df

class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """Custom LabelEncoder that handles unseen labels during transform by mapping them to an 'Unknown' category."""
    def __init__(self):
        self.mapping = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            X = X.fillna("Missing").astype(str)
            unique_vals = X.unique()
            self.mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
        else:
            X_df = pd.DataFrame(X)
            self.mapping = {}
            for col in X_df.columns:
                series = X_df[col].fillna("Missing").astype(str)
                unique_vals = series.unique()
                self.mapping[col] = {val: i for i, val in enumerate(sorted(unique_vals))}
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            series = X.fillna("Missing").astype(str)
            # Map unknown values to len(mapping) (last index + 1)
            unknown_val = len(self.mapping)
            return series.map(self.mapping).fillna(unknown_val).astype(int)
        else:
            X_df = pd.DataFrame(X)
            X_transformed = X_df.copy()
            for col in X_df.columns:
                series = X_df[col].fillna("Missing").astype(str)
                col_map = self.mapping.get(col, {})
                unknown_val = len(col_map)
                X_transformed[col] = series.map(col_map).fillna(unknown_val).astype(int)
            return X_transformed

class MedianImputer(BaseEstimator, TransformerMixin):
    """Imputes numeric features with their median. Stores medians during fit."""
    def __init__(self):
        self.medians = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.medians = {}
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                self.medians[col] = X_df[col].median()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_transformed = X_df.copy()
        for col in X_df.columns:
            if col in self.medians:
                # If median is nan (all values were nan), fill with 0
                fill_val = self.medians[col]
                if pd.isna(fill_val):
                    fill_val = 0.0
                X_transformed[col] = X_transformed[col].fillna(fill_val)
        return X_transformed

def one_hot_encode(df: pd.DataFrame, dummy_cols: list = None) -> tuple:
    """Performs one-hot encoding on the dataframe for object/category types or specific columns."""
    if dummy_cols is None:
        dummy_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    df_encoded = pd.get_dummies(df, columns=dummy_cols, dummy_na=True)
    new_cols = [c for c in df_encoded.columns if c not in df.columns]
    return df_encoded, new_cols

def safe_divide(numerator, denominator, fill_value=0.0):
    """Safely divides numerator by denominator handling zeros and NaNs."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        if isinstance(result, pd.Series) or isinstance(result, np.ndarray):
            result = np.where(np.isfinite(result), result, fill_value)
        else:
            result = fill_value if not np.isfinite(result) else result
    return result
