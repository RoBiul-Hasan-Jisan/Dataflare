import pandas as pd
import numpy as np
from io import StringIO, BytesIO

class DataProcessor:
    """Handles all data processing operations"""
    
    @staticmethod
    def process_uploaded_file(file):
        """Process uploaded file (CSV or Excel)"""
        filename = file.filename.lower()
        
        try:
            if filename.endswith('.csv'):
                # Try different encodings
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode file with common encodings")
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format")
            
            # Basic validation
            if df.empty:
                raise ValueError("File contains no data")
            
            # Convert types for JSON serialization
            for col in df.select_dtypes(include=['object', 'category']).columns:
                df[col] = df[col].astype(str)
            
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            metadata = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
            
            return df, metadata
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    @staticmethod
    def load_sample_dataset(sample_name):
        """Load sample datasets"""
        samples = {
            "titanic": {
                "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
                "suggested_target": "Survived"
            },
            "diamonds": {
                "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
                "suggested_target": "price"
            },
            "iris": {
                "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
                "suggested_target": "species"
            },
            "tips": {
                "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
                "suggested_target": "tip"
            }
        }
        
        if sample_name not in samples:
            raise ValueError("Invalid sample dataset")
        
        df = pd.read_csv(samples[sample_name]["url"])
        metadata = {
            'rows': len(df),
            'columns': len(df.columns),
            'suggested_target': samples[sample_name]["suggested_target"],
            'is_sample': True
        }
        
        return df, metadata
    
    @staticmethod
    def get_dataset_info(df):
        """Get comprehensive dataset information"""
        null_count = df.isnull().sum().sum()
        null_pct = round(null_count / df.size * 100, 2) if df.size > 0 else 0
        dup_count = int(df.duplicated().sum())
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Health score calculation
        health_score = round(max(0, 100 - null_pct * 2 - (dup_count / len(df) * 300 if len(df) > 0 else 0)), 1)
        
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'num_cols': len(num_cols),
            'cat_cols': len(cat_cols),
            'date_cols': len(date_cols),
            'null_pct': null_pct,
            'null_count': int(null_count),
            'duplicates': dup_count,
            'health_score': health_score,
            'column_names': df.columns.tolist(),
            'num_columns': num_cols,
            'cat_columns': cat_cols,
            'date_columns': date_cols,
            'dtypes': df.dtypes.astype(str).to_dict()
        }
    
    @staticmethod
    def get_statistical_summary(df):
        """Get statistical summary for numeric columns"""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not num_cols:
            return {}
        
        stats = df[num_cols].describe(percentiles=[.25, .5, .75]).round(3)
        
        # Add skewness and kurtosis
        for col in num_cols:
            stats.loc['skewness', col] = round(df[col].skew(), 3)
            stats.loc['kurtosis', col] = round(df[col].kurtosis(), 3)
        
        return stats.to_dict()
    
    @staticmethod
    def get_column_details(df):
        """Get detailed information for each column"""
        details = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = round(null_count / len(df) * 100, 1)
            unique_count = df[col].nunique()
            
            detail = {
                'column': col,
                'type': dtype,
                'non_null': int(df[col].count()),
                'null_count': int(null_count),
                'null_pct': null_pct,
                'unique': int(unique_count),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col])
            }
            
            if detail['is_numeric']:
                detail.update({
                    'mean': round(df[col].mean(), 4) if not pd.isna(df[col].mean()) else None,
                    'std': round(df[col].std(), 4) if not pd.isna(df[col].std()) else None,
                    'min': round(df[col].min(), 4) if not pd.isna(df[col].min()) else None,
                    'max': round(df[col].max(), 4) if not pd.isna(df[col].max()) else None
                })
            
            details.append(detail)
        
        return details
    
    @staticmethod
    def get_column_stats(df, column):
        """Get statistics for a specific column"""
        if column not in df.columns:
            return {}, False
        
        col_data = df[column]
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        stats = {
            'count': int(col_data.count()),
            'missing': int(col_data.isnull().sum()),
            'missing_pct': round(col_data.isnull().mean() * 100, 1),
            'unique': int(col_data.nunique())
        }
        
        if is_numeric:
            stats.update({
                'mean': round(col_data.mean(), 4) if not pd.isna(col_data.mean()) else None,
                'std': round(col_data.std(), 4) if not pd.isna(col_data.std()) else None,
                'min': round(col_data.min(), 4) if not pd.isna(col_data.min()) else None,
                'q1': round(col_data.quantile(0.25), 4) if not pd.isna(col_data.quantile(0.25)) else None,
                'median': round(col_data.median(), 4) if not pd.isna(col_data.median()) else None,
                'q3': round(col_data.quantile(0.75), 4) if not pd.isna(col_data.quantile(0.75)) else None,
                'max': round(col_data.max(), 4) if not pd.isna(col_data.max()) else None,
                'skew': round(col_data.skew(), 3) if not pd.isna(col_data.skew()) else None
            })
        
        return stats, is_numeric
    
    @staticmethod
    def analyze_target_column(target_series):
        """Analyze target column to determine problem type"""
        unique_count = target_series.nunique()
        total_count = len(target_series)
        
        if target_series.dtype == 'object' or target_series.dtype.name == 'category':
            problem_type = 'classification'
            type_label = 'Classification'
        elif target_series.dtype == 'bool':
            problem_type = 'classification'
            type_label = 'Classification'
        elif unique_count <= 10 and pd.api.types.is_integer_dtype(target_series):
            problem_type = 'classification'
            type_label = 'Classification'
        elif unique_count / total_count < 0.05 and unique_count <= 20:
            problem_type = 'classification'
            type_label = 'Classification'
        else:
            problem_type = 'regression'
            type_label = 'Regression'
        
        # Get value distribution for classification
        value_distribution = None
        if problem_type == 'classification' and unique_count <= 20:
            value_distribution = target_series.value_counts().head(10).to_dict()
        
        return {
            'problem_type': problem_type,
            'type_label': type_label,
            'unique_values': int(unique_count),
            'value_distribution': value_distribution,
            'recommended_models': 'Random Forest, XGBoost, LightGBM' if problem_type == 'classification' else 'Linear Regression, Random Forest, XGBoost'
        }
    
    @staticmethod
    def get_outlier_stats(df, column, method='iqr'):
        """Get outlier statistics for a column"""
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return {}
        
        col_data = df[column].dropna()
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        else:  # z-score
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outliers = col_data[z_scores > 3]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': round(len(outliers) / len(col_data) * 100, 2),
            'lower_bound': round(lower_bound if method == 'iqr' else col_data.mean() - 3 * col_data.std(), 4),
            'upper_bound': round(upper_bound if method == 'iqr' else col_data.mean() + 3 * col_data.std(), 4)
        }
    
    @staticmethod
    def clean_data(df, operation):
        """Apply data cleaning operations"""
        original_rows = len(df)
        
        if operation == 'drop_duplicates':
            df_cleaned = df.drop_duplicates().reset_index(drop=True)
            removed = original_rows - len(df_cleaned)
            message = f"Removed {removed} duplicate rows"
        elif operation == 'drop_null_cols':
            cols_before = len(df.columns)
            df_cleaned = df.dropna(axis=1, how='all')
            removed = cols_before - len(df_cleaned.columns)
            message = f"Removed {removed} completely empty columns"
        elif operation == 'drop_null_rows':
            df_cleaned = df.dropna()
            removed = original_rows - len(df_cleaned)
            message = f"Removed {removed} rows with missing values"
        elif operation == 'fill_numeric_median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_cleaned = df.copy()
            for col in numeric_cols:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            message = "Filled missing numeric values with median"
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return df_cleaned, message