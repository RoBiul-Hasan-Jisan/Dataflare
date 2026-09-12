import os
import time
import gc
import uuid
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import (
    setup as clf_setup, compare_models as clf_compare,
    pull as clf_pull, save_model as clf_save
)
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare,
    pull as reg_pull, save_model as reg_save
)

warnings.filterwarnings("ignore")

class ModelTrainer:
    """Handles all model training operations"""
    
    # Model configurations
    MAX_ROWS_TRAINING = 5000
    MAX_ROWS_WARNING = 2000
    SAMPLE_RANDOM_STATE = 42
    
    ALL_CLF_MODELS = ["lr", "dt", "rf", "et", "ridge", "knn", "nb", "ada", 
                      "xgboost", "lightgbm", "catboost", "gbc", "lda"]
    ALL_REG_MODELS = ["lr", "dt", "rf", "et", "ridge", "lasso", "knn", "ada", 
                      "en", "xgboost", "lightgbm", "catboost", "gbr", "br"]
    
    def __init__(self, model_folder):
        self.model_folder = model_folder
        os.makedirs(model_folder, exist_ok=True)
    
    def _get_memory_usage_mb(self):
        """Get current memory usage"""
        try:
            import psutil
            import os
            return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except:
            return 0
    
    def _force_gc(self):
        """Force garbage collection"""
        gc.collect()
        gc.collect()
    
    def _smart_sample(self, df, target_col, max_rows):
        """Smart sampling with stratification for classification"""
        if len(df) <= max_rows:
            return df
        
        try:
            target_series = df[target_col]
            if target_series.dtype == "object" or target_series.nunique() <= 20:
                _, sampled = train_test_split(
                    df, test_size=max_rows / len(df),
                    stratify=target_series, random_state=self.SAMPLE_RANDOM_STATE
                )
                return sampled.reset_index(drop=True)
        except:
            pass
        
        return df.sample(n=max_rows, random_state=self.SAMPLE_RANDOM_STATE).reset_index(drop=True)
    
    def _detect_problem_type(self, series):
        """Detect if problem is classification or regression"""
        if series.dtype == "object" or str(series.dtype) == "category":
            return "classification"
        if series.dtype == "bool":
            return "classification"
        unique_count = series.nunique()
        total_count = len(series)
        if unique_count <= 10 and pd.api.types.is_integer_dtype(series):
            return "classification"
        if unique_count / total_count < 0.05 and unique_count <= 20:
            return "classification"
        return "regression"
    
    def train_model(self, df, target_col, train_size=0.8, fold=5,
                   normalize=True, remove_outliers=False, max_models=10):
        """Execute model training with memory safety"""
        warnings_list = []
        start_time = time.time()
        
        # Detect problem type
        problem_type = self._detect_problem_type(df[target_col])
        
        # Handle large datasets
        original_rows = len(df)
        if original_rows > self.MAX_ROWS_TRAINING:
            df_train = self._smart_sample(df, target_col, self.MAX_ROWS_TRAINING)
            warnings_list.append(
                f"Dataset {original_rows:,} rows — auto-sampled to {self.MAX_ROWS_TRAINING:,} rows for training."
            )
        elif original_rows > self.MAX_ROWS_WARNING:
            df_train = df.copy()
            warnings_list.append(
                f"Dataset {original_rows:,} rows — training may be slow. Consider using a sample."
            )
        else:
            df_train = df.copy()
        
        # Select models
        include_models = self.ALL_CLF_MODELS if problem_type == "classification" else self.ALL_REG_MODELS
        if max_models and max_models < len(include_models):
            include_models = include_models[:max_models]
        
        # Memory management
        mem_before = self._get_memory_usage_mb()
        if mem_before > 400:
            self._force_gc()
        
        # Setup PyCaret
        setup_kwargs = {
            'data': df_train,
            'target': target_col,
            'train_size': float(train_size),
            'fold': int(fold),
            'normalize': normalize,
            'verbose': False,
            'html': False,
            'session_id': 42,
            'n_jobs': 1,
            'use_gpu': False,
        }
        
        if remove_outliers and problem_type == "regression" and len(df_train) > 100:
            setup_kwargs["remove_outliers"] = True
        
        try:
            if problem_type == "classification":
                clf_setup(**setup_kwargs)
                pull_fn, save_fn, cmp_fn = clf_pull, clf_save, clf_compare
            else:
                reg_setup(**setup_kwargs)
                pull_fn, save_fn, cmp_fn = reg_pull, reg_save, reg_compare
        except Exception as e:
            if "memory" in str(e).lower() or "killed" in str(e).lower():
                raise MemoryError(f"Setup failed due to memory constraints.")
            raise
        
        self._force_gc()
        
        # Compare models
        try:
            best = cmp_fn(verbose=False, n_select=1, include=include_models, errors="ignore")
            results_df = pull_fn()
        except MemoryError:
            self._force_gc()
            light_models = ["lr", "dt", "ridge"]
            warnings_list.append("Memory issue — using only 3 lightest models.")
            best = cmp_fn(verbose=False, n_select=1, include=light_models)
            results_df = pull_fn()
        except Exception as e:
            if any(k in str(e).lower() for k in ["memory", "killed", "oom"]):
                raise MemoryError("Model comparison failed due to memory.")
            raise
        
        # Save model
        model_id = str(uuid.uuid4())[:8]
        model_path = os.path.join(self.model_folder, f"best_model_{model_id}")
        try:
            save_fn(best, model_path)
        except:
            pass
        
        # Extract results
        model_col = "Model" if "Model" in results_df.columns else results_df.columns[0]
        metric_cols = results_df.select_dtypes(include=[np.number]).columns
        best_model_name = str(results_df.iloc[0][model_col])
        best_score = float(results_df.iloc[0][metric_cols[0]]) if len(metric_cols) > 0 else 0.0
        metric_name = metric_cols[0] if len(metric_cols) > 0 else "Score"
        
        self._force_gc()
        elapsed = time.time() - start_time
        
        return {
            'best_model': best_model_name,
            'best_score': best_score,
            'metric_name': metric_name,
            'results_df': results_df,
            'elapsed': elapsed,
            'trained_rows': len(df_train),
            'warnings': warnings_list,
            'model_id': model_id,
            'problem_type': problem_type
        }
    
    def get_model_path(self, model_id):
        """Get model file path"""
        model_path = os.path.join(self.model_folder, f"best_model_{model_id}.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_folder, f"best_model_{model_id}")
        return model_path