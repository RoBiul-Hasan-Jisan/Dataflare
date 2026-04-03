import os
import io
import gc
import time
import uuid
import warnings
import base64
from datetime import datetime
from functools import wraps
from collections import Counter

import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.utils
import json
from flask import (
    Flask, render_template, request, jsonify, 
    send_file, session
)
from flask_session import Session
from pycaret.classification import (
    setup as clf_setup, compare_models as clf_compare,
    pull as clf_pull, save_model as clf_save
)
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare,
    pull as reg_pull, save_model as reg_save
)
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Flask App Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = '/tmp/Dataflare_uploads'
app.config['MODEL_FOLDER'] = '/tmp/Dataflare_models'
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize server-side session
Session(app)

# Memory-Safe Training Config
MAX_ROWS_TRAINING = 5_000
MAX_ROWS_WARNING = 2_000
SAMPLE_RANDOM_STATE = 42

ALL_CLF_MODELS = ["lr", "dt", "rf", "et", "ridge", "knn", "nb", "ada", 
                  "xgboost", "lightgbm", "catboost", "gbc", "lda"]
ALL_REG_MODELS = ["lr", "dt", "rf", "et", "ridge", "lasso", "knn", "ada", 
                  "en", "xgboost", "lightgbm", "catboost", "gbr", "br"]

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    return obj

def get_memory_usage_mb():
    """Get current memory usage in MB"""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

def force_gc():
    """Force garbage collection"""
    gc.collect()
    gc.collect()

def smart_sample(df, target_col, max_rows=MAX_ROWS_TRAINING):
    """Smart sampling with stratification for classification"""
    if len(df) <= max_rows:
        return df
    try:
        target_series = df[target_col]
        if target_series.dtype == "object" or target_series.nunique() <= 20:
            from sklearn.model_selection import train_test_split
            _, sampled = train_test_split(
                df, test_size=max_rows / len(df),
                stratify=target_series, random_state=SAMPLE_RANDOM_STATE
            )
            return sampled.reset_index(drop=True)
    except Exception:
        pass
    return df.sample(n=max_rows, random_state=SAMPLE_RANDOM_STATE).reset_index(drop=True)

def detect_problem_type(s):
    """Detect if problem is classification or regression"""
    if s.dtype == "object" or str(s.dtype) == "category":
        return "classification"
    if s.dtype == "bool":
        return "classification"
    u, n = s.nunique(), len(s)
    if u <= 10 and pd.api.types.is_integer_dtype(s):
        return "classification"
    if u / n < 0.05 and u <= 20:
        return "classification"
    return "regression"

def fmt_time(seconds):
    """Format time duration"""
    if seconds >= 60:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    return f"{seconds:.1f}s"

def generate_smart_insights(df):
    """Generate top 5 insights from dataset"""
    insights = []
    
    total_rows = len(df)
    total_cols = len(df.columns)
    null_pct = (df.isnull().sum().sum() / (total_rows * total_cols)) * 100 if total_rows * total_cols > 0 else 0
    
    insights.append({
        'title': 'Dataset Overview',
        'description': f'Dataset contains {total_rows:,} rows and {total_cols} columns',
        'severity': 'info',
        'metric': f'{total_rows:,} × {total_cols}',
        'action': 'Review column types and distributions below'
    })
    
    if null_pct > 0:
        insights.append({
            'title': 'Missing Values Detected',
            'description': f'{null_pct:.1f}% of data contains missing values',
            'severity': 'warning',
            'metric': f'{null_pct:.1f}% missing',
            'action': 'Consider handling missing values through imputation or removal'
        })
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        outlier_cols = []
        for col in num_cols[:10]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].shape[0]
            if outliers > 0:
                outlier_pct = (outliers / total_rows) * 100
                if outlier_pct > 5:
                    outlier_cols.append(col)
        
        if outlier_cols:
            insights.append({
                'title': 'Outliers Present',
                'description': f'Columns with significant outliers: {", ".join(outlier_cols[:3])}',
                'severity': 'warning',
                'metric': f'{len(outlier_cols)} columns',
                'action': 'Consider outlier removal or robust scaling for better model performance'
            })
        
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr:
                insights.append({
                    'title': 'High Correlations Found',
                    'description': f'Found {len(high_corr)} highly correlated feature pairs',
                    'severity': 'info',
                    'metric': f'{len(high_corr)} pairs > 0.8',
                    'action': 'Consider feature selection or PCA to reduce multicollinearity'
                })
        
        skewed_cols = []
        for col in num_cols:
            if df[col].std() > 0:
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    skewed_cols.append(col)
        
        if skewed_cols:
            insights.append({
                'title': 'Skewed Distributions',
                'description': f'{len(skewed_cols)} features show significant skewness',
                'severity': 'info',
                'metric': f'{len(skewed_cols)} skewed',
                'action': 'Apply log transformation or Box-Cox for normalization'
            })
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        high_card_cols = []
        for col in cat_cols:
            unique_ratio = df[col].nunique() / total_rows
            if unique_ratio > 0.5 and df[col].nunique() > 20:
                high_card_cols.append(col)
        
        if high_card_cols:
            insights.append({
                'title': 'High Cardinality Features',
                'description': f'Columns with many unique values: {", ".join(high_card_cols[:3])}',
                'severity': 'warning',
                'metric': f'{len(high_card_cols)} columns',
                'action': 'Consider encoding strategies or feature grouping for categorical variables'
            })
    
    if len(insights) < 5:
        potential_targets = [col for col in cat_cols if df[col].nunique() <= 10]
        if potential_targets:
            target = potential_targets[0]
            class_counts = df[target].value_counts()
            if len(class_counts) > 1:
                imbalance_ratio = class_counts.max() / class_counts.min()
                if imbalance_ratio > 3:
                    insights.append({
                        'title': 'Class Imbalance Detected',
                        'description': f'Target column "{target}" shows significant class imbalance',
                        'severity': 'critical',
                        'metric': f'Ratio: {imbalance_ratio:.1f}:1',
                        'action': 'Use stratified sampling, class weights, or SMOTE for better training'
                    })
    
    return insights[:5]

def generate_auto_eda_report(df):
    """Generate comprehensive EDA report like pandas profiling"""
    report = {
        'dataset_overview': {},
        'variable_summaries': [],
        'correlation_analysis': {},
        'missing_values': {},
        'outlier_analysis': {},
        'statistical_tests': {},
        'data_quality': {}
    }
    
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    duplicate_rows = df.duplicated().sum()
    
    report['dataset_overview'] = {
        'rows': convert_to_serializable(total_rows),
        'columns': convert_to_serializable(total_cols),
        'missing_cells': convert_to_serializable(missing_cells),
        'missing_percentage': round(missing_pct, 2),
        'duplicate_rows': convert_to_serializable(duplicate_rows),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    for col in df.columns:
        col_data = df[col]
        summary = {
            'name': col,
            'type': str(col_data.dtype),
            'count': convert_to_serializable(col_data.count()),
            'missing': convert_to_serializable(col_data.isnull().sum()),
            'missing_percentage': round(col_data.isnull().mean() * 100, 2),
            'unique': convert_to_serializable(col_data.nunique()),
            'unique_percentage': round((col_data.nunique() / total_rows) * 100, 2) if total_rows > 0 else 0
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            desc = col_data.describe()
            summary.update({
                'mean': convert_to_serializable(round(desc['mean'], 4) if 'mean' in desc else None),
                'std': convert_to_serializable(round(desc['std'], 4) if 'std' in desc else None),
                'min': convert_to_serializable(round(desc['min'], 4) if 'min' in desc else None),
                'q1': convert_to_serializable(round(desc['25%'], 4) if '25%' in desc else None),
                'median': convert_to_serializable(round(desc['50%'], 4) if '50%' in desc else None),
                'q3': convert_to_serializable(round(desc['75%'], 4) if '75%' in desc else None),
                'max': convert_to_serializable(round(desc['max'], 4) if 'max' in desc else None),
                'skewness': convert_to_serializable(round(col_data.skew(), 3)),
                'kurtosis': convert_to_serializable(round(col_data.kurtosis(), 3)),
                'variance': convert_to_serializable(round(col_data.var(), 4)),
                'range': convert_to_serializable(round(desc['max'] - desc['min'], 4) if 'max' in desc and 'min' in desc else None)
            })
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
            summary['outliers_count'] = convert_to_serializable(len(outliers))
            summary['outliers_percentage'] = round((len(outliers) / total_rows) * 100, 2)
            
        else:
            value_counts = col_data.value_counts()
            summary['most_frequent'] = str(value_counts.index[0]) if len(value_counts) > 0 else None
            summary['most_frequent_count'] = convert_to_serializable(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            summary['most_frequent_percentage'] = round((value_counts.iloc[0] / total_rows) * 100, 2) if len(value_counts) > 0 and total_rows > 0 else 0
            
            if len(value_counts) > 1:
                summary['second_most_frequent'] = str(value_counts.index[1]) if len(value_counts) > 1 else None
                summary['second_most_frequent_count'] = convert_to_serializable(value_counts.iloc[1]) if len(value_counts) > 1 else 0
        
        report['variable_summaries'].append(summary)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': convert_to_serializable(round(corr_matrix.iloc[i, j], 3))
                })
        
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        report['correlation_analysis'] = {
            'has_correlations': True,
            'num_variables': len(num_cols),
            'top_correlations': correlations[:10],
            'highly_correlated': [c for c in correlations if abs(c['correlation']) > 0.7]
        }
    else:
        report['correlation_analysis'] = {'has_correlations': False}
    
    missing_by_column = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_by_column.append({
                'column': col,
                'missing_count': convert_to_serializable(missing_count),
                'missing_percentage': round((missing_count / total_rows) * 100, 2)
            })
    
    report['missing_values'] = {
        'has_missing': len(missing_by_column) > 0,
        'columns_with_missing': missing_by_column,
        'total_missing_cells': convert_to_serializable(missing_cells)
    }
    
    outlier_summary = []
    for col in num_cols[:20]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        
        if len(outliers) > 0:
            outlier_summary.append({
                'column': col,
                'outlier_count': convert_to_serializable(len(outliers)),
                'outlier_percentage': round((len(outliers) / total_rows) * 100, 2),
                'lower_bound': round(Q1 - 1.5 * IQR, 3),
                'upper_bound': round(Q3 + 1.5 * IQR, 3)
            })
    
    report['outlier_analysis'] = {
        'has_outliers': len(outlier_summary) > 0,
        'outliers_by_column': outlier_summary
    }
    
    statistical_tests = []
    for col in num_cols[:10]:
        if df[col].count() > 50:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))
                is_normal = shapiro_p > 0.05
                statistical_tests.append({
                    'test': 'Normality (Shapiro-Wilk)',
                    'column': col,
                    'statistic': round(shapiro_stat, 4),
                    'p_value': float(shapiro_p),
                    'is_normal': bool(is_normal),
                    'interpretation': 'Normal distribution' if is_normal else 'Non-normal distribution'
                })
            except:
                pass
    
    report['statistical_tests'] = statistical_tests[:5]
    
    quality_score = 100
    quality_issues = []
    
    if missing_pct > 10:
        quality_score -= 20
        quality_issues.append(f'High missing values ({missing_pct:.1f}%)')
    elif missing_pct > 5:
        quality_score -= 10
        quality_issues.append(f'Moderate missing values ({missing_pct:.1f}%)')
    
    if duplicate_rows > 0:
        quality_score -= min(15, (duplicate_rows / total_rows) * 100)
        quality_issues.append(f'{duplicate_rows} duplicate rows')
    
    constant_cols = [col for col in num_cols if df[col].std() == 0]
    if constant_cols:
        quality_score -= len(constant_cols) * 5
        quality_issues.append(f'{len(constant_cols)} constant columns')
    
    report['data_quality'] = {
        'quality_score': round(max(0, quality_score), 1),
        'quality_grade': 'A' if quality_score >= 90 else 'B' if quality_score >= 80 else 'C' if quality_score >= 70 else 'D' if quality_score >= 60 else 'F',
        'issues': quality_issues,
        'recommendations': []
    }
    
    if missing_pct > 0:
        report['data_quality']['recommendations'].append(f'Handle {missing_cells} missing values using imputation or deletion')
    if duplicate_rows > 0:
        report['data_quality']['recommendations'].append(f'Remove {duplicate_rows} duplicate rows')
    if constant_cols:
        report['data_quality']['recommendations'].append(f'Drop constant columns: {", ".join(constant_cols[:3])}')
    if len(num_cols) > 10:
        report['data_quality']['recommendations'].append('Consider dimensionality reduction for better performance')
    
    return report

def create_all_visualizations(df):
    """Create all possible visualizations for comprehensive analysis"""
    visualizations = {}
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Correlation Heatmap
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        fig_corr.update_layout(
            title=' Feature Correlation Matrix',
            height=600,
            width=800,
            xaxis_tickangle=-45,
            font=dict(size=12)
        )
        visualizations['correlation_heatmap'] = json.dumps(fig_corr, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. Distribution plots for all numeric columns
    if num_cols:
        n_cols = min(9, len(num_cols))
        fig_dist = go.Figure()
        for col in num_cols[:n_cols]:
            fig_dist.add_trace(go.Histogram(
                x=df[col].dropna(),
                name=col,
                opacity=0.7,
                nbinsx=30,
                histnorm='probability density'
            ))
        fig_dist.update_layout(
            title=' Feature Distributions (Density)',
            barmode='overlay',
            height=500,
            legend_title="Features",
            font=dict(size=12)
        )
        visualizations['distributions'] = json.dumps(fig_dist, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. Box plots for all numeric columns
    if num_cols:
        fig_box = go.Figure()
        for col in num_cols[:12]:
            fig_box.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                boxmean='sd',
                boxpoints='outliers'
            ))
        fig_box.update_layout(
            title=' Box Plot Analysis (with Mean & Standard Deviation)',
            height=500,
            showlegend=True,
            font=dict(size=12)
        )
        visualizations['boxplot_comparison'] = json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 4. Violin plots for distribution shape
    if num_cols:
        fig_violin = go.Figure()
        for col in num_cols[:8]:
            fig_violin.add_trace(go.Violin(
                y=df[col].dropna(),
                name=col,
                box_visible=True,
                meanline_visible=True,
                opacity=0.6
            ))
        fig_violin.update_layout(
            title='🎻 Violin Plots (Distribution Shape)',
            height=500,
            font=dict(size=12)
        )
        visualizations['violin_plots'] = json.dumps(fig_violin, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 5. Missing values heatmap
    missing_matrix = df.isnull().astype(int)
    if missing_matrix.sum().sum() > 0:
        fig_missing = go.Figure(data=go.Heatmap(
            z=missing_matrix.T.values,
            x=missing_matrix.index,
            y=missing_matrix.columns,
            colorscale=[[0, 'rgb(0,100,80)'], [1, 'rgb(255,100,100)']],
            showscale=True,
            zmin=0,
            zmax=1
        ))
        fig_missing.update_layout(
            title=' Missing Values Pattern',
            height=400 + len(missing_matrix.columns) * 20,
            width=900,
            font=dict(size=10)
        )
        visualizations['missing_pattern'] = json.dumps(fig_missing, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 6. PCA 2D Projection
    if len(num_cols) >= 3 and len(df) >= 10:
        try:
            pca_data = df[num_cols].dropna()
            if len(pca_data) >= 10:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(pca_data)
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_data)
                
                fig_pca = go.Figure(data=go.Scatter(
                    x=pca_result[:, 0],
                    y=pca_result[:, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=np.arange(len(pca_result)),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sample Index")
                    ),
                    text=pca_data.index,
                    hoverinfo='text+x+y'
                ))
                fig_pca.update_layout(
                    title=f' PCA Projection ({pca.explained_variance_ratio_[0]:.1%} / {pca.explained_variance_ratio_[1]:.1%} variance)',
                    xaxis_title='First Principal Component',
                    yaxis_title='Second Principal Component',
                    height=500,
                    font=dict(size=12)
                )
                visualizations['pca_projection'] = json.dumps(fig_pca, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass
    
    # 7. t-SNE visualization
    if len(num_cols) >= 3 and len(df) >= 30 and len(df) <= 1000:
        try:
            tsne_data = df[num_cols].dropna()
            if len(tsne_data) >= 30:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(tsne_data)
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_data)-1))
                tsne_result = tsne.fit_transform(scaled_data)
                
                fig_tsne = go.Figure(data=go.Scatter(
                    x=tsne_result[:, 0],
                    y=tsne_result[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=np.arange(len(tsne_result)),
                        colorscale='Plasma',
                        showscale=True
                    ),
                    text=tsne_data.index,
                    hoverinfo='text+x+y'
                ))
                fig_tsne.update_layout(
                    title=' t-SNE Visualization (2D Projection)',
                    xaxis_title='t-SNE Component 1',
                    yaxis_title='t-SNE Component 2',
                    height=500,
                    font=dict(size=12)
                )
                visualizations['tsne_projection'] = json.dumps(fig_tsne, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass
    
    # 8. Time series analysis
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols and len(num_cols) > 0:
        try:
            time_col = datetime_cols[0]
            for value_col in num_cols[:3]:
                df_time = df[[time_col, value_col]].dropna()
                df_time = df_time.sort_values(time_col)
                
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(
                    x=df_time[time_col],
                    y=df_time[value_col],
                    mode='lines+markers',
                    name=value_col,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
                fig_time.update_layout(
                    title=f' Time Series: {value_col} over Time',
                    xaxis_title='Time',
                    yaxis_title=value_col,
                    height=400,
                    font=dict(size=12)
                )
                visualizations[f'time_series_{value_col}'] = json.dumps(fig_time, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass
    
    # 9. Feature importance (if target detected)
    potential_targets = cat_cols if cat_cols else num_cols[:1] if num_cols else []
    if potential_targets and len(num_cols) > 1:
        try:
            target_col = potential_targets[0]
            feature_cols = [c for c in num_cols if c != target_col]
            
            if len(feature_cols) > 0:
                X = df[feature_cols].dropna()
                y = df[target_col].loc[X.index]
                
                if len(y.unique()) <= 10:
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X, y)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X, y)
                
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig_importance = go.Figure(data=go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    marker_color='rgb(55, 83, 109)',
                    text=importance_df['importance'].round(3),
                    textposition='outside'
                ))
                fig_importance.update_layout(
                    title=f'⭐ Feature Importance (Target: {target_col})',
                    xaxis_title='Importance Score',
                    yaxis_title='Features',
                    height=500,
                    font=dict(size=12)
                )
                visualizations['feature_importance'] = json.dumps(fig_importance, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass
    
    # 10. Pair plot matrix
    if len(num_cols) >= 2 and len(num_cols) <= 6:
        try:
            selected = num_cols[:min(5, len(num_cols))]
            pair_data = df[selected].dropna()
            
            fig = ff.create_scatterplotmatrix(
                pair_data,
                diag='histogram',
                index=selected[0] if len(selected) > 0 else None,
                size=5,
                title='📊 Pair Plot Matrix'
            )
            visualizations['pair_plot'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass
    
    # 11. 3D Scatter plot
    if len(num_cols) >= 3:
        try:
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=df[num_cols[0]].dropna(),
                y=df[num_cols[1]].dropna(),
                z=df[num_cols[2]].dropna(),
                mode='markers',
                marker=dict(
                    size=5,
                    color=df[num_cols[2]].dropna(),
                    colorscale='Viridis',
                    showscale=True
                )
            )])
            fig_3d.update_layout(
                title=f'🎨 3D Scatter Plot: {num_cols[0]} vs {num_cols[1]} vs {num_cols[2]}',
                scene=dict(
                    xaxis_title=num_cols[0],
                    yaxis_title=num_cols[1],
                    zaxis_title=num_cols[2]
                ),
                height=600,
                font=dict(size=12)
            )
            visualizations['scatter_3d'] = json.dumps(fig_3d, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass
    
    # 12. Categorical bar charts
    if cat_cols:
        for col in cat_cols[:4]:
            try:
                value_counts = df[col].value_counts().head(20)
                fig_bar = go.Figure(data=go.Bar(
                    x=value_counts.index.astype(str),
                    y=value_counts.values,
                    marker_color='lightsalmon',
                    text=value_counts.values,
                    textposition='auto'
                ))
                fig_bar.update_layout(
                    title=f'📊 Top 20 Values in {col}',
                    xaxis_title=col,
                    yaxis_title='Count',
                    height=400,
                    xaxis_tickangle=-45,
                    font=dict(size=11)
                )
                visualizations[f'categorical_{col}'] = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)
            except:
                pass
    
    # 13. Pie charts for categorical columns with few categories
    if cat_cols:
        for col in cat_cols[:3]:
            if df[col].nunique() <= 10:
                try:
                    value_counts = df[col].value_counts()
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=value_counts.index.astype(str),
                        values=value_counts.values,
                        hole=0.3,
                        textinfo='label+percent',
                        textposition='auto'
                    )])
                    fig_pie.update_layout(
                        title=f'🥧 Distribution of {col}',
                        height=450,
                        font=dict(size=12)
                    )
                    visualizations[f'pie_{col}'] = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
                except:
                    pass
    
    # 14. Scatter plots for numeric relationships
    if len(num_cols) >= 2:
        for i in range(min(3, len(num_cols))):
            for j in range(i+1, min(4, len(num_cols))):
                try:
                    fig_scatter = px.scatter(
                        df, x=num_cols[i], y=num_cols[j],
                        title=f'📉 {num_cols[i]} vs {num_cols[j]}',
                        trendline="ols",
                        opacity=0.6
                    )
                    fig_scatter.update_layout(height=450, font=dict(size=12))
                    visualizations[f'scatter_{num_cols[i]}_{num_cols[j]}'] = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)
                except:
                    pass
    
    # 15. Area charts for trends
    if datetime_cols and len(num_cols) > 0:
        try:
            time_col = datetime_cols[0]
            for value_col in num_cols[:2]:
                df_time = df[[time_col, value_col]].dropna().sort_values(time_col)
                fig_area = go.Figure(data=go.Scatter(
                    x=df_time[time_col],
                    y=df_time[value_col],
                    fill='tozeroy',
                    mode='lines',
                    name=value_col
                ))
                fig_area.update_layout(
                    title=f'📈 Area Chart: {value_col} over Time',
                    xaxis_title='Time',
                    yaxis_title=value_col,
                    height=400,
                    font=dict(size=12)
                )
                visualizations[f'area_{value_col}'] = json.dumps(fig_area, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass
    
    # 16. Heatmap of missing values by column
    if df.isnull().sum().sum() > 0:
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
        }).sort_values('Missing_Percentage', ascending=False)
        
        fig_missing_bar = go.Figure(data=go.Bar(
            x=missing_df['Column'],
            y=missing_df['Missing_Percentage'],
            marker_color='coral',
            text=missing_df['Missing_Percentage'].round(1),
            textposition='outside'
        ))
        fig_missing_bar.update_layout(
            title='📊 Missing Values by Column',
            xaxis_title='Columns',
            yaxis_title='Missing Percentage (%)',
            height=500,
            xaxis_tickangle=-45,
            font=dict(size=12)
        )
        visualizations['missing_bar'] = json.dumps(fig_missing_bar, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 17. QQ plots for normality check
    if num_cols:
        for col in num_cols[:4]:
            try:
                data = df[col].dropna()
                if len(data) > 10:
                    from scipy import stats
                    qq_data = stats.probplot(data, dist="norm")
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(
                        x=qq_data[0][0],
                        y=qq_data[0][1],
                        mode='markers',
                        name='Sample',
                        marker=dict(size=6, color='blue')
                    ))
                    fig_qq.add_trace(go.Scatter(
                        x=qq_data[0][0],
                        y=qq_data[1][1] * np.array(qq_data[0][0]) + qq_data[1][0],
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red', width=2)
                    ))
                    fig_qq.update_layout(
                        title=f'📐 Q-Q Plot for {col}',
                        xaxis_title='Theoretical Quantiles',
                        yaxis_title='Sample Quantiles',
                        height=400,
                        font=dict(size=12)
                    )
                    visualizations[f'qq_{col}'] = json.dumps(fig_qq, cls=plotly.utils.PlotlyJSONEncoder)
            except:
                pass
    
    # 18. Dendrogram for hierarchical clustering
    if len(num_cols) >= 2 and len(df) >= 10 and len(df) <= 200:
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            cluster_data = df[num_cols[:5]].dropna()
            if len(cluster_data) >= 10:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                linkage_matrix = linkage(scaled_data, method='ward')
                
                fig_dendro = ff.create_dendrogram(scaled_data, orientation='left', labels=cluster_data.index)
                fig_dendro.update_layout(
                    title='🌲 Hierarchical Clustering Dendrogram',
                    height=600,
                    width=800,
                    font=dict(size=12)
                )
                visualizations['dendrogram'] = json.dumps(fig_dendro, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass
    
    # 19. Parallel coordinates for multidimensional data
    if len(num_cols) >= 3 and len(num_cols) <= 10:
        try:
            fig_parallel = px.parallel_coordinates(
                df[num_cols].dropna(),
                dimensions=num_cols,
                color_continuous_scale=px.colors.diverging.Thermal,
                title='📐 Parallel Coordinates Plot'
            )
            fig_parallel.update_layout(height=600, font=dict(size=12))
            visualizations['parallel_coordinates'] = json.dumps(fig_parallel, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass
    
    # 20. Radar chart for feature comparison
    if len(num_cols) >= 3:
        try:
            normalized_data = df[num_cols[:8]].mean().reset_index()
            normalized_data.columns = ['feature', 'value']
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=normalized_data['value'],
                theta=normalized_data['feature'],
                fill='toself',
                name='Average Values'
            ))
            fig_radar.update_layout(
                title='📡 Radar Chart - Feature Averages',
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, normalized_data['value'].max()])
                ),
                height=500,
                font=dict(size=12)
            )
            visualizations['radar_chart'] = json.dumps(fig_radar, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass
    
    return visualizations

def create_business_dashboard(df):
    """Create business-focused dashboard components"""
    dashboard = {}
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Key Performance Indicators
    kpis = []
    if num_cols:
        for col in num_cols[:3]:
            kpis.append({
                'title': f'Avg {col.replace("_", " ").title()}',
                'value': f'{df[col].mean():,.2f}',
                'trend': 'up' if df[col].mean() > df[col].median() else 'down',
                'change': f'{((df[col].mean() - df[col].median()) / df[col].median() * 100):.1f}%'
            })
    
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    kpis.append({
        'title': 'Data Completeness',
        'value': f'{completeness:.1f}%',
        'trend': 'up' if completeness > 90 else 'down',
        'change': f'{100 - completeness:.1f}% missing'
    })
    
    dashboard['kpis'] = kpis
    
    # Business Recommendations
    recommendations = []
    
    price_cols = [col for col in num_cols if any(word in col.lower() for word in ['price', 'revenue', 'sales', 'cost', 'amount'])]
    if price_cols:
        recommendations.append({
            'title': 'Revenue Optimization',
            'description': f'Analyze {price_cols[0]} distribution to identify pricing opportunities',
            'priority': 'High',
            'impact': 'Potential 15-25% revenue increase'
        })
    
    segment_cols = [col for col in cat_cols if any(word in col.lower() for word in ['segment', 'category', 'type', 'region'])]
    if segment_cols:
        recommendations.append({
            'title': 'Customer Segmentation',
            'description': f'Use {segment_cols[0]} to identify high-value customer segments',
            'priority': 'Medium',
            'impact': 'Improved targeting and retention'
        })
    
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        recommendations.append({
            'title': 'Trend Analysis',
            'description': 'Leverage time-based patterns for forecasting and planning',
            'priority': 'High',
            'impact': 'Better demand prediction and resource allocation'
        })
    
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr:
            recommendations.append({
                'title': 'Feature Optimization',
                'description': f'High correlation detected between {high_corr[0][0]} and {high_corr[0][1]} (r={high_corr[0][2]:.2f})',
                'priority': 'Medium',
                'impact': 'Simplify model and reduce overfitting'
            })
    
    dashboard['recommendations'] = recommendations[:5]
    
    # Risk Assessment
    risks = []
    
    high_missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0.3]
    if high_missing_cols:
        risks.append({
            'title': 'Data Quality Risk',
            'description': f'{len(high_missing_cols)} columns have >30% missing values',
            'severity': 'High',
            'mitigation': 'Implement data imputation or collection improvements'
        })
    
    for col in num_cols[:5]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_pct = (df[(df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)].shape[0] / len(df)) * 100
        if outlier_pct > 5:
            risks.append({
                'title': f'Extreme Values in {col}',
                'description': f'{outlier_pct:.1f}% of records are extreme outliers',
                'severity': 'Medium',
                'mitigation': 'Consider robust statistical methods or outlier capping'
            })
            break
    
    potential_targets = [col for col in cat_cols if df[col].nunique() <= 10 and df[col].nunique() > 1]
    if potential_targets:
        target = potential_targets[0]
        class_counts = df[target].value_counts()
        if len(class_counts) > 1:
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 5:
                risks.append({
                    'title': 'Class Imbalance Risk',
                    'description': f'Target variable has {imbalance_ratio:.1f}:1 imbalance ratio',
                    'severity': 'High',
                    'mitigation': 'Use SMOTE, class weights, or ensemble methods'
                })
    
    dashboard['risks'] = risks[:3]
    
    # Predictive Insights
    predictive = []
    
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols and num_cols:
        try:
            time_col = datetime_cols[0]
            value_col = num_cols[0]
            df_time = df[[time_col, value_col]].dropna().sort_values(time_col)
            
            if len(df_time) > 10:
                from scipy import stats
                x = np.arange(len(df_time))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, df_time[value_col].values)
                
                trend = "upward" if slope > 0 else "downward"
                predictive.append({
                    'title': 'Trend Prediction',
                    'description': f'{value_col} shows {trend} trend (r²={r_value**2:.2f})',
                    'confidence': f'{abs(r_value)*100:.0f}%',
                    'recommendation': f'Expected {"increase" if slope > 0 else "decrease"} in future periods'
                })
        except:
            pass
    
    if len(num_cols) >= 2 and len(df) >= 50:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            cluster_data = df[num_cols[:5]].dropna()
            if len(cluster_data) >= 50:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                inertias = []
                for k in range(2, min(8, len(cluster_data) // 10)):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(scaled_data)
                    inertias.append(kmeans.inertia_)
                
                if inertias:
                    optimal_k = np.argmin(np.diff(inertias)) + 2
                    predictive.append({
                        'title': 'Segmentation Opportunity',
                        'description': f'Data suggests {optimal_k} natural segments/clusters',
                        'confidence': 'Medium',
                        'recommendation': 'Use clustering for customer segmentation and targeted strategies'
                    })
        except:
            pass
    
    dashboard['predictive_insights'] = predictive[:3]
    
    return dashboard

def run_memory_safe_training(df, target_col, problem_type, train_size, fold,
                             normalize, remove_out, max_models=None):
    """Execute training with memory safety"""
    warnings_list = []
    t0 = time.time()

    original_rows = len(df)
    if original_rows > MAX_ROWS_TRAINING:
        df_train = smart_sample(df, target_col, MAX_ROWS_TRAINING)
        warnings_list.append(
            f"Dataset {original_rows:,} rows — auto-sampled to {MAX_ROWS_TRAINING:,} rows."
        )
    elif original_rows > MAX_ROWS_WARNING:
        df_train = df.copy()
        warnings_list.append(
            f"Dataset {original_rows:,} rows — training will proceed but may be slow."
        )
    else:
        df_train = df.copy()

    include_models = ALL_CLF_MODELS if problem_type == "classification" else ALL_REG_MODELS

    if max_models and max_models < len(include_models):
        include_models = include_models[:max_models]

    mem_before = get_memory_usage_mb()
    if mem_before > 400:
        force_gc()

    setup_kwargs = dict(
        data=df_train, target=target_col,
        train_size=float(train_size), fold=int(fold),
        normalize=normalize, verbose=False, html=False,
        session_id=42, n_jobs=1, use_gpu=False,
    )
    if remove_out and problem_type == "regression" and len(df_train) > 100:
        setup_kwargs["remove_outliers"] = True

    try:
        if problem_type == "classification":
            clf_setup(**setup_kwargs)
            pull_fn, save_fn, cmp_fn = clf_pull, clf_save, clf_compare
        else:
            reg_setup(**setup_kwargs)
            pull_fn, save_fn, cmp_fn = reg_pull, reg_save, reg_compare
    except Exception as e:
        err = str(e).lower()
        if "memory" in err or "killed" in err:
            raise MemoryError(f"Setup failed due to memory — reduce dataset size.")
        raise

    force_gc()

    try:
        best = cmp_fn(verbose=False, n_select=1, include=include_models, errors="ignore")
        results = pull_fn()
    except MemoryError:
        force_gc()
        light = ["lr", "dt", "ridge"]
        warnings_list.append("Memory issue — trying with only 3 lightest models.")
        best = cmp_fn(verbose=False, n_select=1, include=light)
        results = pull_fn()
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ["memory", "killed", "oom", "cannot allocate"]):
            raise MemoryError("Model comparison failed due to memory.")
        raise

    # Save model with unique ID
    model_id = str(uuid.uuid4())[:8]
    model_path = os.path.join(app.config['MODEL_FOLDER'], f"best_model_{model_id}")
    try:
        save_fn(best, model_path)
    except Exception:
        pass

    force_gc()
    elapsed = time.time() - t0
    return best, results, elapsed, warnings_list, len(df_train), model_id

def generate_plotly_json(fig):
    """Convert plotly figure to JSON for rendering"""
    if fig is None:
        return None
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Decorators
def session_required(f):
    """Ensure session exists"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            session['data'] = None
            session['results'] = None
            session['training_history'] = []
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
@session_required
def index():
    """Main page"""
    return render_template('index.html', data_exists=session.get('data') is not None)

@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint to check if server is running"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
@session_required
def upload_file():
    """Handle file upload with comprehensive error handling"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        allowed_extensions = ('.csv', '.xlsx', '.xls')
        if not file.filename.lower().endswith(allowed_extensions):
            return jsonify({'error': 'Unsupported file format. Please upload CSV or Excel files.'}), 400

        try:
            if file.filename.lower().endswith('.csv'):
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, encoding='latin1')
                    except UnicodeDecodeError:
                        file.seek(0)
                        df = pd.read_csv(file, encoding='cp1252')
            else:
                df = pd.read_excel(file)
                
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'The uploaded file is empty'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400

        if df.empty:
            return jsonify({'error': 'The uploaded file contains no data'}), 400

        # Convert to JSON serializable
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype(str)
        
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        session['data'] = df.to_json(orient='split', date_format='iso')
        session['dataset_name'] = file.filename
        session['results'] = None
        session['best_model'] = None
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'rows': len(df),
            'columns': len(df.columns),
            'message': f'Successfully uploaded {file.filename} ({len(df)} rows, {len(df.columns)} columns)'
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/load-sample', methods=['POST'])
@session_required
def load_sample():
    """Load sample dataset"""
    data = request.get_json()
    sample_name = data.get('sample')
    
    urls = {
        "titanic": ("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", "Survived"),
        "diamonds": ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv", "price"),
        "iris": ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv", "species"),
        "tips": ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv", "tip"),
        "mpg": ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv", "mpg")
    }
    
    if sample_name not in urls:
        return jsonify({'error': 'Invalid sample'}), 400
    
    try:
        url, hint = urls[sample_name]
        df = pd.read_csv(url)
        
        session['data'] = df.to_json(orient='split')
        session['dataset_name'] = sample_name.capitalize()
        session['sample_hint'] = hint
        session['results'] = None
        session['best_model'] = None
        
        return jsonify({
            'success': True,
            'sample': sample_name,
            'rows': len(df),
            'columns': len(df.columns),
            'target_hint': hint
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights')
@session_required
def get_insights():
    """Get smart insights from dataset"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    df = pd.read_json(session['data'], orient='split')
    insights = generate_smart_insights(df)
    return jsonify({'insights': insights})

@app.route('/api/auto-eda')
@session_required
def auto_eda():
    """Get comprehensive auto EDA report"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    df = pd.read_json(session['data'], orient='split')
    report = generate_auto_eda_report(df)
    return jsonify(convert_to_serializable(report))

@app.route('/api/all-visualizations')
@session_required
def all_visualizations():
    """Get all possible visualizations"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    df = pd.read_json(session['data'], orient='split')
    visualizations = create_all_visualizations(df)
    return jsonify(visualizations)

@app.route('/api/business-dashboard')
@session_required
def business_dashboard():
    """Get business dashboard data"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    df = pd.read_json(session['data'], orient='split')
    dashboard = create_business_dashboard(df)
    return jsonify(convert_to_serializable(dashboard))

@app.route('/api/data-info')
@session_required
def data_info():
    """Get dataset information"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    df = pd.read_json(session['data'], orient='split')
    
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 2) if df.size > 0 else 0
    dup_cnt = int(df.duplicated().sum())
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    hs = round(max(0, 100 - null_pct * 2 - (dup_cnt / len(df) * 300 if len(df) > 0 else 0)), 1)
    
    return jsonify({
        'rows': len(df),
        'columns': len(df.columns),
        'num_cols': len(num_cols),
        'cat_cols': len(cat_cols),
        'null_pct': null_pct,
        'null_count': int(df.isnull().sum().sum()),
        'duplicates': dup_cnt,
        'health_score': hs,
        'column_names': df.columns.tolist(),
        'num_columns': num_cols,
        'cat_columns': cat_cols,
        'dtypes': df.dtypes.astype(str).to_dict()
    })

@app.route('/api/data-preview')
@session_required
def data_preview():
    """Get data preview"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    df = pd.read_json(session['data'], orient='split')
    rows = int(request.args.get('rows', 20))
    columns = request.args.get('columns', '')
    
    if columns:
        cols = [c.strip() for c in columns.split(',') if c.strip() in df.columns]
        preview_df = df[cols].head(rows) if cols else df.head(rows)
    else:
        preview_df = df.head(rows)
    
    preview_data = preview_df.replace({np.nan: None}).to_dict(orient='records')
    
    return jsonify({
        'columns': preview_df.columns.tolist(),
        'data': preview_data,
        'total_rows': len(df)
    })

@app.route('/api/detect-target', methods=['POST'])
@session_required
def detect_target():
    """Detect problem type for target column"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    data = request.get_json()
    target = data.get('target')
    
    df = pd.read_json(session['data'], orient='split')
    
    if target not in df.columns:
        return jsonify({'error': 'Target column not found'}), 404
    
    ptype = detect_problem_type(df[target])
    uniq = int(df[target].nunique())
    
    return jsonify({
        'problem_type': ptype,
        'unique_values': uniq,
        'type_label': 'Classification' if ptype == 'classification' else 'Regression'
    })

@app.route('/api/train', methods=['POST'])
@session_required
def train_model():
    """Train model endpoint"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    data = request.get_json()
    df = pd.read_json(session['data'], orient='split')
    
    target = data.get('target')
    train_size = float(data.get('train_size', 0.8))
    fold = int(data.get('fold', 5))
    normalize = data.get('normalize', True)
    remove_out = data.get('remove_outliers', False)
    max_models = int(data.get('max_models', 999))
    
    if target not in df.columns:
        return jsonify({'error': 'Target column not found'}), 404
    
    ptype = detect_problem_type(df[target])
    
    try:
        best, results, elapsed, warnings_list, trained_rows, model_id = run_memory_safe_training(
            df=df,
            target_col=target,
            problem_type=ptype,
            train_size=train_size,
            fold=fold,
            normalize=normalize,
            remove_out=remove_out,
            max_models=max_models
        )
        
        session['results'] = results.to_json()
        session['training_time'] = elapsed
        session['last_model_id'] = model_id
        session['folds_used'] = fold
        
        model_col = "Model" if "Model" in results.columns else results.columns[0]
        num_res = results.select_dtypes(include=[np.number]).columns
        best_name = str(results.iloc[0][model_col])
        best_score = float(results.iloc[0][num_res[0]]) if len(num_res) else 0.0
        
        history_entry = {
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset': session.get('dataset_name', 'Uploaded CSV'),
            'problem_type': ptype,
            'best_model': best_name,
            'score': round(best_score, 4),
            'rows': trained_rows,
            'cols': len(df.columns),
            'model_id': model_id
        }
        
        if 'training_history' not in session:
            session['training_history'] = []
        session['training_history'].append(history_entry)
        
        results_dict = results.head(10).to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'best_model': best_name,
            'best_score': best_score,
            'metric_name': num_res[0] if len(num_res) else 'Score',
            'elapsed': fmt_time(elapsed),
            'elapsed_seconds': elapsed,
            'trained_rows': trained_rows,
            'warnings': warnings_list,
            'results': results_dict,
            'results_columns': results.columns.tolist(),
            'model_id': model_id
        })
        
    except MemoryError as e:
        return jsonify({'error': 'memory_error', 'message': str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'training_failed', 'message': str(e)}), 500

@app.route('/api/results')
@session_required
def get_results():
    """Get training results"""
    if not session.get('results'):
        return jsonify({'error': 'No results available'}), 404
    
    results = pd.read_json(session['results'])
    num_res = results.select_dtypes(include=[np.number]).columns.tolist()
    
    top_models = results.head(6).to_dict(orient='records')
    
    if len(num_res) > 0:
        best_metrics = results.iloc[0][num_res[:6]].to_dict()
    else:
        best_metrics = {}
    
    return jsonify({
        'exists': True,
        'columns': results.columns.tolist(),
        'num_columns': num_res,
        'top_models': top_models,
        'best_metrics': best_metrics,
        'training_time': session.get('training_time'),
        'model_id': session.get('last_model_id'),
        'folds_used': session.get('folds_used', 5)
    })

@app.route('/api/download-model/<model_id>')
@session_required
def download_model(model_id):
    """Download trained model"""
    model_path = os.path.join(app.config['MODEL_FOLDER'], f"best_model_{model_id}.pkl")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(app.config['MODEL_FOLDER'], f"best_model_{model_id}")
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    
    return send_file(
        model_path,
        as_attachment=True,
        download_name=f"model_{model_id}.pkl",
        mimetype='application/octet-stream'
    )

@app.route('/api/download-results')
@session_required
def download_results():
    """Download results as CSV"""
    if not session.get('results'):
        return jsonify({'error': 'No results available'}), 404
    
    results = pd.read_json(session['results'])
    
    output = io.StringIO()
    results.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        as_attachment=True,
        download_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mimetype='text/csv'
    )

@app.route('/api/history')
@session_required
def get_history():
    """Get training history"""
    history = session.get('training_history', [])
    return jsonify({'history': history})

@app.route('/api/download-history')
@session_required
def download_history():
    """Download history as CSV"""
    history = session.get('training_history', [])
    
    if not history:
        return jsonify({'error': 'No history available'}), 404
    
    df = pd.DataFrame(history)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        as_attachment=True,
        download_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mimetype='text/csv'
    )

@app.route('/api/clear-session', methods=['POST'])
@session_required
def clear_session():
    """Clear current session data"""
    session['data'] = None
    session['results'] = None
    session['best_model'] = None
    session['training_history'] = []
    return jsonify({'success': True})

# Error Handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Run Application
port = int(os.environ.get("PORT", 10000))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=True)