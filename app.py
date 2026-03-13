

import os
import io
import gc
import time
import uuid
import warnings
from datetime import datetime
from functools import wraps

import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
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

warnings.filterwarnings("ignore")


#  Flask App Configuration

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


#  Memory-Safe Training Config

MAX_ROWS_TRAINING = 5_000
MAX_ROWS_WARNING = 2_000
SAMPLE_RANDOM_STATE = 42

ALL_CLF_MODELS = ["lr", "dt", "rf", "et", "ridge", "knn", "nb", "ada", 
                  "xgboost", "lightgbm", "catboost", "gbc", "lda"]
ALL_REG_MODELS = ["lr", "dt", "rf", "et", "ridge", "lasso", "knn", "ada", 
                  "en", "xgboost", "lightgbm", "catboost", "gbr", "br"]


#  Utility Functions

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

def create_correlation_heatmap(df, max_cols=15):
    """Create a properly formatted correlation heatmap"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(num_cols) < 2:
        return None
    
    # Limit to max_cols for readability
    if len(num_cols) > max_cols:
        num_cols = num_cols[:max_cols]
    
    # Calculate correlation matrix
    corr_matrix = df[num_cols].corr().round(2)
    
    # Create heatmap with proper styling
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(
            title="Correlation",
            titleside="right"
        )
    ))
    
    # Update layout for better appearance
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500,
        xaxis=dict(
            tickangle=-45,
            side="bottom"
        ),
        yaxis=dict(
            tickangle=0
        ),
        margin=dict(l=100, r=50, t=80, b=100)
    )
    
    return fig

def create_distribution_plot(df, column, chart_type='histogram'):
    """Create distribution plot with proper styling"""
    if column not in df.columns:
        return None
    
    col_data = df[column]
    is_numeric = pd.api.types.is_numeric_dtype(col_data)
    
    if is_numeric:
        if chart_type == 'histogram':
            fig = px.histogram(
                df, x=column, nbins=40,
                title=f"Distribution of {column}"
            )
        elif chart_type == 'box':
            fig = px.box(
                df, y=column,
                title=f"Box Plot of {column}"
            )
        elif chart_type == 'violin':
            fig = px.violin(
                df, y=column, box=True,
                title=f"Violin Plot of {column}"
            )
        else:
            fig = px.histogram(df, x=column, nbins=40)
    else:
        # Categorical data - bar chart of top 15 values
        value_counts = col_data.value_counts().head(15)
        fig = px.bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            title=f"Top 15 Values in {column}",
            labels={'x': column, 'y': 'Count'}
        )
    
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def get_column_stats(df, column):
    """Get statistics for a column"""
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

def make_json_serializable(obj):
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
    return obj


#  Decorators

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


#  Routes

@app.route('/')
@session_required
def index():
    """Main page"""
    return render_template('index.html', 
                         data_exists=session.get('data') is not None)

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
        # Log request details for debugging
        app.logger.info(f"Upload request received. Files: {request.files}")
        
        # Check if file exists in request
        if 'file' not in request.files:
            app.logger.error("No file part in request")
            return jsonify({'error': 'No file part in the request. Please select a file.'}), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            app.logger.error("Empty filename")
            return jsonify({'error': 'No file selected. Please choose a file.'}), 400

        # Log file details
        app.logger.info(f"File received: {file.filename}")
        
        # Check file extension
        allowed_extensions = ('.csv', '.xlsx', '.xls')
        if not file.filename.lower().endswith(allowed_extensions):
            return jsonify({'error': 'Unsupported file format. Please upload CSV or Excel files.'}), 400

        # Read the file with proper encoding handling
        try:
            if file.filename.lower().endswith('.csv'):
                # Try different encodings for CSV
                try:
                    # Try UTF-8 first
                    file.seek(0)
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        # Try Latin-1
                        file.seek(0)
                        df = pd.read_csv(file, encoding='latin1')
                    except UnicodeDecodeError:
                        # Try CP1252
                        file.seek(0)
                        df = pd.read_csv(file, encoding='cp1252')
            else:
                # Excel files
                df = pd.read_excel(file)
                
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'The uploaded file is empty'}), 400
        except Exception as e:
            app.logger.error(f"Error reading file: {str(e)}")
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400

        # Basic data validation
        if df.empty:
            return jsonify({'error': 'The uploaded file contains no data'}), 400

        # Convert data types to JSON serializable
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype(str)
        
        # Handle datetime columns
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Store in session
        session['data'] = df.to_json(orient='split', date_format='iso')
        session['dataset_name'] = file.filename
        session['results'] = None
        session['best_model'] = None
        
        app.logger.info(f"File uploaded successfully: {file.filename} ({len(df)} rows)")
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'rows': len(df),
            'columns': len(df.columns),
            'message': f'Successfully uploaded {file.filename} ({len(df)} rows, {len(df.columns)} columns)'
        })
        
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}", exc_info=True)
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
    
    # Calculate health score
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
    
    # Convert to JSON-friendly format
    preview_data = preview_df.replace({np.nan: None}).to_dict(orient='records')
    
    return jsonify({
        'columns': preview_df.columns.tolist(),
        'data': preview_data,
        'total_rows': len(df)
    })

@app.route('/api/data-stats')
@session_required
def data_stats():
    """Get statistical summary"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    df = pd.read_json(session['data'], orient='split')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not num_cols:
        return jsonify({'stats': {}})
    
    stats = df[num_cols].describe(percentiles=[.25, .5, .75]).round(3).to_dict()
    return jsonify({'stats': stats})

@app.route('/api/column-details')
@session_required
def column_details():
    """Get detailed column information"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    df = pd.read_json(session['data'], orient='split')
    
    details = []
    for col in df.columns:
        details.append({
            'column': col,
            'type': str(df[col].dtype),
            'non_null': int(df[col].count()),
            'null_pct': round(df[col].isnull().mean() * 100, 1),
            'unique': int(df[col].nunique())
        })
    
    return jsonify({'details': details})

@app.route('/api/eda/distribution')
@session_required
def eda_distribution():
    """Get distribution plot data"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    df = pd.read_json(session['data'], orient='split')
    column = request.args.get('column')
    chart_type = request.args.get('chart_type', 'histogram')
    
    if column not in df.columns:
        return jsonify({'error': 'Column not found'}), 404
    
    fig = create_distribution_plot(df, column, chart_type)
    stats, is_numeric = get_column_stats(df, column)
    
    return jsonify({
        'plot': generate_plotly_json(fig),
        'stats': stats,
        'is_numeric': is_numeric
    })

@app.route('/api/eda/correlation')
@session_required
def eda_correlation():
    """Get correlation heatmap"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    df = pd.read_json(session['data'], orient='split')
    fig = create_correlation_heatmap(df)
    
    if fig is None:
        return jsonify({'error': 'Need at least 2 numeric columns'}), 400
    
    return jsonify({'plot': generate_plotly_json(fig)})

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
        
        # Store results in session
        session['results'] = results.to_json()
        session['training_time'] = elapsed
        session['last_model_id'] = model_id
        session['folds_used'] = fold
        
        # Add to history
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
        
        # Format results for response
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
    
    # Get top models for chart
    top_models = results.head(6).to_dict(orient='records')
    
    # Get best model metrics for radar
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

@app.route('/api/clean-data', methods=['POST'])
@session_required
def clean_data():
    """Apply data cleaning operations"""
    if not session.get('data'):
        return jsonify({'error': 'No data loaded'}), 404
    
    data = request.get_json()
    operation = data.get('operation')
    
    df = pd.read_json(session['data'], orient='split')
    original_shape = df.shape
    
    if operation == 'drop_duplicates':
        df = df.drop_duplicates().reset_index(drop=True)
        message = f"Removed {original_shape[0] - len(df)} duplicates"
    elif operation == 'drop_null_cols':
        df = df.dropna(axis=1, how='all')
        message = f"Removed {original_shape[1] - len(df.columns)} empty columns"
    else:
        return jsonify({'error': 'Invalid operation'}), 400
    
    session['data'] = df.to_json(orient='split')
    
    return jsonify({
        'success': True,
        'message': message,
        'new_shape': df.shape
    })

@app.route('/api/clear-session', methods=['POST'])
@session_required
def clear_session():
    """Clear current session data"""
    session['data'] = None
    session['results'] = None
    session['best_model'] = None
    session['training_history'] = []
    return jsonify({'success': True})


#  Error Handlers

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


#  Run Application

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)  
