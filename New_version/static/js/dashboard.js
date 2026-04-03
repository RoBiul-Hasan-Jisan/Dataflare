// Global variables
let currentData = null;
let trainingInProgress = false;

// Initialize on page load
$(document).ready(function() {
    initializeEventListeners();
    loadDataInfo();
});

function initializeEventListeners() {
    $('#uploadForm').on('submit', handleUpload);
    $('#targetSelect').on('change', handleTargetChange);
    $('#trainSize').on('input', function() {
        const value = $(this).val();
        const testSize = ((1 - value) * 100).toFixed(0);
        $(this).next().text(`${(value * 100).toFixed(0)}% Training / ${testSize}% Testing`);
    });
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertDiv = $(`
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    $('#alertContainer').append(alertDiv);
    setTimeout(() => alertDiv.alert('close'), 5000);
}

// Handle file upload
async function handleUpload(e) {
    e.preventDefault();
    const fileInput = $('#fileInput')[0];
    if (!fileInput.files[0]) {
        showAlert('Please select a file to upload', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    showAlert('Uploading file...', 'info');
    $('#uploadForm button').prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Uploading...');
    
    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await response.json();
        
        if (data.success) {
            showAlert(`Successfully uploaded ${data.filename} (${data.rows} rows, ${data.columns} columns)`, 'success');
            await loadDataInfo();
            await loadDataPreview();
            enableTrainingConfig();
        } else {
            showAlert(data.error || 'Upload failed', 'danger');
        }
    } catch (error) {
        showAlert('Error uploading file: ' + error.message, 'danger');
    } finally {
        $('#uploadForm button').prop('disabled', false).html('<i class="fas fa-cloud-upload-alt"></i> Upload');
    }
}

// Load sample dataset
async function loadSample(sampleName) {
    showAlert(`Loading ${sampleName} dataset...`, 'info');
    
    try {
        const response = await fetch('/api/load-sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sample: sampleName })
        });
        const data = await response.json();
        
        if (data.success) {
            showAlert(`Successfully loaded ${sampleName} dataset (${data.rows} rows, ${data.columns} columns)`, 'success');
            await loadDataInfo();
            await loadDataPreview();
            enableTrainingConfig();
            
            if (data.target_hint) {
                $('#targetSelect').val(data.target_hint).trigger('change');
            }
        } else {
            showAlert(data.error || 'Failed to load sample', 'danger');
        }
    } catch (error) {
        showAlert('Error loading sample: ' + error.message, 'danger');
    }
}

// Load data information
async function loadDataInfo() {
    try {
        const response = await fetch('/api/data-info');
        const data = await response.json();
        
        if (data.error) {
            $('#dataInfoCard').hide();
            return;
        }
        
        $('#dataInfoCard').show();
        const infoHtml = `
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>${data.rows.toLocaleString()}</h3>
                    <p>Rows</p>
                </div>
                <div class="stat-card">
                    <h3>${data.columns}</h3>
                    <p>Columns</p>
                </div>
                <div class="stat-card">
                    <h3>${data.num_cols}</h3>
                    <p>Numeric</p>
                </div>
                <div class="stat-card">
                    <h3>${data.cat_cols}</h3>
                    <p>Categorical</p>
                </div>
            </div>
            <div class="mt-3">
                <div class="progress">
                    <div class="progress-bar bg-success" style="width: ${data.health_score}%">
                        Health: ${data.health_score}%
                    </div>
                </div>
                <small class="text-muted">
                    Missing: ${data.null_pct}% | Duplicates: ${data.duplicates}
                </small>
            </div>
            <div class="mt-3">
                <strong>Columns:</strong>
                <div class="table-responsive">
                    <table class="table table-sm">
                        <thead>
                            <tr><th>Column</th><th>Type</th><th>Non-Null</th><th>Unique</th></tr>
                        </thead>
                        <tbody id="columnList"></tbody>
                    </table>
                </div>
            </div>
        `;
        $('#dataInfoContent').html(infoHtml);
        
        // Load column details
        const colResponse = await fetch('/api/column-details');
        const colData = await colResponse.json();
        
        if (colData.details) {
            const tbody = $('#columnList');
            tbody.empty();
            colData.details.slice(0, 10).forEach(col => {
                tbody.append(`
                    <tr>
                        <td>${col.column}</td>
                        <td>${col.type}</td>
                        <td>${col.non_null.toLocaleString()}</td>
                        <td>${col.unique.toLocaleString()}</td>
                    </tr>
                `);
            });
            if (colData.details.length > 10) {
                tbody.append(`<tr><td colspan="4" class="text-muted">... and ${colData.details.length - 10} more columns</td></tr>`);
            }
        }
        
        // Populate target select
        const targetSelect = $('#targetSelect');
        targetSelect.empty();
        targetSelect.append('<option value="">Select target column...</option>');
        data.column_names.forEach(col => {
            targetSelect.append(`<option value="${col}">${col}</option>`);
        });
        
        // Populate EDA column select
        const edaSelect = $('#edaColumnSelect');
        edaSelect.empty();
        data.column_names.forEach(col => {
            edaSelect.append(`<option value="${col}">${col}</option>`);
        });
        
    } catch (error) {
        console.error('Error loading data info:', error);
    }
}

// Load data preview
async function loadDataPreview() {
    try {
        const response = await fetch('/api/data-preview?rows=10');
        const data = await response.json();
        
        if (data.error) return;
        
        let previewHtml = `
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-table"></i> Data Preview (First ${data.data.length} rows)
                </div>
                <div class="card-body table-responsive">
                    <table class="table table-sm table-bordered">
                        <thead>
                            <tr>
                                ${data.columns.map(col => `<th>${col}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody>
                            ${data.data.map(row => `
                                <tr>
                                    ${data.columns.map(col => `<td>${row[col] !== null ? row[col] : '<em>null</em>'}</td>`).join('')}
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        $('#overviewContent').html(previewHtml);
        
    } catch (error) {
        console.error('Error loading data preview:', error);
    }
}

// Enable training configuration
function enableTrainingConfig() {
    $('#trainConfigCard').show();
}

// Handle target column change
async function handleTargetChange() {
    const target = $('#targetSelect').val();
    if (!target) return;
    
    try {
        const response = await fetch('/api/detect-target', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target: target })
        });
        const data = await response.json();
        
        if (!data.error) {
            showAlert(`Target: ${target} (${data.type_label} problem with ${data.unique_values} unique values)`, 'info');
        }
    } catch (error) {
        console.error('Error detecting target:', error);
    }
}

// Train model
async function trainModel() {
    if (trainingInProgress) {
        showAlert('Training already in progress...', 'warning');
        return;
    }
    
    const target = $('#targetSelect').val();
    if (!target) {
        showAlert('Please select a target column', 'warning');
        return;
    }
    
    const config = {
        target: target,
        train_size: parseFloat($('#trainSize').val()),
        fold: parseInt($('#foldSelect').val()),
        normalize: $('#normalizeCheck').is(':checked'),
        remove_outliers: $('#removeOutliers').is(':checked'),
        max_models: parseInt($('#maxModels').val())
    };
    
    trainingInProgress = true;
    $('#trainBtn').prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Training...');
    showAlert('Training started. This may take a few minutes...', 'info');
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await response.json();
        
        if (data.success) {
            showAlert(`Training completed! Best model: ${data.best_model} (${data.metric_name}: ${data.best_score.toFixed(4)})`, 'success');
            await loadTrainingResults();
            await loadHistory();
            
            // Switch to training tab
            $('#mainTabs button[data-bs-target="#training"]').tab('show');
        } else {
            showAlert(`Training failed: ${data.message || data.error}`, 'danger');
        }
    } catch (error) {
        showAlert('Error during training: ' + error.message, 'danger');
    } finally {
        trainingInProgress = false;
        $('#trainBtn').prop('disabled', false).html('<i class="fas fa-play"></i> Start Training');
    }
}

// Load training results
async function loadTrainingResults() {
    try {
        const response = await fetch('/api/results');
        const data = await response.json();
        
        if (data.error || !data.exists) {
            $('#trainingContent').html(`
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> No training results yet. Train a model to see results.
                </div>
            `);
            return;
        }
        
        let resultsHtml = `
            <div class="card mb-3">
                <div class="card-header bg-success text-white">
                    <i class="fas fa-trophy"></i> Best Model: ${data.top_models[0].Model || data.top_models[0].model}
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h5>Training Summary</h5>
                            <ul class="list-group">
                                <li class="list-group-item d-flex justify-content-between">
                                    <span>Best Model:</span>
                                    <strong>${data.top_models[0].Model || data.top_models[0].model}</strong>
                                </li>
                                <li class="list-group-item d-flex justify-content-between">
                                    <span>Best Score:</span>
                                    <strong>${Object.values(data.best_metrics)[0]?.toFixed(4) || 'N/A'}</strong>
                                </li>
                                <li class="list-group-item d-flex justify-content-between">
                                    <span>Training Time:</span>
                                    <strong>${data.training_time ? data.training_time.toFixed(1) + 's' : 'N/A'}</strong>
                                </li>
                                <li class="list-group-item d-flex justify-content-between">
                                    <span>CV Folds:</span>
                                    <strong>${data.folds_used}</strong>
                                </li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h5>Actions</h5>
                            <div class="d-grid gap-2">
                                <button class="btn btn-primary" onclick="downloadModel('${data.model_id}')">
                                    <i class="fas fa-download"></i> Download Model
                                </button>
                                <button class="btn btn-success" onclick="downloadResults()">
                                    <i class="fas fa-file-csv"></i> Download Results CSV
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-chart-bar"></i> Model Comparison
                </div>
                <div class="card-body">
                    <div id="modelComparisonPlot"></div>
                    <div class="table-responsive mt-3">
                        <table class="table table-sm table-bordered">
                            <thead>
                                <tr>
                                    ${data.columns.map(col => `<th>${col}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${data.top_models.map(model => `
                                    <tr>
                                        ${data.columns.map(col => `<td>${model[col] !== undefined ? (typeof model[col] === 'number' ? model[col].toFixed(4) : model[col]) : 'N/A'}</td>`).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
        
        $('#trainingContent').html(resultsHtml);
        
        // Render comparison plot
        if (data.comparison_plot) {
            const plotData = JSON.parse(data.comparison_plot);
            Plotly.newPlot('modelComparisonPlot', plotData.data, plotData.layout);
        }
        
    } catch (error) {
        console.error('Error loading training results:', error);
    }
}

// Load distribution plot
async function loadDistribution() {
    const column = $('#edaColumnSelect').val();
    const chartType = $('#chartTypeSelect').val();
    
    if (!column) {
        showAlert('Please select a column', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/api/eda/distribution?column=${encodeURIComponent(column)}&chart_type=${chartType}`);
        const data = await response.json();
        
        if (data.error) {
            showAlert(data.error, 'danger');
            return;
        }
        
        // Render plot
        if (data.plot) {
            const plotData = JSON.parse(data.plot);
            Plotly.newPlot('distributionPlot', plotData.data, plotData.layout);
        }
        
        // Show statistics
        if (data.stats && Object.keys(data.stats).length > 0) {
            let statsHtml = '<div class="mt-3"><h6>Column Statistics</h6><table class="table table-sm">';
            for (const [key, value] of Object.entries(data.stats)) {
                statsHtml += `<tr><th>${key}</th><td>${value}</td></tr>`;
            }
            statsHtml += '</table></div>';
            $('#distributionStats').html(statsHtml);
        }
        
    } catch (error) {
        console.error('Error loading distribution:', error);
        showAlert('Error loading distribution', 'danger');
    }
}

// Load correlation plot
async function loadCorrelation() {
    try {
        const response = await fetch('/api/eda/correlation');
        const data = await response.json();
        
        if (data.error) {
            $('#correlationPlot').html(`<div class="alert alert-warning">${data.error}</div>`);
            return;
        }
        
        if (data.plot) {
            const plotData = JSON.parse(data.plot);
            Plotly.newPlot('correlationPlot', plotData.data, plotData.layout);
        }
        
    } catch (error) {
        console.error('Error loading correlation:', error);
        showAlert('Error loading correlation matrix', 'danger');
    }
}

// Load training history
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (!data.history || data.history.length === 0) {
            $('#historyContent').html(`
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> No training history yet.
                </div>
            `);
            return;
        }
        
        let historyHtml = `
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-history"></i> Training History (${data.history.length} sessions)
                </div>
                <div class="card-body table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Dataset</th>
                                <th>Problem Type</th>
                                <th>Best Model</th>
                                <th>Score</th>
                                <th>Rows</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.history.map(session => `
                                <tr>
                                    <td>${session.time}</td>
                                    <td>${session.dataset}</td>
                                    <td><span class="badge badge-${session.problem_type}">${session.problem_type}</span></td>
                                    <td>${session.best_model}</td>
                                    <td>${session.score}</td>
                                    <td>${session.rows.toLocaleString()}</td>
                                    <td>
                                        <button class="btn btn-sm btn-primary" onclick="downloadModel('${session.model_id}')">
                                            <i class="fas fa-download"></i>
                                        </button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        $('#historyContent').html(historyHtml);
        
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Download model
async function downloadModel(modelId) {
    try {
        window.location.href = `/api/download-model/${modelId}`;
        showAlert('Download started...', 'success');
    } catch (error) {
        showAlert('Error downloading model: ' + error.message, 'danger');
    }
}

// Download results
async function downloadResults() {
    try {
        window.location.href = '/api/download-results';
        showAlert('Download started...', 'success');
    } catch (error) {
        showAlert('Error downloading results: ' + error.message, 'danger');
    }
}

// Download history
async function downloadHistory() {
    try {
        window.location.href = '/api/download-history';
        showAlert('Download started...', 'success');
    } catch (error) {
        showAlert('Error downloading history: ' + error.message, 'danger');
    }
}

// Clear session
async function clearSession() {
    if (confirm('Are you sure you want to clear all data? This action cannot be undone.')) {
        try {
            const response = await fetch('/api/clear-session', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                showAlert('Session cleared successfully', 'success');
                location.reload();
            }
        } catch (error) {
            showAlert('Error clearing session: ' + error.message, 'danger');
        }
    }
}