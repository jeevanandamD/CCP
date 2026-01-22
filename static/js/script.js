// Form validation

document.addEventListener('DOMContentLoaded', function() {
    // Initialize form validation
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            let isValid = true;
            const inputs = form.querySelectorAll('input[required], select[required]');
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    isValid = false;
                    input.classList.add('is-invalid');
        
                    // Show error message
                    if (!input.nextElementSibling || !input.nextElementSibling.classList.contains('invalid-feedback')) {
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'invalid-feedback';
                        errorDiv.textContent = 'This field is required';
                        input.parentNode.insertBefore(errorDiv, input.nextSibling);
                    }
                } else {
                    input.classList.remove('is-invalid');
                    input.classList.add('is-valid');
                    
                    // Remove error message if exists
                    const errorDiv = input.nextElementSibling;
                    if (errorDiv && errorDiv.classList.contains('invalid-feedback')) {
                        errorDiv.remove();
                    }
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                showAlert('Please fill in all required fields.', 'warning');
            }
        });
    }

    
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            const min = parseFloat(this.getAttribute('min'));
            const max = parseFloat(this.getAttribute('max'));
            const value = parseFloat(this.value);
            
            if (!isNaN(value)) {
                if ((min !== null && value < min) || (max !== null && value > max)) {
                    this.classList.add('is-invalid');
                    this.classList.remove('is-valid');
                } else {
                    this.classList.remove('is-invalid');
                    this.classList.add('is-valid');
                }
            }
        });
    });
});

// Batch file upload
async function uploadBatch() {
    const fileInput = document.getElementById('batchFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('Please select a CSV file first.', 'warning');
        return;
    }
    
    // Validate file type
    if (!file.name.endsWith('.csv')) {
        showAlert('Please upload a CSV file.', 'error');
        return;
    }
    
    // Show loading
    Swal.fire({
        title: 'Processing...',
        text: 'Analyzing batch data. This may take a moment.',
        allowOutsideClick: false,
        didOpen: () => {
            Swal.showLoading();
        }
    });
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/batch_predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error || !data.success) {
            throw new Error(data.error || 'Unknown error occurred');
        }
        
        const summary = data.summary;
        
        // Show detailed results
        Swal.fire({
            title: '✅ Batch Analysis Complete!',
            html: `
                <div class="text-start" style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <h5 class="mb-3"><i class="fas fa-chart-pie me-2"></i>Summary Statistics</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card mb-2 border-0 bg-light">
                                <div class="card-body">
                                    <h6 class="text-muted mb-1">Total Patients</h6>
                                    <h3 class="text-primary">${summary.total_patients}</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card mb-2 border-0 bg-light">
                                <div class="card-body">
                                    <h6 class="text-muted mb-1">Avg Risk Probability</h6>
                                    <h3 class="text-info">${summary.avg_probability}%</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card mb-2 border-left-danger" style="border-left: 4px solid #dc3545;">
                                <div class="card-body">
                                    <h6 class="text-muted mb-1"><i class="fas fa-exclamation-triangle me-1"></i>High Risk</h6>
                                    <h3 class="text-danger">${summary.high_risk_count} (${summary.high_risk_percentage}%)</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card mb-2 border-left-success" style="border-left: 4px solid #28a745;">
                                <div class="card-body">
                                    <h6 class="text-muted mb-1"><i class="fas fa-check-circle me-1"></i>Low Risk</h6>
                                    <h3 class="text-success">${summary.low_risk_count} (${summary.low_risk_percentage}%)</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-6">
                            <div class="card mb-2 border-0 bg-light">
                                <div class="card-body">
                                    <h6 class="text-muted mb-1">Max Probability</h6>
                                    <h3 class="text-warning">${summary.max_probability}%</h3>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card mb-2 border-0 bg-light">
                                <div class="card-body">
                                    <h6 class="text-muted mb-1">Min Probability</h6>
                                    <h3 class="text-secondary">${summary.min_probability}%</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                    <hr class="my-3">
                    <p class="small text-muted mb-0">
                        <i class="fas fa-download me-1"></i>
                        Download results as JSON for further analysis
                    </p>
                </div>
            `,
            icon: 'success',
            showCancelButton: true,
            confirmButtonText: 'Download Results (JSON)',
            cancelButtonText: 'Close',
            width: '600px'
        }).then((result) => {
            if (result.isConfirmed) {
                // Download results as JSON
                downloadJSON({summary: summary, predictions: data.full_data}, 'batch_predictions_results.json');
                showAlert('Results downloaded successfully!', 'success');
            }
            // Clear file input
            fileInput.value = '';
        });
        
    } catch (error) {
        Swal.fire({
            title: '❌ Error!',
            text: error.message,
            icon: 'error',
            confirmButtonText: 'OK'
        });
        console.error('Batch prediction error:', error);
    }
}

// Download JSON helper function
function downloadJSON(data, filename) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Show alert helper function
function showAlert(message, type = 'info') {
    const alertClass = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    }[type];
    
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert ${alertClass} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Feature value helper
function showFeatureInfo(feature) {
    const info = {
        'age': 'Patient age in years. Advanced age is a risk factor.',
        'ejection_fraction': 'Percentage of blood pumped out of the heart with each beat. Normal: 50-70%.',
        'serum_creatinine': 'Kidney function marker. High levels indicate kidney impairment.',
        'serum_sodium': 'Electrolyte balance marker. Low levels can indicate heart failure.'
    };
    
    if (info[feature]) {
        showAlert(info[feature], 'info');
    }
}