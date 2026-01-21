// Form validation and batch upload functionality

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
                    
                    // Add error message
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
    
    // Real-time validation for numeric inputs
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
        text: 'Analyzing batch data',
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
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Show results
        Swal.fire({
            title: 'Batch Analysis Complete!',
            html: `
                <div class="text-start">
                    <h5>Summary Statistics:</h5>
                    <ul class="list-unstyled">
                        <li>📊 Total Patients: <strong>${data.summary.total_patients}</strong></li>
                        <li>⚠️ High Risk: <strong>${data.summary.high_risk_count}</strong> (${data.summary.high_risk_percentage}%)</li>
                        <li>📈 Average Risk Probability: <strong>${data.summary.avg_probability}%</strong></li>
                    </ul>
                    <hr>
                    <p class="small text-muted">
                        <i class="fas fa-download me-1"></i>
                        Results can be downloaded as JSON for further analysis
                    </p>
                </div>
            `,
            icon: 'success',
            showCancelButton: true,
            confirmButtonText: 'Download Results',
            cancelButtonText: 'Close'
        }).then((result) => {
            if (result.isConfirmed) {
                // Download results as JSON
                downloadJSON(data, 'batch_predictions.json');
            }
        });
        
    } catch (error) {
        Swal.fire({
            title: 'Error!',
            text: error.message,
            icon: 'error',
            confirmButtonText: 'OK'
        });
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
