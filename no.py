# app.py
from flask import Flask, render_template, request, jsonify, session
import joblib
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import json
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Load the trained model and scaler
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully!")
except:
    print("Warning: Model files not found. Please run model_training.py first.")
    model = None
    scaler = None

# Feature names (in correct order)
feature_names = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

# Feature descriptions for tooltips
feature_descriptions = {
    'age': 'Age of the patient (years)',
    'anaemia': 'Decrease of red blood cells or hemoglobin (0=No, 1=Yes)',
    'creatinine_phosphokinase': 'Level of the CPK enzyme in the blood (mcg/L)',
    'diabetes': 'If the patient has diabetes (0=No, 1=Yes)',
    'ejection_fraction': 'Percentage of blood leaving the heart at each contraction (%)',
    'high_blood_pressure': 'If the patient has hypertension (0=No, 1=Yes)',
    'platelets': 'Platelets in the blood (kiloplatelets/mL)',
    'serum_creatinine': 'Level of serum creatinine in the blood (mg/dL)',
    'serum_sodium': 'Level of serum sodium in the blood (mEq/L)',
    'sex': 'Sex of the patient (0=Female, 1=Male)',
    'smoking': 'If the patient smokes (0=No, 1=Yes)',
    'time': 'Follow-up period (days)'
}

# Normal ranges for features
normal_ranges = {
    'age': (0, 120),
    'creatinine_phosphokinase': (10, 120),  # Normal: 10-120 mcg/L
    'ejection_fraction': (50, 75),  # Normal: 50-75%
    'platelets': (150000, 450000),  # Normal: 150k-450k
    'serum_creatinine': (0.6, 1.3),  # Normal: 0.6-1.3 mg/dL
    'serum_sodium': (135, 145),  # Normal: 135-145 mEq/L
    'time': (0, 365)
}

@app.route('/')
def home():
    """Render home page with prediction form"""
    return render_template('index.html', 
                         features=feature_names,
                         descriptions=feature_descriptions,
                         ranges=normal_ranges)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    try:
        # Get data from form
        input_data = []
        for feature in feature_names:
            value = request.form.get(feature, '').strip()
            if value == '':
                return jsonify({'error': f'Please provide value for {feature}'})
            
            # Convert to appropriate type
            if feature in ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']:
                input_data.append(int(float(value)))
            else:
                input_data.append(float(value))
        
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Get feature importances (from individual models if available)
        if hasattr(model, 'estimators_'):
            rf_model = model.estimators_[2]  # Random Forest
            importances = rf_model.feature_importances_
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)
        
        # Store in session for dashboard
        session['last_prediction'] = {
            'features': dict(zip(feature_names, input_data)),
            'prediction': int(prediction),
            'probability': float(probability),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_importances': dict(zip(feature_names, importances.tolist()))
        }
        
        # Prepare response
        result = {
            'prediction': 'High Risk of Heart Failure' if prediction == 1 else 'Low Risk of Heart Failure',
            'probability': round(probability * 100, 2),
            'recommendation': get_recommendation(prediction, probability, input_data),
            'feature_values': dict(zip(feature_names, input_data)),
            'feature_importances': dict(zip(feature_names, importances.tolist())),
            'abnormal_values': check_abnormal_values(input_data)
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/dashboard')
def dashboard():
    """Show prediction dashboard with analytics"""
    if 'last_prediction' not in session:
        return render_template('dashboard.html', 
                             has_data=False,
                             feature_names=feature_names)
    
    pred_data = session['last_prediction']
    
    # Create visualizations
    graphs = create_visualizations(pred_data)
    
    return render_template('dashboard.html',
                         has_data=True,
                         prediction=pred_data,
                         graphs=graphs,
                         feature_names=feature_names,
                         descriptions=feature_descriptions)

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions from CSV file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Check if all required columns are present
        missing_cols = set(feature_names) - set(df.columns)
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'})
        
        # Prepare data
        X = df[feature_names]
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['probability'] = probabilities
        df['risk_level'] = np.where(predictions == 1, 'High Risk', 'Low Risk')
        
        # Generate summary statistics
        summary = {
            'total_patients': len(df),
            'high_risk_count': int(predictions.sum()),
            'high_risk_percentage': round(predictions.mean() * 100, 2),
            'avg_probability': round(probabilities.mean() * 100, 2)
        }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'predictions': df.to_dict('records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

def get_recommendation(prediction, probability, input_data):
    """Generate recommendations based on prediction"""
    recommendations = []
    
    if prediction == 1 or probability > 0.6:
        recommendations.append("🚨 **Urgent Action Required:** Consult a cardiologist immediately.")
        recommendations.append("📋 Schedule comprehensive cardiac evaluation.")
    
    # Specific recommendations based on abnormal values
    if input_data[feature_names.index('ejection_fraction')] < 40:
        recommendations.append("💓 **Low Ejection Fraction:** Consider echocardiogram and possible medication adjustment.")
    
    if input_data[feature_names.index('serum_creatinine')] > 1.5:
        recommendations.append("🧪 **Elevated Creatinine:** Monitor kidney function and adjust medications if needed.")
    
    if input_data[feature_names.index('serum_sodium')] < 135:
        recommendations.append("🧂 **Low Sodium:** Electrolyte imbalance detected. Review medication and diet.")
    
    if input_data[feature_names.index('age')] > 70:
        recommendations.append("👴 **Advanced Age:** Regular follow-ups and comprehensive geriatric assessment recommended.")
    
    # General lifestyle recommendations
    recommendations.append("🏃 **Lifestyle:** Regular moderate exercise (30 min/day, 5 days/week)")
    recommendations.append("🍎 **Diet:** Low-sodium, heart-healthy diet rich in fruits and vegetables")
    recommendations.append("🚭 **Avoid:** Smoking and excessive alcohol consumption")
    recommendations.append("😴 **Stress:** Practice stress management techniques")
    
    return recommendations

def check_abnormal_values(input_data):
    """Check which values are outside normal ranges"""
    abnormal = []
    for i, feature in enumerate(feature_names):
        if feature in normal_ranges:
            min_val, max_val = normal_ranges[feature]
            value = input_data[i]
            
            # For binary features, just check if they're present
            if feature in ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking']:
                if value == 1:
                    abnormal.append(feature)
            elif value < min_val or value > max_val:
                abnormal.append(feature)
    
    return abnormal

def create_visualizations(pred_data):
    """Create Plotly graphs for dashboard"""
    graphs = []
    
    # 1. Risk Probability Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_data['probability'] * 100,
        title={'text': "Risk Probability (%)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    graphs.append(json.dumps(gauge, cls=plotly.utils.PlotlyJSONEncoder))
    
    # 2. Feature Importance Bar Chart
    if 'feature_importances' in pred_data:
        importances = pred_data['feature_importances']
        features = list(importances.keys())
        values = list(importances.values())
        
        # Sort by importance
        sorted_idx = np.argsort(values)[::-1]
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        bar_fig = go.Figure(data=[
            go.Bar(x=features[:10], y=values[:10],
                  marker_color='royalblue')
        ])
        bar_fig.update_layout(
            title="Top 10 Most Important Features",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            height=400,
            margin=dict(l=20, r=20, t=50, b=100)
        )
        graphs.append(json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder))
    
    return graphs

@app.route('/api/model_info')
def model_info():
    """Return information about the trained model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    info = {
        'model_type': type(model).__name__,
        'n_features': len(feature_names),
        'features': feature_names,
        'feature_descriptions': feature_descriptions,
        'normal_ranges': normal_ranges
    }
    
    return jsonify(info)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("\n" + "="*50)
    print("Heart Disease Prediction App")
    print("="*50)
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("\nEndpoints:")
    print("  /                   - Home page with prediction form")
    print("  /predict            - Make prediction (POST)")
    print("  /dashboard          - Prediction analytics dashboard")
    print("  /api/batch_predict  - Batch predictions from CSV")
    print("  /api/model_info     - Model information")
    print("\n" + "="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)