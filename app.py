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
            'abnormal_values': check_abnormal_values(input_data),
            'risk_factors_explanation': get_risk_factors_explanation(input_data, importances),
            'risk_factor_chart': create_risk_factor_chart(feature_names, input_data, importances),
            'gauge_chart': create_gauge_chart(probability * 100)
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
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
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
            return jsonify({'error': f'Missing columns: {", ".join(missing_cols)}. Required columns: {", ".join(feature_names)}'})
        
        # Check for empty dataframe
        if df.empty:
            return jsonify({'error': 'CSV file is empty'})
        
        # Prepare data
        X = df[feature_names]
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['probability'] = (probabilities * 100).round(2)
        df['risk_level'] = df['prediction'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')
        
        # Generate summary statistics
        summary = {
            'total_patients': int(len(df)),
            'high_risk_count': int(predictions.sum()),
            'low_risk_count': int(len(df) - predictions.sum()),
            'high_risk_percentage': round(predictions.mean() * 100, 2),
            'low_risk_percentage': round((1 - predictions.mean()) * 100, 2),
            'avg_probability': round(probabilities.mean() * 100, 2),
            'max_probability': round(probabilities.max() * 100, 2),
            'min_probability': round(probabilities.min() * 100, 2)
        }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'predictions': df[['prediction', 'probability', 'risk_level']].to_dict('records'),
            'full_data': df.to_dict('records')
        })
    
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'CSV file is empty or corrupted'})
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

def create_gauge_chart(probability):
    """Create a gauge chart for risk probability"""
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Heart Failure Risk Probability (%)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue", 'thickness': 0.7},
            'steps': [
                {'range': [0, 30], 'color': "#90EE90"},
                {'range': [30, 70], 'color': "#FFD700"},
                {'range': [70, 100], 'color': "#FF6B6B"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    gauge.update_layout(
        height=350, 
        margin=dict(l=20, r=20, t=70, b=20),
        font={'size': 12}
    )
    return json.dumps(gauge, cls=plotly.utils.PlotlyJSONEncoder)

def create_risk_factor_chart(features, input_data, importances):
    """Create a horizontal bar chart showing risk factors"""
    # Convert importance dict to sorted list
    imp_dict = dict(zip(features, importances))
    sorted_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:8]
    
    feature_names_chart = [f.replace('_', ' ').title() for f, _ in sorted_features]
    importance_values = [v for _, v in sorted_features]
    
    fig = go.Figure(data=[
        go.Bar(
            y=feature_names_chart,
            x=importance_values,
            orientation='h',
            marker=dict(
                color=importance_values,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{v:.4f}' for v in importance_values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Risk Factor Importance Analysis",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        margin=dict(l=150, r=20, t=50, b=20),
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def get_risk_factors_explanation(input_data, importances):
    explanations = []
    
    # Get top 3 risk factors
    imp_dict = dict(zip(feature_names, importances))
    top_factors = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    
    for factor, importance in top_factors:
        idx = feature_names.index(factor)
        value = input_data[idx]
        
        # Generate explanation based on factor
        explanation = {
            'factor': factor.replace('_', ' ').title(),
            'value': value,
            'importance': round(importance * 100, 2),
            'status': '',
            'description': ''
        }
        
        if factor == 'ejection_fraction':
            if value < 40:
                explanation['status'] = 'Critical'
                explanation['description'] = f"Ejection fraction of {value}% is significantly below normal (50-75%). This indicates the heart is pumping less blood than normal with each beat."
            elif value < 50:
                explanation['status'] = 'Low'
                explanation['description'] = f"Ejection fraction of {value}% is slightly below normal range. This may indicate some heart weakness."
            else:
                explanation['status'] = 'Normal'
                explanation['description'] = f"Ejection fraction of {value}% is within healthy range (50-75%)."
        
        elif factor == 'serum_creatinine':
            if value > 1.5:
                explanation['status'] = 'Critical'
                explanation['description'] = f"Serum creatinine level of {value} mg/dL is elevated, suggesting kidney dysfunction which increases heart failure risk."
            elif value > 1.3:
                explanation['status'] = 'High'
                explanation['description'] = f"Serum creatinine level of {value} mg/dL is above normal (0.6-1.3), indicating possible kidney issues."
            else:
                explanation['status'] = 'Normal'
                explanation['description'] = f"Serum creatinine level of {value} mg/dL is within normal range."
        
        elif factor == 'serum_sodium':
            if value < 135:
                explanation['status'] = 'Critical'
                explanation['description'] = f"Serum sodium level of {value} mEq/L is below normal (135-145), causing electrolyte imbalance which is a serious risk factor."
            elif value > 145:
                explanation['status'] = 'High'
                explanation['description'] = f"Serum sodium level of {value} mEq/L is above normal range, indicating dehydration or sodium retention."
            else:
                explanation['status'] = 'Normal'
                explanation['description'] = f"Serum sodium level of {value} mEq/L is within normal range (135-145)."
        
        elif factor == 'age':
            if value > 70:
                explanation['status'] = 'High Risk'
                explanation['description'] = f"At age {int(value)}, the patient is in a higher risk age group. Advanced age is a significant risk factor for heart failure."
            elif value > 60:
                explanation['status'] = 'Moderate Risk'
                explanation['description'] = f"At age {int(value)}, the patient should monitor cardiovascular health regularly."
            else:
                explanation['status'] = 'Normal'
                explanation['description'] = f"At age {int(value)}, age is not a major risk factor."
        
        elif factor == 'creatinine_phosphokinase':
            if value > 500:
                explanation['status'] = 'High'
                explanation['description'] = f"CPK level of {value} mcg/L is elevated (normal: 10-120), suggesting muscle or heart damage."
            else:
                explanation['status'] = 'Normal'
                explanation['description'] = f"CPK level of {value} mcg/L is within normal range."
        
        elif factor == 'platelets':
            if value < 25000:
                explanation['status'] = 'Critical'
                explanation['description'] = f"Platelet count of {int(value)} is critically low (normal: 150k-450k), increasing bleeding and clotting risks."
            elif value < 100000:
                explanation['status'] = 'Low'
                explanation['description'] = f"Platelet count of {int(value)} is below normal range, which may affect clotting."
            else:
                explanation['status'] = 'Normal'
                explanation['description'] = f"Platelet count of {int(value)} is within normal range."
        
        elif factor == 'anaemia':
            if value == 1:
                explanation['status'] = 'Present'
                explanation['description'] = "Patient has anemia (low red blood cells/hemoglobin), reducing oxygen delivery to the heart and increasing stress."
            else:
                explanation['status'] = 'Not Present'
                explanation['description'] = "Patient does not have anemia."
        
        elif factor == 'diabetes':
            if value == 1:
                explanation['status'] = 'Present'
                explanation['description'] = "Patient has diabetes, which significantly increases the risk of heart failure due to metabolic stress."
            else:
                explanation['status'] = 'Not Present'
                explanation['description'] = "Patient does not have diabetes."
        
        elif factor == 'high_blood_pressure':
            if value == 1:
                explanation['status'] = 'Present'
                explanation['description'] = "Patient has high blood pressure (hypertension), which puts extra strain on the heart."
            else:
                explanation['status'] = 'Not Present'
                explanation['description'] = "Patient does not have high blood pressure."
        
        elif factor == 'smoking':
            if value == 1:
                explanation['status'] = 'Active'
                explanation['description'] = "Patient is a smoker, which significantly increases cardiovascular disease and heart failure risk."
            else:
                explanation['status'] = 'Non-smoker'
                explanation['description'] = "Patient is a non-smoker."
        
        explanations.append(explanation)
    
    return explanations

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

@app.route('/api/download_template')
def download_template():
    """Download CSV template for batch predictions"""
    # Create template with sample data
    template_data = {
        feature: [normal_ranges.get(feature, (0, 100))[0] if feature not in ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'] else 0] 
        for feature in feature_names
    }
    
    df_template = pd.DataFrame(template_data)
    
    # Convert to CSV
    csv_buffer = df_template.to_csv(index=False)
    
    # Create response
    from flask import Response
    return Response(
        csv_buffer,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=batch_prediction_template.csv"}
    )

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