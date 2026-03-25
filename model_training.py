import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

def train_and_save_model():
    """Train ensemble model using only 5 key features."""
    os.makedirs("models", exist_ok=True)
    try:
        data = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")
    except FileNotFoundError:
        print("❌ Dataset not found! Please place 'heart_failure_clinical_records_dataset.csv' in the 'data/' folder.")
        return None
    
    print(f"✅ Dataset loaded. Shape: {data.shape}")
    print(f"Class distribution:\n{data['DEATH_EVENT'].value_counts()}")
    key_features = [
        'age',
        'ejection_fraction', 
        'serum_creatinine',
        'serum_sodium',
        'time'
    ]
    
    print(f"\n✅ Using 5 key features: {key_features}")
    X = data[key_features]
    y = data["DEATH_EVENT"]
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "models/feature_names.pkl")
    print(f"✅ Feature names saved: {feature_names}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train
    )
    
    print("\n📊 After SMOTE class distribution:")
    print(pd.Series(y_train_resampled).value_counts())
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    svm = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,  # Reduced for 5 features
        max_depth=6,       # Adjusted for simpler model
        class_weight='balanced',
        random_state=42
    )
    print("\n🔄 Training models...")
    lr.fit(X_train_resampled, y_train_resampled)
    svm.fit(X_train_resampled, y_train_resampled)
    rf.fit(X_train_resampled, y_train_resampled)
    voting_clf = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
        voting='soft'
    )
    
    voting_clf.fit(X_train_resampled, y_train_resampled)
    print("✅ Ensemble model trained successfully!")
    y_prob = voting_clf.predict_proba(X_test_scaled)[:, 1]
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.3, 0.7, 0.01):
        y_pred_threshold = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n🎯 Optimal Threshold: {best_threshold:.2f} with F1 Score: {best_f1:.4f}")
    y_pred = (y_prob >= best_threshold).astype(int)
    print("\n" + "="*60)
    print("5-FEATURE MODEL EVALUATION")
    print("="*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    
    print(f"\n📈 Accuracy: {accuracy:.4f}")
    print(f"📈 AUC-ROC: {auc_roc:.4f}")
    print(f"📈 F1 Score: {best_f1:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    joblib.dump(voting_clf, "models/heart_disease_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("\n✅ Model and scaler saved successfully!")
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\n🏆 Feature Importance (5 features):")
    print(feature_importance_df.to_string(index=False))
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance - 5 Key Features')
    plt.tight_layout()
    plt.savefig("models/feature_importance_5features.png")
    print("✅ Feature importance plot saved!")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_roc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - 5 Feature Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/roc_curve_5features.png")
    print("✅ ROC curve plot saved!")
    
    return voting_clf, scaler, feature_importance_df

def shap_interpretability():
    """Generate SHAP interpretability plots for 5 features."""
    try:
        voting_clf = joblib.load("models/heart_disease_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        data = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")
        key_features = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']
        X = data[key_features]
        
        X_scaled = scaler.transform(X)
        
        rf_model = voting_clf.named_estimators_['rf']
        explainer = shap.TreeExplainer(rf_model)
        
        sample_indices = np.random.choice(len(X_scaled), min(100, len(X_scaled)), replace=False)
        X_sample = X_scaled[sample_indices]
        
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_to_plot,
            X_sample,
            feature_names=key_features,
            plot_type="bar",
            show=False
        )
        plt.title("SHAP Feature Importance - 5 Features")
        plt.tight_layout()
        plt.savefig("models/shap_summary_5features.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value,
            shap_values_to_plot[0],
            X_sample[0],
            feature_names=key_features,
        )
        
        # Save as HTML
        shap.save_html("models/shap_force_plot_5features.html", force_plot)
        
        print("\n✅ SHAP interpretability plots saved successfully!")
        
    except Exception as e:
        print(f"\n⚠️  SHAP plotting error: {e}")
        print("Continuing without SHAP plots...")

if __name__ == "__main__":
    print("🚀 Training 5-Feature Heart Failure Model")
    print("="*60)
    
    model_result = train_and_save_model()
    
    if model_result:
        shap_interpretability()
        
        print("\n" + "="*60)
        print("🎉 Training completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run the app: streamlit run app.py")
        print("2. Open browser: http://localhost:8501")
        print("3. Adjust the 5 sliders and predict!")