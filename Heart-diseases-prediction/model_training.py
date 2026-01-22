# model_training.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_save_model():
    # Load dataset
    data = pd.read_csv("data/Heart Failure Clinical Records.csv")
    print(f"Dataset Shape: {data.shape}")
    print(f"Class distribution:\n{data['DEATH_EVENT'].value_counts()}")
    
    # Split features & target
    X = data.drop("DEATH_EVENT", axis=1)
    y = data["DEATH_EVENT"]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Models
    print("\nTraining models...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    
    lr.fit(X_train_scaled, y_train)
    svm.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)
    
    # Create Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
        voting='soft'  # Use soft voting for probability estimates
    )
    
    voting_clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = voting_clf.predict(X_test_scaled)
    y_pred_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nVoting Classifier Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model and scaler
    joblib.dump(voting_clf, 'heart_disease_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("\n" + "="*50)
    print("Model and scaler saved successfully!")
    print("="*50)
    
    # Feature importance (from Random Forest)
    feature_names = X.columns
    rf_importance = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance_df.head())
    
    return voting_clf, scaler, feature_importance_df

if __name__ == "__main__":
    model, scaler, feature_importance = train_and_save_model()