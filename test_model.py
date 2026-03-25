import pandas as pd
from pages.app import load_models

def test_model_loading():
    model, scaler, features, err = load_models()
    assert err is None
    assert model is not None
    assert len(features) == 5

def test_prediction():
    model, scaler, features, _ = load_models()
    sample = pd.DataFrame([[60, 35, 1.2, 137, 100]], columns=features)
    scaled = scaler.transform(sample)
    prob = model.predict_proba(scaled)[0][1]
    assert 0 <= prob <= 1