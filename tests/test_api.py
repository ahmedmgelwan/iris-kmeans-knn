from fastapi.testclient import TestClient
from src.api import main

client = TestClient(app=main.app)

def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}

def test_predict():
    payload = {
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert 'class_label' in response.json()

def test_predict_invalid_input():
    payload = {
        "sepal_width": -1.0,  # Invalid value
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 422  
