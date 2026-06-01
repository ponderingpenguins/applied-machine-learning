from fastapi.testclient import TestClient

from gait_classification.api import app

client = TestClient(app)


def test_root_page_renders_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Walk to Authenticate" in response.text
    assert "Get Started" in response.text


def test_methods_page_renders_options():
    # /models redirects to /methods; TestClient follows redirects by default
    response = client.get("/models")

    assert response.status_code == 200
    assert "Choose a detection method" in response.text
    assert "Select a model" in response.text
    assert "Transformer" in response.text
    assert "LSTM" in response.text
    assert "FFT Centroids" in response.text


def test_model_page_renders_for_transformer():
    response = client.get("/models/transformer")

    assert response.status_code == 200
    assert "Gait Authentication" in response.text
    assert "Transformer" in response.text
    assert "Set up your walking profile" in response.text


def test_models_data_returns_available_models():
    response = client.get("/models/data")

    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert set(data["available_models"]) >= {"transformer", "lstm", "fft_centroids"}
