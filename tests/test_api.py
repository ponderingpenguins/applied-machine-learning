from fastapi.testclient import TestClient

from gait_classification.api import app


client = TestClient(app)


def test_models_page_renders_selection_window():
    response = client.get("/models")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Choose a model to inspect or use." in response.text
    assert "Select a model" in response.text
    assert "Transformer" in response.text
    assert "Lstm" in response.text


def test_model_page_renders_selected_model_window():
    response = client.get("/models/transformer")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Transformer" in response.text
    assert "/models/transformer" in response.text


def test_root_page_renders_selection_window():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Choose a model to inspect or use." in response.text
    assert "Select a model" in response.text


def test_models_data_returns_available_models():
    response = client.get("/models/data")

    assert response.status_code == 200
    assert response.json() == {"available_models": ["transformer", "lstm"]}