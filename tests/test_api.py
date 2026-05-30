from fastapi.testclient import TestClient

from gait_classification.api import app, trusted_users
from gait_classification.utils import ModelType

client = TestClient(app)


def test_models_page_renders_selection_window():
    response = client.get("/models")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Choose a model to inspect or use." in response.text
    assert "Select a model" in response.text
    assert "Transformer" in response.text
    assert "Lstm" in response.text
    assert "Go to model page" in response.text
    assert "onchange=" not in response.text


def test_model_page_renders_selected_model_window():
    response = client.get("/models/transformer")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Transformer model page." in response.text
    assert "Add trusted user" in response.text
    assert "Classify user" in response.text
    assert "#add-trusted-user" in response.text
    assert "/models/transformer/classify" in response.text
    assert "forward pass on our model" in response.text


def test_model_action_pages_render():
    upload_response = client.post(
        "/models/transformer/encode",
        files={"trusted_user_file": ("trusted.txt", b"sample trusted gait data", "text/plain")},
        data={"source_mode": "upload"},
    )
    classify_response = client.get("/models/transformer/classify")

    assert upload_response.status_code == 200
    assert "Trusted user embedding added from trusted.txt." in upload_response.text
    assert classify_response.status_code == 200
    assert "Classify a user with the Transformer model." in classify_response.text


def test_model_encode_page_accepts_recorded_data():
    before_count = len(trusted_users[ModelType.TRANSFORMER])
    response = client.post(
        "/models/transformer/encode",
        data={"source_mode": "record"},
    )

    assert response.status_code == 200
    assert "Trusted user embedding added from recorded data." in response.text
    assert len(trusted_users[ModelType.TRANSFORMER]) == before_count + 1


def test_root_page_renders_selection_window():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Choose a model to inspect or use." in response.text
    assert "Select a model" in response.text
    assert "Go to model page" in response.text
    assert "Go to model page" in response.text


def test_models_data_returns_available_models():
    response = client.get("/models/data")

    assert response.status_code == 200
    assert response.json() == {"available_models": ["transformer", "lstm"]}
