import pytest
from src.app import app


@pytest.mark.e2e
def test_e2e_health():
    client = app.test_client()
    assert client.get("/health").status_code == 200
