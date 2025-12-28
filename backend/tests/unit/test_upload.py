"""
Unit tests for the upload endpoint.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer


def test_upload_success(client, sample_image):
    """Test successful image upload."""
    response = client.post(
        "/api/upload",
        files={"file": ("test_saree.jpg", sample_image, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "saree_id" in data
    assert "upload_path" in data
    assert data["saree_id"]  # Not empty
    assert "artifacts/" in data["upload_path"]


def test_upload_missing_file(client):
    """Test upload without file."""
    response = client.post("/api/upload")
    
    assert response.status_code == 422  # Validation error


def test_upload_invalid_content_type(client):
    """Test upload with invalid content type."""
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_upload_empty_file(client):
    """Test upload with empty file."""
    response = client.post(
        "/api/upload",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    
    assert response.status_code == 400
    assert "Empty file" in response.json()["detail"]


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    assert "running" in response.json()["message"].lower()
