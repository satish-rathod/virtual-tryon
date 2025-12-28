"""
Unit tests for the generate endpoint.
"""

import io
import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.core.config import settings


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage directory."""
    storage_path = tmp_path / "storage"
    storage_path.mkdir()
    
    # Patch settings to use temp storage
    with patch.object(settings, 'STORAGE_ROOT', storage_path):
        yield storage_path


@pytest.fixture
def sample_saree(temp_storage):
    """Create a sample uploaded saree."""
    # Create a saree directory with original image
    from app.core.storage import generate_saree_id, get_saree_dir
    
    saree_id = generate_saree_id()
    saree_dir = temp_storage / saree_id
    saree_dir.mkdir()
    
    # Create a sample image
    img = Image.new('RGB', (200, 300), color='blue')
    img.save(saree_dir / "original.jpg")
    
    return saree_id


def test_generate_success(client, temp_storage, sample_saree):
    """Test successful generate request."""
    with patch.object(settings, 'STORAGE_ROOT', temp_storage):
        # Mock Redis and RQ
        with patch('app.api.generate.Redis') as mock_redis:
            with patch('app.api.generate.Queue') as mock_queue:
                mock_queue_instance = MagicMock()
                mock_queue.return_value = mock_queue_instance
                
                response = client.post(
                    "/api/generate",
                    json={"saree_id": sample_saree, "mode": "standard"}
                )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "job_id" in data
        assert data["status"] == "queued"
        
        # Verify job.json was created
        job_path = temp_storage / sample_saree / "job.json"
        assert job_path.exists()
        
        with open(job_path) as f:
            job_data = json.load(f)
        
        assert job_data["saree_id"] == sample_saree
        assert job_data["mode"] == "standard"
        assert job_data["status"] == "queued"
        assert len(job_data["poses"]) == 4  # Standard mode has 4 poses


def test_generate_saree_not_found(client, temp_storage):
    """Test generate with non-existent saree."""
    with patch.object(settings, 'STORAGE_ROOT', temp_storage):
        response = client.post(
            "/api/generate",
            json={"saree_id": "non-existent-id", "mode": "standard"}
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def test_generate_extend_mode(client, temp_storage, sample_saree):
    """Test generate with extend mode."""
    with patch.object(settings, 'STORAGE_ROOT', temp_storage):
        with patch('app.api.generate.Redis') as mock_redis:
            with patch('app.api.generate.Queue') as mock_queue:
                mock_queue_instance = MagicMock()
                mock_queue.return_value = mock_queue_instance
                
                response = client.post(
                    "/api/generate",
                    json={"saree_id": sample_saree, "mode": "extend"}
                )
        
        assert response.status_code == 200
        
        # Verify 8 poses for extend mode
        job_path = temp_storage / sample_saree / "job.json"
        with open(job_path) as f:
            job_data = json.load(f)
        
        assert len(job_data["poses"]) == 8


def test_generate_retry_failed_no_previous_job(client, temp_storage, sample_saree):
    """Test retry_failed without a previous job."""
    with patch.object(settings, 'STORAGE_ROOT', temp_storage):
        response = client.post(
            "/api/generate",
            json={"saree_id": sample_saree, "mode": "retry_failed"}
        )
        
        assert response.status_code == 400
        assert "No previous job" in response.json()["detail"]


def test_generate_invalid_mode(client, temp_storage, sample_saree):
    """Test generate with invalid mode."""
    with patch.object(settings, 'STORAGE_ROOT', temp_storage):
        response = client.post(
            "/api/generate",
            json={"saree_id": sample_saree, "mode": "invalid_mode"}
        )
        
        # Pydantic validation should reject invalid mode
        assert response.status_code == 422
