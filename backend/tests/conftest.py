"""
Pytest configuration and fixtures.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings


@pytest.fixture(scope="session")
def temp_storage_session():
    """Create a session-scoped temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "storage"
        storage_path.mkdir()
        
        with patch.object(settings, 'STORAGE_ROOT', storage_path):
            yield storage_path


@pytest.fixture
def temp_storage(tmp_path):
    """Create a function-scoped temporary storage directory."""
    storage_path = tmp_path / "storage"
    storage_path.mkdir()
    
    with patch.object(settings, 'STORAGE_ROOT', storage_path):
        yield storage_path


@pytest.fixture
def assets_dir(tmp_path):
    """Create a temporary assets directory."""
    assets_path = tmp_path / "assets"
    assets_path.mkdir()
    (assets_path / "model" / "poses").mkdir(parents=True)
    (assets_path / "layout").mkdir()
    
    with patch.object(settings, 'ASSETS_ROOT', assets_path):
        yield assets_path
