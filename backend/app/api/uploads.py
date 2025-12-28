"""
Upload API endpoint.

POST /api/upload - Accept multipart file, generate UUID, save original.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.storage import (
    generate_saree_id,
    get_saree_dir,
    save_original,
    StorageError,
)
from app.models import UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_saree(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a floor-shot saree image.
    
    Accepts image files (JPEG, PNG, WebP).
    Creates a unique saree_id and saves the image to storage.
    
    Returns:
        UploadResponse with saree_id and upload_path
    """
    # Validate file type
    if not file.content_type:
        raise HTTPException(status_code=400, detail="File content type required")
    
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")
    
    # Generate unique saree ID
    saree_id = generate_saree_id()
    logger.info(f"Uploading saree with ID: {saree_id}")
    
    try:
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Save the original image
        saved_path = save_original(saree_id, content, file.filename)
        
        # Build relative upload path for response
        upload_path = f"artifacts/{saree_id}/{saved_path.name}"
        
        logger.info(f"Successfully uploaded saree {saree_id} to {saved_path}")
        
        return UploadResponse(
            saree_id=saree_id,
            upload_path=upload_path,
        )
        
    except HTTPException:
        raise
    except StorageError as e:
        logger.error(f"Storage error during upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

