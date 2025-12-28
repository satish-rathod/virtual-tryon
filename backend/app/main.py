"""
Saree Virtual Try-On Backend API.

FastAPI application with endpoints for:
- Upload saree images
- Generate virtual try-on views
- Check job status
- Browse gallery
- Retrieve artifacts and logs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import uploads, generate, status
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting Saree Virtual Try-On API")
    logger.info(f"Storage root: {settings.get_storage_root()}")
    logger.info(f"Redis URL: {settings.REDIS_URL}")
    
    # Ensure storage directory exists
    settings.get_storage_root()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Saree Virtual Try-On API")


app = FastAPI(
    title="Saree Virtual Try-On API",
    description="Backend API for the Saree Virtual Try-On system",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(uploads.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(status.router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Saree Virtual Try-On API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
