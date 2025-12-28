"""
AI Adapter interface and implementations.

Provides an abstract base adapter and a deterministic mock implementation
for controlled saree image transformations without external API calls.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)


@dataclass
class AdapterResponse:
    """Response from an AI adapter call."""
    output_path: Path
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class AdapterRequest:
    """Request to an AI adapter."""
    image_path: Path
    placement_map_path: Optional[Path] = None
    seed: int = 0
    strict_instructions: str = ""
    forbidden_changes: list[str] = field(default_factory=list)
    forbidden_changes: list[str] = field(default_factory=list)
    retry_instruction: Optional[str] = None
    description: Optional[str] = None


class AIAdapter(ABC):
    """Abstract base class for AI adapters."""
    
    @abstractmethod
    def run(
        self,
        image_path: Path,
        output_path: Path,
        placement_map_path: Optional[Path] = None,
        seed: int = 0,
        strict_instructions: str = "",
        forbidden_changes: Optional[list[str]] = None,
        retry_instruction: Optional[str] = None,
    ) -> AdapterResponse:
        """
        Run the AI adapter on an image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the output image
            placement_map_path: Optional placement map for guided transformations
            seed: Random seed for deterministic output
            strict_instructions: Instructions for the AI model
            forbidden_changes: List of prohibited changes
            retry_instruction: Additional instruction for retry attempts
            
        Returns:
            AdapterResponse with output path and metadata
        """
        pass
    
    @abstractmethod
    def flatten(self, image_path: Path, output_path: Path, seed: int = 0) -> AdapterResponse:
        """Flatten a saree image (remove folds)."""
        pass
    
    @abstractmethod
    def compose(
        self,
        saree_path: Path,
        pose_path: Path,
        output_path: Path,
        parts_paths: Optional[dict[str, Path]] = None,
        placement_map_path: Optional[Path] = None,
        seed: int = 0,
        retry_instruction: Optional[str] = None,
        description: Optional[str] = None,
    ) -> AdapterResponse:
        """Compose saree onto a pose/model image."""
        pass
    
    @abstractmethod
    def isolate_saree(
        self,
        image_path: Path,
        output_path: Path,
        seed: int = 0,
    ) -> AdapterResponse:
        """
        Extract only the saree from an image, removing background and persons.
        
        Args:
            image_path: Path to original image with saree
            output_path: Path to save isolated saree (RGBA with transparency)
            seed: Random seed for deterministic output
            
        Returns:
            AdapterResponse with output path and metadata
        """
        pass
    
    @abstractmethod
    def generate_flat_layout(
        self,
        image_path: Path,
        output_path: Path,
        seed: int = 0,
    ) -> AdapterResponse:
        """
        Generate a flat, rectangular layout of the saree.
        
        Args:
            image_path: Path to isolated saree image
            output_path: Path to save flat layout
            seed: Random seed for deterministic output
            
        Returns:
            AdapterResponse with output path and metadata
        """
        pass

    @abstractmethod
    def generate_description(
        self,
        image_path: Path,
        seed: int = 0,
    ) -> AdapterResponse:
        """
        Generate a textual description of the saree.
        
        Args:
            image_path: Path to saree image (flat or clean)
            seed: Random seed for deterministic output
            
        Returns:
            AdapterResponse with text field populated
        """
        pass


class MockAIAdapter(AIAdapter):
    """
    Deterministic mock AI adapter for testing.
    
    Performs reproducible image transformations using OpenCV/Pillow.
    No external API calls are made.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize the mock adapter.
        
        Args:
            log_dir: Optional directory to save adapter logs
        """
        self.log_dir = log_dir
        self.call_log: list[dict[str, Any]] = []
    
    def _compute_seed_hash(self, *args: Any) -> int:
        """Compute a deterministic hash from arguments for seeding."""
        hash_input = "|".join(str(a) for a in args)
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder='big')
    
    def _log_call(
        self,
        operation: str,
        image_path: Path,
        output_path: Path,
        seed: int,
        instructions: str,
        response: AdapterResponse,
    ) -> None:
        """Log an adapter call for debugging and retry tracking."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "input_path": str(image_path),
            "output_path": str(output_path),
            "seed": seed,
            "instructions": instructions,
            "success": response.success,
            "error": response.error,
            "metadata": response.metadata,
        }
        self.call_log.append(entry)
        
        if self.log_dir:
            log_path = self.log_dir / "adapter_log.json"
            try:
                existing = []
                if log_path.exists():
                    with open(log_path) as f:
                        existing = json.load(f)
                existing.append(entry)
                with open(log_path, 'w') as f:
                    json.dump(existing, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write adapter log: {e}")
    
    def run(
        self,
        image_path: Path,
        output_path: Path,
        placement_map_path: Optional[Path] = None,
        seed: int = 0,
        strict_instructions: str = "",
        forbidden_changes: Optional[list[str]] = None,
        retry_instruction: Optional[str] = None,
    ) -> AdapterResponse:
        """Generic transformation - applies subtle adjustments."""
        np.random.seed(seed)
        
        try:
            # Load image
            img = Image.open(image_path)
            
            # Apply deterministic, subtle transformations
            # Slight brightness adjustment based on seed
            brightness_factor = 0.95 + (seed % 10) * 0.01
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
            
            # Slight contrast adjustment
            contrast_factor = 0.98 + (seed % 5) * 0.01
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
            
            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, quality=95)
            
            response = AdapterResponse(
                output_path=output_path,
                seed=seed,
                metadata={
                    "operation": "generic_transform",
                    "brightness_factor": brightness_factor,
                    "contrast_factor": contrast_factor,
                },
                success=True,
            )
            
            self._log_call(
                "run",
                image_path,
                output_path,
                seed,
                strict_instructions,
                response,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Adapter run failed: {e}")
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                success=False,
                error=str(e),
            )
    
    def flatten(self, image_path: Path, output_path: Path, seed: int = 0) -> AdapterResponse:
        """
        Flatten a saree image (remove folds).
        
        Uses deterministic image processing to simulate flattening:
        - Bilateral filter for smoothing while preserving edges
        - Subtle perspective correction
        - Color normalization
        """
        np.random.seed(seed)
        
        try:
            # Load with OpenCV for processing
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            
            # Apply bilateral filter for fold smoothing while preserving edges
            # Parameters are deterministic based on seed
            d = 9 + (seed % 3)  # Diameter
            sigma_color = 75 + (seed % 25)
            sigma_space = 75 + (seed % 25)
            
            # Process RGB channels only (preserve alpha)
            rgb = img[:, :, :3]
            rgb_filtered = cv2.bilateralFilter(rgb, d, sigma_color, sigma_space)
            
            # Slight color normalization
            lab = cv2.cvtColor(rgb_filtered, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE for more uniform lighting
            clahe = cv2.createCLAHE(clipLimit=2.0 + (seed % 10) * 0.1, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            rgb_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Recombine with alpha channel
            result = np.dstack([rgb_normalized, img[:, :, 3]])
            
            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result)
            
            response = AdapterResponse(
                output_path=output_path,
                seed=seed,
                metadata={
                    "operation": "flatten",
                    "bilateral_d": d,
                    "sigma_color": sigma_color,
                    "sigma_space": sigma_space,
                },
                success=True,
            )
            
            self._log_call(
                "flatten",
                image_path,
                output_path,
                seed,
                "Flatten saree, remove folds",
                response,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Flatten failed: {e}")
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                success=False,
                error=str(e),
            )
    
    def compose(
        self,
        saree_path: Path,
        pose_path: Path,
        output_path: Path,
        parts_paths: Optional[dict[str, Path]] = None,
        placement_map_path: Optional[Path] = None,
        seed: int = 0,
        retry_instruction: Optional[str] = None,
        description: Optional[str] = None,
    ) -> AdapterResponse:
        """
        Compose saree onto a pose/model image.
        
        Mock implementation:
        - Overlays saree parts onto the model pose
        - Applies color/lighting adjustments for blending
        - Uses placement map if provided for guided placement
        """
        np.random.seed(seed)
        
        try:
            # Load images
            saree = cv2.imread(str(saree_path), cv2.IMREAD_UNCHANGED)
            pose = cv2.imread(str(pose_path), cv2.IMREAD_UNCHANGED)
            
            if saree is None:
                raise ValueError(f"Could not load saree: {saree_path}")
            if pose is None:
                raise ValueError(f"Could not load pose: {pose_path}")
            
            # Resize saree to fit pose dimensions
            pose_h, pose_w = pose.shape[:2]
            saree_h, saree_w = saree.shape[:2]
            
            # Calculate scaling to fit saree within pose
            scale = min(pose_w / saree_w, pose_h / saree_h) * 0.6
            new_w = int(saree_w * scale)
            new_h = int(saree_h * scale)
            
            saree_resized = cv2.resize(saree, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Calculate position (center-bottom area, typical for saree on model)
            x_offset = (pose_w - new_w) // 2
            y_offset = int(pose_h * 0.2)  # Start from 20% height
            
            # Create output image (start with pose)
            if pose.shape[2] == 3:
                pose = cv2.cvtColor(pose, cv2.COLOR_BGR2BGRA)
            result = pose.copy()
            
            # Ensure saree has alpha channel
            if saree_resized.shape[2] == 3:
                saree_resized = cv2.cvtColor(saree_resized, cv2.COLOR_BGR2BGRA)
            
            # Blend saree onto pose using alpha compositing
            for y in range(new_h):
                for x in range(new_w):
                    py = y_offset + y
                    px = x_offset + x
                    
                    if 0 <= py < pose_h and 0 <= px < pose_w:
                        alpha = saree_resized[y, x, 3] / 255.0
                        
                        if alpha > 0.1:  # Skip mostly transparent pixels
                            for c in range(3):
                                result[py, px, c] = int(
                                    alpha * saree_resized[y, x, c] +
                                    (1 - alpha) * result[py, px, c]
                                )
                            result[py, px, 3] = max(result[py, px, 3], saree_resized[y, x, 3])
            
            # Apply slight color adjustment based on seed for variety
            brightness = 0.98 + (seed % 5) * 0.01
            result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
            enhancer = ImageEnhance.Brightness(result_pil)
            result_pil = enhancer.enhance(brightness)
            
            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_pil.save(output_path, quality=95)
            
            response = AdapterResponse(
                output_path=output_path,
                seed=seed,
                metadata={
                    "operation": "compose",
                    "saree_scale": scale,
                    "position": {"x": x_offset, "y": y_offset},
                    "position": {"x": x_offset, "y": y_offset},
                    "retry_instruction": retry_instruction,
                    "description": description,
                },
                success=True,
            )
            
            self._log_call(
                "compose",
                saree_path,
                output_path,
                seed,
                f"Compose saree onto pose. Retry: {retry_instruction or 'N/A'}",
                response,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Compose failed: {e}")
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                success=False,
                error=str(e),
            )
    
    def isolate_saree(
        self,
        image_path: Path,
        output_path: Path,
        seed: int = 0,
    ) -> AdapterResponse:
        """
        Mock implementation: Use OpenCV thresholding for basic background removal.
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Otsu's thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find largest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                clean_mask = np.zeros_like(mask)
                cv2.drawContours(clean_mask, [largest], -1, 255, -1)
                mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            
            # Apply slight feathering
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            
            # Create RGBA output
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), rgba)
            
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                metadata={"operation": "mock_isolate", "method": "otsu_threshold"},
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Mock isolate failed: {e}")
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                success=False,
                error=str(e),
            )
    
    def generate_flat_layout(
        self,
        image_path: Path,
        output_path: Path,
        seed: int = 0,
    ) -> AdapterResponse:
        """
        Mock implementation: Simply copy and enhance the image.
        In production, this would generate a true flat layout.
        """
        try:
            img = Image.open(image_path)
            
            # Apply minor enhancements
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.02)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.05)
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, quality=95)
            
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                metadata={"operation": "mock_flat_layout", "method": "copy_enhance"},
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Mock flat layout failed: {e}")
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                success=False,
                error=str(e),
            )

    def generate_description(
        self,
        image_path: Path,
        seed: int = 0,
    ) -> AdapterResponse:
        """
        Mock implementation: Return a static description.
        """
        description = (
            "A beautiful pure silk Kanjeevaram saree in deep crimson red. "
            "The body features scattered gold zari buttas (small floral motifs). "
            "The border is broad (test_width_px), capable of heavy gold zari work with traditional mango (paisley) patterns. "
            "The pallu is rich and elaborate with geometric diamond patterns and peacock motifs. "
            "The fabric has a lustrous sheen characteristic of high-quality silk."
        )
        
        return AdapterResponse(
            output_path=Path("memory_only"),
            seed=seed,
            text=description,
            metadata={"operation": "mock_description"},
            success=True,
        )


class GeminiAIAdapter(AIAdapter):
    """
    Gemini AI adapter using Google's Gemini API for image generation.
    
    Uses the latest Gemini image generation models:
    - gemini-2.5-flash-image (Nano Banana) - Fast, 1024px
    - gemini-3-pro-image-preview (Nano Banana Pro) - High quality, up to 4K
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-image",
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize the Gemini adapter.
        
        Args:
            api_key: Gemini API key (falls back to env var if not provided)
            model: Model name to use
            log_dir: Optional directory to save adapter logs
        """
        self.api_key = api_key or self._get_api_key()
        self.model = model
        self.log_dir = log_dir
        self.call_log: list[dict[str, Any]] = []
        
        # Initialize the client
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._genai_available = True
        except ImportError:
            logger.warning("google-genai not installed, falling back to mock")
            self._genai_available = False
    
    def _get_api_key(self) -> str:
        """Get API key from environment or config."""
        import os
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("AI_ADAPTER_API_KEY")
        
        if api_key:
            return api_key
        
        # Try loading from .env file
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
                    if line.startswith("AI_ADAPTER_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        
        raise ValueError("GEMINI_API_KEY not found in environment or .env file")
    
    def _log_call(
        self,
        operation: str,
        input_path: Path,
        output_path: Path,
        seed: int,
        prompt: str,
        response: AdapterResponse,
    ) -> None:
        """Log an adapter call for debugging."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "model": self.model,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "seed": seed,
            "prompt": prompt[:500],  # Truncate long prompts
            "success": response.success,
            "error": response.error,
        }
        self.call_log.append(entry)
        
        if self.log_dir:
            log_path = self.log_dir / "gemini_adapter_log.json"
            try:
                existing = []
                if log_path.exists():
                    with open(log_path) as f:
                        existing = json.load(f)
                existing.append(entry)
                with open(log_path, 'w') as f:
                    json.dump(existing, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write adapter log: {e}")
    
    def _load_image_bytes(self, image_path: Path) -> tuple:
        """Load image and return bytes with mime type."""
        import io
        
        img = Image.open(image_path)
        
        # Determine format
        img_format = img.format or "PNG"
        mime_map = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg", 
            "PNG": "image/png",
            "WEBP": "image/webp",
        }
        mime_type = mime_map.get(img_format.upper(), "image/png")
        
        # Convert to bytes
        buffer = io.BytesIO()
        save_format = img_format if img_format.upper() in ["JPEG", "PNG", "WEBP"] else "PNG"
        if save_format == "JPEG" and img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buffer, format=save_format)
        
        return buffer.getvalue(), mime_type
    
    def _generate_with_gemini(
        self,
        prompt: str,
        input_image_path: Optional[Path] = None,
        reference_image_path: Optional[Path] = None,
        extra_images: Optional[list[tuple[str, Path]]] = None, # List of (label, path)
    ) -> Optional[Image.Image]:
        """
        Generate image using Gemini API.
        
        Args:
            prompt: Text prompt for generation
            input_image_path: Primary input image (e.g., saree)
            reference_image_path: Reference image (e.g., pose)
            
        Returns:
            Generated PIL Image or None on failure
        """
        if not self._genai_available:
            return None
        
        import io
        from google.genai import types
        
        try:
            # Build content parts
            contents = []
            
            # Add input image if provided
            if input_image_path and input_image_path.exists():
                img_bytes, mime_type = self._load_image_bytes(input_image_path)
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
            
            # Add reference image if provided (Pose)
            if reference_image_path and reference_image_path.exists():
                ref_bytes, ref_mime = self._load_image_bytes(reference_image_path)
                contents.append(types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime))
            
            # Add extra images (Parts, Placement Map)
            if extra_images:
                for label, path in extra_images:
                    if path and path.exists():
                        img_bytes, mime_type = self._load_image_bytes(path)
                        # We can't strictly label images in the API, so we mention them in the prompt
                        # But we add them to the content list.
                        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
            
            # Add text prompt
            contents.append(types.Part.from_text(text=prompt))
            
            # Generate
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            
            # Extract image from response
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data is not None:
                    image_data = part.inline_data.data
                    return Image.open(io.BytesIO(image_data))
            
            logger.warning("No image in Gemini response")
            return None
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None
    
    def run(
        self,
        image_path: Path,
        output_path: Path,
        placement_map_path: Optional[Path] = None,
        seed: int = 0,
        strict_instructions: str = "",
        forbidden_changes: Optional[list[str]] = None,
        retry_instruction: Optional[str] = None,
    ) -> AdapterResponse:
        """Generic transformation using Gemini."""
        
        prompt = (
            f"{strict_instructions} "
            f"{retry_instruction or ''} "
            "Apply subtle enhancements while preserving all original details."
        ).strip()
        
        try:
            result_image = self._generate_with_gemini(
                prompt=prompt,
                input_image_path=image_path,
            )
            
            if result_image:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result_image.save(output_path, quality=95)
                
                response = AdapterResponse(
                    output_path=output_path,
                    seed=seed,
                    metadata={"operation": "gemini_run", "model": self.model},
                    success=True,
                )
            else:
                # Fallback: copy input to output
                img = Image.open(image_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(output_path, quality=95)
                
                response = AdapterResponse(
                    output_path=output_path,
                    seed=seed,
                    metadata={"operation": "fallback_copy"},
                    success=True,
                )
            
            self._log_call("run", image_path, output_path, seed, prompt, response)
            return response
            
        except Exception as e:
            logger.error(f"Gemini run failed: {e}")
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                success=False,
                error=str(e),
            )
    
    def flatten(self, image_path: Path, output_path: Path, seed: int = 0) -> AdapterResponse:
        """Flatten saree image using Gemini."""
        
        prompt = (
            "This is a saree fabric image. Please flatten and enhance it: "
            "1. Remove any folds or wrinkles from the fabric. "
            "2. Make the fabric appear flat and spread out evenly. "
            "3. Preserve all the original patterns, colors, and textures exactly. "
            "4. Maintain the same resolution and aspect ratio. "
            "5. Do NOT add any new elements or change the design."
        )
        
        try:
            result_image = self._generate_with_gemini(
                prompt=prompt,
                input_image_path=image_path,
            )
            
            if result_image:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result_image.save(output_path, quality=95)
                
                response = AdapterResponse(
                    output_path=output_path,
                    seed=seed,
                    metadata={"operation": "gemini_flatten", "model": self.model},
                    success=True,
                )
            else:
                # Fallback: copy with minor enhancement
                img = Image.open(image_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(output_path, quality=95)
                
                response = AdapterResponse(
                    output_path=output_path,
                    seed=seed,
                    metadata={"operation": "fallback_copy"},
                    success=True,
                )
            
            self._log_call("flatten", image_path, output_path, seed, prompt, response)
            return response
            
        except Exception as e:
            logger.error(f"Gemini flatten failed: {e}")
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                success=False,
                error=str(e),
            )
    
    def compose(
        self,
        saree_path: Path,
        pose_path: Path,
        output_path: Path,
        parts_paths: Optional[dict[str, Path]] = None,
        placement_map_path: Optional[Path] = None,
        seed: int = 0,
        retry_instruction: Optional[str] = None,
        description: Optional[str] = None,
    ) -> AdapterResponse:
        """Compose saree onto a pose/model image."""
        
        base_prompt = (
            "You are given a collection of images to generating a Virtual Try-On result:\n"
            "1. MAIN SAREE: The full flattened saree fabric.\n"
            "2. MODEL POSE: Target model and pose.\n"
        )

        if placement_map_path:
            base_prompt += "3. PLACEMENT MAP: A color-coded guide showing where the saree parts should go on the body.\n"

        if parts_paths:
             base_prompt += "4. DETAILED PARTS: High-resolution crops of specific saree regions (Pallu, Borders, Body).\n"

        base_prompt += (
            "\nTASK: Generate a photorealistic image of the model wearing this saree.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- USE THE PARTS: You MUST use the textures from the provided 'Detailed Parts' (Pallu, Body, Borders) "
            "to ensure exact fidelity. The 'Main Saree' image is for global context, but the parts are for texture details.\n"
            "- MATCH THE MAP: If a placement map is provided, strictly follow its segmentation for where the fabric flows.\n"
            "- PRESERVE IDENTITY: The saree pattern, motifs, and colors must be IDENTICAL to the input. "
            "Do not hallucinate new designs.\n"
            "- NATURAL DRAPE: The fold and fall of the fabric must follow the model's body physics.\n"
        )
        
        if retry_instruction:
            base_prompt += f"\nADDITIONAL INSTRUCTIONS: {retry_instruction}\n"
            
        if description:
            base_prompt += f"\nSAREE DESCRIPTION: {description}\n"

        extra_images = []
        if placement_map_path:
            extra_images.append(("Placement Map", placement_map_path))
        
        if parts_paths:
            for name, path in parts_paths.items():
                extra_images.append((f"Part: {name}", path))
        
        try:
            result_image = self._generate_with_gemini(
                prompt=base_prompt,
                input_image_path=saree_path,
                reference_image_path=pose_path,
                extra_images=extra_images,
            )
            
            if result_image:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result_image.save(output_path, quality=95)
                
                response = AdapterResponse(
                    output_path=output_path,
                    seed=seed,
                    metadata={
                        "operation": "gemini_compose",
                        "model": self.model,
                        "retry_instruction": retry_instruction,
                    },
                    success=True,
                )
            else:
                # Fallback to mock compose
                logger.warning("Gemini compose failed, using mock fallback")
                mock = MockAIAdapter(log_dir=self.log_dir)
                return mock.compose(
                    saree_path=saree_path,
                    pose_path=pose_path, 
                    output_path=output_path,
                    placement_map_path=placement_map_path,
                    seed=seed,
                    retry_instruction=retry_instruction,
                )
            
            self._log_call("compose", saree_path, output_path, seed, base_prompt, response)
            return response
            
        except Exception as e:
            logger.error(f"Gemini compose failed: {e}")
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                success=False,
                error=str(e),
            )
    
    def isolate_saree(
        self,
        image_path: Path,
        output_path: Path,
        seed: int = 0,
    ) -> AdapterResponse:
        """Extract only the saree from an image using Gemini AI."""
        from app.core.image_utils import resize_for_api
        
        prompt = (
            "OBJECTIVE: Perform a strict background removal and object extraction.\n\n"
            "INPUT: An image containing a saree (Indian garment).\n"
            "TASK: Create a transparent PNG containing ONLY the saree fabric.\n\n"
            "STRICT RULES:\n"
            "1. NO PERSONS: Remove all visible body parts (hands, face, feet, mannequin). The output must be fabric ONLY.\n"
            "2. NO BACKGROUND: Remove walls, floors, hangers, and any other surounding objects.\n"
            "3. KEEP EXACT PIXELS: Do not generate new patterns, do not change colors, do not 'enhance' the design.\n"
            "4. TRANSPARENCY: The background must be 100% transparent alpha channel.\n"
            "5. COMPLETENESS: Ensure the entire visible saree is extracted, do not crop parts of the fabric.\n"
            "6. OUTPUT: Return the resulting image only."
        )
        
        try:
            # Resize for API
            resized_img = resize_for_api(image_path, max_size=2048)
            
            # Generate with Gemini
            result_image = self._generate_with_gemini(
                prompt=prompt,
                input_image_path=image_path,
            )
            
            if result_image:
                # Convert to RGBA for transparency support
                if result_image.mode != "RGBA":
                    result_image = result_image.convert("RGBA")
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result_image.save(output_path, "PNG")
                
                response = AdapterResponse(
                    output_path=output_path,
                    seed=seed,
                    metadata={"operation": "gemini_isolate", "model": self.model},
                    success=True,
                )
                self._log_call("isolate_saree", image_path, output_path, seed, prompt, response)
                return response
            else:
                # Fallback to mock
                logger.warning("Gemini isolate failed, using mock fallback")
                mock = MockAIAdapter(log_dir=self.log_dir)
                return mock.isolate_saree(image_path, output_path, seed)
                
        except Exception as e:
            logger.error(f"Gemini isolate failed: {e}")
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                success=False,
                error=str(e),
            )
    
    def generate_flat_layout(
        self,
        image_path: Path,
        output_path: Path,
        seed: int = 0,
    ) -> AdapterResponse:
        """Generate a flat, rectangular layout of the saree using Gemini AI."""
        from app.core.image_utils import resize_for_api
        
        prompt = (
            "OBJECTIVE: Transform this saree image into a technical flat lay for pattern cutting.\n\n"
            "INPUT: An image of a saree (extracted/transparent background).\n"
            "TASK: Warping/Rectifying the fabric into a perfect flat rectangle.\n\n"
            "CRITICAL FIDELITY RULES:\n"
            "1. NO NEW GENERATION: You must NOT generate new patterns or designs. You must ONLY warp/move the existing pixels.\n"
            "2. EXACT TEXTURE COPY: The texture, weave, and embroidery must match the input EXACTLY.\n"
            "3. NO HALLUCINATION: Do not add borders where there are none. Do not change colors.\n\n"
            "LAYOUT REQUIREMENTS:\n"
            "1. SHAPE: A perfect wide rectangle (approx 4:1 aspect ratio).\n"
            "2. BACKGROUND: The output MUST have a transparent background. Do not add white or black padding.\n"
            "3. STRUCTURE: \n"
            "   - Left side: Pallu (the decorative end piece)\n"
            "   - Middle: Main body pleats/design\n"
            "   - Top/Bottom Edges: The borders running straight across.\n"
            "4. FLATTENING: Remove all folds and wrinkles digitally, as if ironing the fabric flat.\n"
            "5. ORIENTATION: Horizontal layout."
        )
        
        try:
            # Resize for API
            resized_img = resize_for_api(image_path, max_size=2048)
            
            # Generate with Gemini
            result_image = self._generate_with_gemini(
                prompt=prompt,
                input_image_path=image_path,
            )
            
            if result_image:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result_image.save(output_path, quality=95)
                
                response = AdapterResponse(
                    output_path=output_path,
                    seed=seed,
                    metadata={
                        "operation": "gemini_flat_layout",
                        "model": self.model,
                        "output_size": result_image.size,
                    },
                    success=True,
                )
                self._log_call("generate_flat_layout", image_path, output_path, seed, prompt, response)
                return response
            else:
                # Fallback to mock
                logger.warning("Gemini flat layout failed, using mock fallback")
                mock = MockAIAdapter(log_dir=self.log_dir)
                return mock.generate_flat_layout(image_path, output_path, seed)
                
        except Exception as e:
            logger.error(f"Gemini flat layout failed: {e}")
            return AdapterResponse(
                output_path=output_path,
                seed=seed,
                success=False,
                error=str(e),
            )

    def generate_description(
        self,
        image_path: Path,
        seed: int = 0,
    ) -> AdapterResponse:
        """Generate textual description using Gemini."""
        prompt = (
            "Detailed Visual Analysis of Indian Saree:\n"
            "Analyze this saree image and provide a comprehensive textual description.\n"
            "Focus on:\n"
            "1. Fabric type and texture (e.g., silk, cotton, shear, metallic sheen).\n"
            "2. Base color and secondary colors.\n"
            "3. Border details: width (broad/narrow), motifs, zari work type.\n"
            "4. Pallu design: heavy/simple, specific motifs (peacocks, geometric, floral).\n"
            "5. Body patterns: plain, buttas, checks, stripes, etc.\n"
            "NOTE: Be precise and descriptive as this text will be used for indexing and search."
        )
        
        try:
            # We use the text generation capability of the vision model
            if not self._genai_available:
                return MockAIAdapter(log_dir=self.log_dir).generate_description(image_path, seed)

            import io
            from google.genai import types
            
            contents = []
            
            # Add image
            if image_path.exists():
                img_bytes, mime_type = self._load_image_bytes(image_path)
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
            
            # Add prompt
            contents.append(types.Part.from_text(text=prompt))
            
            # Generate
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
            )
            
            text_result = response.text
            
            return AdapterResponse(
                output_path=Path("memory_only"),  # No specific output file for this step
                seed=seed,
                text=text_result,
                metadata={"operation": "gemini_description", "model": self.model},
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Gemini description generation failed: {e}")
            # Fallback to mock
            return MockAIAdapter(log_dir=self.log_dir).generate_description(image_path, seed)


def get_adapter(adapter_type: str = "mock", log_dir: Optional[Path] = None, **kwargs) -> AIAdapter:
    """
    Factory function to get an AI adapter instance.
    
    Args:
        adapter_type: "mock", "gemini", or "external"
        log_dir: Optional directory to save adapter logs
        **kwargs: Additional arguments passed to adapter constructor
        
    Returns:
        AIAdapter instance
    """
    if adapter_type == "mock":
        return MockAIAdapter(log_dir=log_dir)
    elif adapter_type in ("gemini", "external"):
        return GeminiAIAdapter(log_dir=log_dir, **kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
