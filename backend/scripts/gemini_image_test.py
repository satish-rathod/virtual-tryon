#!/usr/bin/env python3
"""
Gemini Image Generation Test Script

This script takes input images from a folder and generates output images
using Google's Gemini API (latest image generation models).

Usage:
    python gemini_image_test.py --input ./input_images --output ./output_images [--prompt "your prompt"]

Requirements:
    pip install google-genai pillow

API Key Setup:
    1. Get your API key from https://aistudio.google.com/apikey
    2. Set it as an environment variable: export GEMINI_API_KEY=your_key_here
    3. Or create a .env file with: GEMINI_API_KEY=your_key_here
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai package not installed")
    print("Install with: pip install google-genai")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: pillow package not installed")
    print("Install with: pip install pillow")
    sys.exit(1)


# Available Gemini Image Generation Models (as of Dec 2024)
# ============================================================
# 
# 1. gemini-2.5-flash-image (aka "Nano Banana")
#    - Fast and efficient, optimized for high-volume, low-latency tasks
#    - Generates images at 1024px resolution
#    - Best for: Quick prototyping, batch processing, speed-critical applications
#
# 2. gemini-3-pro-image-preview (aka "Nano Banana Pro Preview")
#    - Professional asset production, complex instructions
#    - Real-world grounding using Google Search
#    - Default "Thinking" process for refined composition
#    - Up to 4K resolution
#    - Best for: High-quality production, detailed editing, professional outputs
#
# Recommendation: Use gemini-2.5-flash-image for testing/speed
#                 Use gemini-3-pro-image-preview for production quality

# Default model (fastest & most reliable for testing)
DEFAULT_MODEL = "gemini-2.5-flash-image"

# Alternative high-quality model
PRO_MODEL = "gemini-3-pro-image-preview"


def get_api_key() -> str:
    """
    Get Gemini API key from environment or .env file.
    
    Priority:
    1. GEMINI_API_KEY environment variable
    2. AI_ADAPTER_API_KEY environment variable (for compatibility with existing config)
    3. .env file in current directory
    4. .env file in backend directory
    """
    # Check environment variables
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("AI_ADAPTER_API_KEY")
    
    if api_key:
        return api_key
    
    # Try loading from .env files
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent / ".env",  # backend/.env
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
                    if line.startswith("AI_ADAPTER_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    
    raise ValueError(
        "GEMINI_API_KEY not found!\n"
        "Please set it using one of these methods:\n"
        "  1. export GEMINI_API_KEY=your_key_here\n"
        "  2. Add GEMINI_API_KEY=your_key_here to backend/.env file\n"
        "  3. Get your key from: https://aistudio.google.com/apikey"
    )


def load_image_as_base64(image_path: Path) -> Tuple[str, str]:
    """Load an image and return as base64 with mime type."""
    img = Image.open(image_path)
    
    # Determine mime type
    format_to_mime = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }
    
    img_format = img.format or "PNG"
    mime_type = format_to_mime.get(img_format.upper(), "image/png")
    
    # Convert to bytes
    import io
    buffer = io.BytesIO()
    save_format = "PNG" if img_format.upper() not in ["JPEG", "JPG", "PNG", "WEBP", "GIF"] else img_format
    img.save(buffer, format=save_format)
    img_bytes = buffer.getvalue()
    
    return base64.standard_b64encode(img_bytes).decode("utf-8"), mime_type


def generate_image_from_input(
    client: genai.Client,
    input_image_path: Path,
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> Optional[Image.Image]:
    """
    Generate a new image based on an input image and prompt.
    
    This uses Gemini's image editing capability (text-and-image-to-image).
    """
    print(f"  Processing: {input_image_path.name}")
    
    # Load input image
    img_base64, mime_type = load_image_as_base64(input_image_path)
    
    # Create the content with image and text
    contents = [
        types.Part.from_bytes(
            data=base64.standard_b64decode(img_base64),
            mime_type=mime_type,
        ),
        types.Part.from_text(text=prompt),
    ]
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        
        # Extract the generated image from response
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                # Decode the image
                image_data = part.inline_data.data
                img = Image.open(io.BytesIO(image_data))
                return img
            elif hasattr(part, 'text') and part.text:
                print(f"    Model response: {part.text[:100]}...")
        
        print("    Warning: No image in response")
        return None
        
    except Exception as e:
        print(f"    Error: {e}")
        return None


def generate_image_text_only(
    client: genai.Client,
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> Optional[Image.Image]:
    """
    Generate an image from text prompt only (text-to-image).
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                image_data = part.inline_data.data
                img = Image.open(io.BytesIO(image_data))
                return img
            elif hasattr(part, 'text') and part.text:
                print(f"  Model response: {part.text[:100]}...")
        
        return None
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def process_folder(
    input_folder: Path,
    output_folder: Path,
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, int]:
    """
    Process all images in input folder and save results to output folder.
    
    Returns a summary dict with processed/failed counts.
    """
    # Get API key and create client
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    input_images = [
        f for f in input_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not input_images:
        print(f"No images found in {input_folder}")
        return {"processed": 0, "failed": 0, "skipped": 0}
    
    print(f"\nFound {len(input_images)} images to process")
    print(f"Using model: {model}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print("-" * 50)
    
    results = {"processed": 0, "failed": 0, "skipped": 0}
    
    for idx, input_path in enumerate(input_images, 1):
        print(f"\n[{idx}/{len(input_images)}] {input_path.name}")
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_generated_{timestamp}.png"
        output_path = output_folder / output_filename
        
        # Generate image
        result_image = generate_image_from_input(
            client=client,
            input_image_path=input_path,
            prompt=prompt,
            model=model,
        )
        
        if result_image:
            result_image.save(output_path)
            print(f"    Saved: {output_path.name}")
            results["processed"] += 1
        else:
            results["failed"] += 1
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Gemini API from input folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python gemini_image_test.py --input ./input --output ./output
    
    # With custom prompt
    python gemini_image_test.py --input ./input --output ./output \\
        --prompt "Transform this saree with a woman model wearing it elegantly"
    
    # Use high-quality model
    python gemini_image_test.py --input ./input --output ./output --model pro

Environment:
    GEMINI_API_KEY: Your Gemini API key (required)
                    Get one at: https://aistudio.google.com/apikey
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input folder containing images"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output folder for generated images"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Based on this saree image, generate a high-quality image of an Indian woman model wearing this exact saree. Preserve all the patterns, colors, and textures of the saree exactly as shown.",
        help="Prompt for image generation"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["flash", "pro"],
        default="flash",
        help="Model to use: 'flash' (fast, default) or 'pro' (high quality)"
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    if not args.input.exists():
        print(f"Error: Input folder does not exist: {args.input}")
        sys.exit(1)
    
    if not args.input.is_dir():
        print(f"Error: Input path is not a directory: {args.input}")
        sys.exit(1)
    
    # Select model
    model = PRO_MODEL if args.model == "pro" else DEFAULT_MODEL
    
    print("=" * 50)
    print("Gemini Image Generation Test")
    print("=" * 50)
    print(f"Input folder:  {args.input.absolute()}")
    print(f"Output folder: {args.output.absolute()}")
    
    # Process folder
    results = process_folder(
        input_folder=args.input,
        output_folder=args.output,
        prompt=args.prompt,
        model=model,
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Processed: {results['processed']}")
    print(f"  Failed:    {results['failed']}")
    print("=" * 50)
    
    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
