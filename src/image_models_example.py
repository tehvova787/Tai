"""
Example Script for Image Models Integration

This script demonstrates how to use the image model integrations
for Lucky Train.
"""

import os
import logging
from dotenv import load_dotenv
import time
from datetime import datetime
import argparse

# Import our image model integrations
from image_model_integrations import create_image_model_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_dalle():
    """Test DALL-E image generation."""
    
    logger.info("Testing DALL-E Image Generation")
    logger.info("-" * 50)
    
    # Initialize DALL-E
    dalle_config = {"model": "dall-e-3", "size": "1024x1024"}
    dalle = create_image_model_interface("dalle", dalle_config)
    
    if os.getenv("OPENAI_API_KEY"):
        # Create output directory
        os.makedirs("output", exist_ok=True)
        save_path = f"output/dalle_gen_{int(time.time())}.png"
        
        # Generate image
        logger.info("Generating image with DALL-E 3...")
        prompt = "A futuristic cryptocurrency trading interface with holographic displays, in a high-tech office setting."
        
        response = dalle.generate(
            prompt=prompt, 
            save_to=save_path
        )
        
        if "error" in response:
            logger.error(f"DALL-E generation failed: {response['error']}")
        else:
            logger.info(f"DALL-E image generated successfully")
            logger.info(f"Image URL: {response.get('image_urls', ['No URL'])[0]}")
            logger.info(f"Image saved to: {response.get('saved_to', 'Not saved')}")
    else:
        logger.warning("Skipping DALL-E test - API key not set")
    
    logger.info("-" * 50)

def test_stable_diffusion():
    """Test Stable Diffusion image generation."""
    
    logger.info("Testing Stable Diffusion Image Generation")
    logger.info("-" * 50)
    
    # Check if local model exists or use HuggingFace
    local_model_path = os.getenv("STABLE_DIFFUSION_MODEL_PATH")
    if local_model_path and os.path.exists(local_model_path):
        sd_config = {"local_model_path": local_model_path}
    else:
        sd_config = {"model_id": "stabilityai/stable-diffusion-2-1"}
    
    try:
        # Initialize Stable Diffusion
        sd = create_image_model_interface("stable-diffusion", sd_config)
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Generate image
        logger.info("Generating image with Stable Diffusion...")
        prompt = "A digital art representation of blockchain technology, with connected nodes forming a network in vibrant colors."
        
        response = sd.generate(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, ugly",
            num_inference_steps=25,
            guidance_scale=7.5,
            save_dir="output"
        )
        
        if "error" in response:
            logger.error(f"Stable Diffusion generation failed: {response['error']}")
        else:
            logger.info(f"Stable Diffusion image generated successfully")
            if response.get("images"):
                for i, img_info in enumerate(response["images"]):
                    logger.info(f"Image {i+1} saved to: {img_info.get('saved_to', 'Not saved')}")
    except Exception as e:
        logger.error(f"Error testing Stable Diffusion: {e}")
    
    logger.info("-" * 50)

def test_vision_models(image_path):
    """Test vision models for image analysis.
    
    Args:
        image_path: Path to the image for analysis
    """
    if not image_path or not os.path.exists(image_path):
        logger.error(f"Image not found at path: {image_path}")
        return
    
    logger.info("Testing Vision Models for Image Analysis")
    logger.info(f"Using image: {image_path}")
    logger.info("-" * 50)
    
    # Test ResNet
    try:
        resnet_config = {"model_name": "resnet50"}
        resnet = create_image_model_interface("resnet", resnet_config)
        
        logger.info("Analyzing image with ResNet-50...")
        response = resnet.generate("", image_path=image_path)
        
        if "error" in response:
            logger.error(f"ResNet analysis failed: {response['error']}")
        else:
            logger.info(f"ResNet-50 analysis completed successfully")
            if response.get("predictions"):
                logger.info("Top 5 predictions:")
                for i, pred in enumerate(response["predictions"]):
                    logger.info(f"  {i+1}. {pred['category']} ({pred['probability']:.4f})")
    except Exception as e:
        logger.error(f"Error testing ResNet: {e}")
    
    logger.info("-" * 50)
    
    # Test CLIP
    try:
        clip_config = {"model_name": "ViT-B/32"}
        clip = create_image_model_interface("clip", clip_config)
        
        logger.info("Analyzing image with CLIP...")
        prompts = "cryptocurrency, blockchain, finance, technology, trading, office, computer"
        response = clip.generate(prompts, image_path=image_path)
        
        if "error" in response:
            logger.error(f"CLIP analysis failed: {response['error']}")
        else:
            logger.info(f"CLIP analysis completed successfully")
            if response.get("scores"):
                logger.info("Text-image matching scores:")
                for i, score in enumerate(response["scores"]):
                    logger.info(f"  {i+1}. {score['prompt']} ({score['score']:.4f})")
    except Exception as e:
        logger.error(f"Error testing CLIP: {e}")
    
    logger.info("-" * 50)
    
    # Test GPT-4 Vision
    if os.getenv("OPENAI_API_KEY"):
        try:
            gpt4v_config = {"model": "gpt-4-vision-preview"}
            gpt4v = create_image_model_interface("gpt4-vision", gpt4v_config)
            
            logger.info("Analyzing image with GPT-4 Vision...")
            prompt = "Describe this image in detail. What does it show and what might it be related to?"
            response = gpt4v.generate(prompt, image_path=image_path)
            
            if "error" in response:
                logger.error(f"GPT-4V analysis failed: {response['error']}")
            else:
                logger.info(f"GPT-4V analysis completed successfully")
                logger.info(f"Analysis: {response.get('analysis', 'No analysis')}")
        except Exception as e:
            logger.error(f"Error testing GPT-4 Vision: {e}")
    else:
        logger.warning("Skipping GPT-4 Vision test - API key not set")
    
    logger.info("-" * 50)

def main():
    """Main function to run the examples."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test image model integrations')
    parser.add_argument('--image', type=str, help='Path to an image for vision model testing')
    parser.add_argument('--test-all', action='store_true', help='Run all tests')
    parser.add_argument('--test-dalle', action='store_true', help='Test DALL-E')
    parser.add_argument('--test-sd', action='store_true', help='Test Stable Diffusion')
    parser.add_argument('--test-vision', action='store_true', help='Test vision models')
    
    args = parser.parse_args()
    
    logger.info("Starting Image Models Integration Example")
    logger.info("=" * 50)
    
    # Check for environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set - DALL-E and GPT-4V tests will be limited")
    
    # Run tests based on arguments
    if args.test_all or args.test_dalle:
        test_dalle()
    
    if args.test_all or args.test_sd:
        test_stable_diffusion()
    
    if (args.test_all or args.test_vision) and args.image:
        test_vision_models(args.image)
    elif args.test_all or args.test_vision:
        logger.error("No image path provided for vision model testing. Use --image parameter.")
    
    # If no specific test selected, run DALL-E and SD
    if not (args.test_all or args.test_dalle or args.test_sd or args.test_vision):
        test_dalle()
        test_stable_diffusion()
    
    logger.info("=" * 50)
    logger.info("Example completed")

if __name__ == "__main__":
    main() 