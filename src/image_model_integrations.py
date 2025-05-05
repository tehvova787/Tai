"""
Image Model Integrations for Lucky Train

This module provides integrations for various image and vision AI models:
- DALL-E 2 and DALL-E 3
- Stable Diffusion XL and 2.1
- Vision models (ResNet, CLIP, GPT-4V)
"""

import os
import logging
import base64
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import io
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseImageModelInterface(ABC):
    """Base class for all image model interfaces."""
    
    def __init__(self, config: Dict = None):
        """Initialize the model interface.
        
        Args:
            config: Configuration for the model
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        logger.info(f"Initializing {self.name}")
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate an image based on the prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Response data including image information
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get information about the model.
        
        Returns:
            Model information
        """
        pass

class DALLEInterface(BaseImageModelInterface):
    """Interface for OpenAI's DALL-E 2 and DALL-E 3 image generation models."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY") or self.config.get("api_key")
        self.model = self.config.get("model", "dall-e-3")
        self.size = self.config.get("size", "1024x1024")
        self.quality = self.config.get("quality", "standard")
        
        if not self.api_key:
            logger.warning("OpenAI API key not set - functionality will be limited")
            self.client = None
        else:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"DALL-E client initialized with model: {self.model}")
            except ImportError:
                logger.error("openai package not installed")
                self.client = None
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate an image using DALL-E.
        
        Args:
            prompt: Text prompt for image generation
            
        Returns:
            Response data including image URL
        """
        if not self.client:
            return {"error": "OpenAI client not initialized", "image_url": None}
        
        try:
            # Override defaults with kwargs if provided
            model = kwargs.get("model", self.model)
            size = kwargs.get("size", self.size)
            quality = kwargs.get("quality", self.quality)
            n = kwargs.get("n", 1)  # Number of images
            
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n
            )
            
            # Get all image URLs
            image_urls = [item.url for item in response.data]
            
            result = {
                "image_urls": image_urls,
                "model": model,
                "provider": "OpenAI",
                "prompt": prompt
            }
            
            # If save_to_file is provided, save the image
            save_to = kwargs.get("save_to")
            if save_to and image_urls:
                self._save_image_from_url(image_urls[0], save_to)
                result["saved_to"] = save_to
            
            return result
        except Exception as e:
            logger.error(f"Error in DALL-E image generation: {e}")
            return {"error": str(e), "image_url": None}
    
    def _save_image_from_url(self, url: str, path: str) -> bool:
        """Download and save image from URL.
        
        Args:
            url: Image URL
            path: Path to save the image
            
        Returns:
            Success status
        """
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                logger.error(f"Failed to download image: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the DALL-E model.
        
        Returns:
            Model information
        """
        models_info = {
            "dall-e-2": {
                "description": "DALL-E 2 image generation model",
                "sizes": ["256x256", "512x512", "1024x1024"],
                "max_prompt_length": 1000
            },
            "dall-e-3": {
                "description": "DALL-E 3 advanced image generation model",
                "sizes": ["1024x1024", "1792x1024", "1024x1792"],
                "qualities": ["standard", "hd"],
                "max_prompt_length": 4000
            }
        }
        
        return models_info.get(self.model, {"description": "Unknown DALL-E model"})

class StableDiffusionInterface(BaseImageModelInterface):
    """Interface for Stable Diffusion XL and Stable Diffusion 2.1 models."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model_id = self.config.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0")
        self.local_model_path = self.config.get("local_model_path", None)
        
        # Check if we should use local or HuggingFace
        self.use_local = self.local_model_path is not None and os.path.exists(self.local_model_path)
        
        try:
            import torch
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self.use_local:
                logger.info(f"Loading local Stable Diffusion model from {self.local_model_path}")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.local_model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                logger.info(f"Loading Stable Diffusion model {self.model_id} from HuggingFace")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            # Use DPM-Solver++ scheduler for faster inference
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.to(self.device)
            
            # Optional: enable memory optimizations
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                
            self.model_loaded = True
        except ImportError as e:
            logger.error(f"Missing dependency for Stable Diffusion: {e}")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Error loading Stable Diffusion model: {e}")
            self.model_loaded = False
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate an image using Stable Diffusion.
        
        Args:
            prompt: Text prompt for image generation
            
        Returns:
            Response data including image or path to saved image
        """
        if not self.model_loaded:
            return {"error": "Stable Diffusion model not loaded", "image": None}
        
        try:
            # Get parameters from kwargs or use defaults
            negative_prompt = kwargs.get("negative_prompt", "blurry, bad quality, distorted, low resolution")
            num_inference_steps = kwargs.get("num_inference_steps", 30)
            guidance_scale = kwargs.get("guidance_scale", 7.5)
            num_images = kwargs.get("num_images", 1)
            
            # Generate images
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images
            )
            
            # Process the results
            images = result.images
            nsfw_content_detected = result.nsfw_content_detected if hasattr(result, "nsfw_content_detected") else [False] * len(images)
            
            # Format the response
            response = {
                "images": [],
                "model": self.model_id if not self.use_local else os.path.basename(self.local_model_path),
                "provider": "Stable Diffusion",
                "prompt": prompt
            }
            
            # Save images if requested
            save_dir = kwargs.get("save_dir")
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            for i, (image, nsfw) in enumerate(zip(images, nsfw_content_detected)):
                image_info = {"nsfw_content_detected": nsfw}
                
                if save_dir:
                    # Generate filename
                    filename = f"sd_gen_{i}_{int(time.time())}.png"
                    filepath = os.path.join(save_dir, filename)
                    image.save(filepath)
                    image_info["saved_to"] = filepath
                
                # Convert image to base64 string
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                image_info["base64"] = img_str
                
                response["images"].append(image_info)
            
            return response
        except Exception as e:
            logger.error(f"Error in Stable Diffusion image generation: {e}")
            return {"error": str(e), "image": None}
    
    def get_model_info(self) -> Dict:
        """Get information about the Stable Diffusion model.
        
        Returns:
            Model information
        """
        models_info = {
            "stabilityai/stable-diffusion-xl-base-1.0": {
                "description": "Stable Diffusion XL base model with 2.6B parameters",
                "resolution": "1024x1024",
                "paper": "https://arxiv.org/abs/2307.01952"
            },
            "stabilityai/stable-diffusion-2-1": {
                "description": "Stable Diffusion 2.1 base model",
                "resolution": "768x768",
                "paper": "https://arxiv.org/abs/2112.10752"
            },
            "runwayml/stable-diffusion-v1-5": {
                "description": "Stable Diffusion v1.5 model",
                "resolution": "512x512",
                "paper": "https://arxiv.org/abs/2112.10752"
            },
            "CompVis/stable-diffusion-v1-4": {
                "description": "Original Stable Diffusion model",
                "resolution": "512x512",
                "paper": "https://arxiv.org/abs/2112.10752"
            }
        }
        
        if self.use_local:
            return {
                "description": f"Local Stable Diffusion model at {self.local_model_path}",
                "resolution": "Varies based on model"
            }
        
        return models_info.get(self.model_id, {"description": "Custom Stable Diffusion model"})

class ResNetVisionInterface(BaseImageModelInterface):
    """Interface for ResNet vision models (ResNet-50, ResNet-101, ResNet-152)."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "resnet50")
        self.pretrained = self.config.get("pretrained", True)
        
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            # Load model
            if self.model_name == "resnet50":
                self.model = models.resnet50(pretrained=self.pretrained)
            elif self.model_name == "resnet101":
                self.model = models.resnet101(pretrained=self.pretrained)
            elif self.model_name == "resnet152":
                self.model = models.resnet152(pretrained=self.pretrained)
            else:
                raise ValueError(f"Unknown ResNet model: {self.model_name}")
            
            # Set to evaluation mode
            self.model.eval()
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            # Image preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Load ImageNet class labels
            with open("imagenet_classes.txt", "r") as f:
                self.categories = [line.strip() for line in f.readlines()]
            
            self.model_loaded = True
            logger.info(f"ResNet model {self.model_name} loaded successfully")
        except ImportError as e:
            logger.error(f"Missing dependency for ResNet: {e}")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Error loading ResNet model: {e}")
            self.model_loaded = False
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Analyze an image with ResNet (note: prompt is ignored, image_path is used).
        
        Args:
            prompt: Ignored, kept for interface consistency
            image_path: Path to the image to analyze
            
        Returns:
            Classification results
        """
        # In this interface, we use the prompt parameter for API consistency,
        # but actually look for image_path in kwargs
        image_path = kwargs.get("image_path")
        if not image_path:
            return {"error": "Image path not provided", "predictions": None}
        
        if not self.model_loaded:
            return {"error": "ResNet model not loaded", "predictions": None}
        
        try:
            import torch
            
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get top 5 predictions
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            # Format results
            predictions = []
            for i, (prob, catid) in enumerate(zip(top5_prob, top5_catid)):
                predictions.append({
                    "category": self.categories[catid],
                    "probability": float(prob)
                })
            
            return {
                "predictions": predictions,
                "model": self.model_name,
                "provider": "ResNet",
                "image_path": image_path
            }
        except Exception as e:
            logger.error(f"Error in ResNet image analysis: {e}")
            return {"error": str(e), "predictions": None}
    
    def get_model_info(self) -> Dict:
        """Get information about the ResNet model.
        
        Returns:
            Model information
        """
        models_info = {
            "resnet50": {
                "description": "ResNet-50 with 50 layers",
                "parameters": "25.6M",
                "paper": "https://arxiv.org/abs/1512.03385",
                "classes": 1000
            },
            "resnet101": {
                "description": "ResNet-101 with 101 layers",
                "parameters": "44.5M",
                "paper": "https://arxiv.org/abs/1512.03385",
                "classes": 1000
            },
            "resnet152": {
                "description": "ResNet-152 with 152 layers",
                "parameters": "60.2M",
                "paper": "https://arxiv.org/abs/1512.03385",
                "classes": 1000
            }
        }
        
        return models_info.get(self.model_name, {"description": "Unknown ResNet model"})

class CLIPInterface(BaseImageModelInterface):
    """Interface for OpenAI's CLIP (Contrastive Language-Image Pre-training) model."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "ViT-B/32")
        
        try:
            import torch
            import clip
            
            # Load CLIP model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            
            self.model_loaded = True
            logger.info(f"CLIP model {self.model_name} loaded successfully")
        except ImportError as e:
            logger.error(f"Missing dependency for CLIP: {e}")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            self.model_loaded = False
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Match image with text prompts using CLIP.
        
        Args:
            prompt: Text prompt or comma-separated list of prompts
            image_path: Path to the image
            
        Returns:
            Matching scores between image and prompts
        """
        image_path = kwargs.get("image_path")
        if not image_path:
            return {"error": "Image path not provided", "scores": None}
        
        if not self.model_loaded:
            return {"error": "CLIP model not loaded", "scores": None}
        
        try:
            import torch
            import clip
            
            # Process text prompts
            prompts = [p.strip() for p in prompt.split(",")]
            
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Process text
            text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            
            # Get features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_inputs)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Format results
            scores = []
            values, indices = similarity[0].topk(len(prompts))
            for value, index in zip(values, indices):
                scores.append({
                    "prompt": prompts[index],
                    "score": float(value)
                })
            
            return {
                "scores": scores,
                "model": self.model_name,
                "provider": "CLIP",
                "image_path": image_path
            }
        except Exception as e:
            logger.error(f"Error in CLIP image-text matching: {e}")
            return {"error": str(e), "scores": None}
    
    def get_model_info(self) -> Dict:
        """Get information about the CLIP model.
        
        Returns:
            Model information
        """
        models_info = {
            "ViT-B/32": {
                "description": "CLIP with ViT-B/32 backbone",
                "paper": "https://arxiv.org/abs/2103.00020",
                "input_resolution": 224
            },
            "ViT-B/16": {
                "description": "CLIP with ViT-B/16 backbone - higher resolution",
                "paper": "https://arxiv.org/abs/2103.00020",
                "input_resolution": 224
            },
            "ViT-L/14": {
                "description": "CLIP with ViT-L/14 backbone - largest model",
                "paper": "https://arxiv.org/abs/2103.00020",
                "input_resolution": 224
            }
        }
        
        return models_info.get(self.model_name, {"description": "Unknown CLIP model"})

class GPT4VisionInterface(BaseImageModelInterface):
    """Interface for OpenAI's GPT-4V (Vision) model."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY") or self.config.get("api_key")
        self.model = self.config.get("model", "gpt-4-vision-preview")
        
        if not self.api_key:
            logger.warning("OpenAI API key not set - functionality will be limited")
            self.client = None
        else:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"GPT-4 Vision client initialized with model: {self.model}")
            except ImportError:
                logger.error("openai package not installed")
                self.client = None
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate a response based on an image and prompt using GPT-4V.
        
        Args:
            prompt: Text prompt asking about the image
            image_path: Path to the image
            
        Returns:
            Response data
        """
        image_path = kwargs.get("image_path")
        if not image_path:
            return {"error": "Image path not provided", "analysis": None}
        
        if not self.client:
            return {"error": "OpenAI client not initialized", "analysis": None}
        
        try:
            # Encode the image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Max tokens to generate
            max_tokens = kwargs.get("max_tokens", 300)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            
            return {
                "analysis": response.choices[0].message.content,
                "model": self.model,
                "provider": "OpenAI",
                "prompt": prompt,
                "image_path": image_path
            }
        except Exception as e:
            logger.error(f"Error in GPT-4V image analysis: {e}")
            return {"error": str(e), "analysis": None}
    
    def get_model_info(self) -> Dict:
        """Get information about the GPT-4V model.
        
        Returns:
            Model information
        """
        return {
            "description": "GPT-4 with vision capabilities",
            "capabilities": [
                "Image understanding",
                "Visual question answering",
                "Image captioning",
                "Scene analysis",
                "Object recognition"
            ],
            "max_images": 20,
            "max_tokens": 4096,
            "input_modalities": ["text", "images"]
        }

# Utility function to create image model interfaces
def create_image_model_interface(model_type: str, config: Dict = None) -> BaseImageModelInterface:
    """Create and return an image model interface based on the model type.
    
    Args:
        model_type: Type of image model interface to create
        config: Configuration for the model
        
    Returns:
        Image model interface instance
    """
    model_interfaces = {
        "dalle": DALLEInterface,
        "stable-diffusion": StableDiffusionInterface,
        "resnet": ResNetVisionInterface,
        "clip": CLIPInterface,
        "gpt4-vision": GPT4VisionInterface
    }
    
    interface_class = model_interfaces.get(model_type.lower())
    
    if interface_class:
        return interface_class(config)
    else:
        logger.error(f"Unknown image model type: {model_type}")
        raise ValueError(f"Unknown image model type: {model_type}") 