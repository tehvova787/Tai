"""
Multimodal Interface for Lucky Train AI Assistant

This module provides multimodal interaction capabilities for the Lucky Train AI assistant,
including voice recognition and generation, image analysis and generation, and AR functionality.
"""

import json
import logging
import os
import base64
import tempfile
from typing import Dict, List, Optional, Union, Tuple, BinaryIO
import io
import requests
import time
from dotenv import load_dotenv
import speech_recognition as sr
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import cv2
from io import BytesIO
import threading
import queue
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MultimodalInterface:
    """Multimodal interface for the Lucky Train AI assistant."""
    
    def __init__(self, config_path: str = "./config/config.json"):
        """Initialize the multimodal interface.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        
        # Initialize API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.stability_api_key = os.getenv("STABILITY_API_KEY")  # For enhanced image generation
        
        # Initialize supported voice models
        self.voice_models = self.config.get("voice_models", {
            "ru": "russian_male_1",
            "en": "english_female_1",
            "es": "spanish_male_1",
            "fr": "french_female_1",
            "de": "german_male_1"
        })
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize OpenAI client if API key is available
        if self.openai_api_key:
            import openai
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Set up processing queues for async operations
        self.speech_queue = queue.Queue()
        self.image_queue = queue.Queue()
        
        # Start processing threads
        self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self.speech_thread.start()
        
        self.image_thread = threading.Thread(target=self._process_image_queue, daemon=True)
        self.image_thread.start()
        
        logger.info("Multimodal Interface initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load the configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            The configuration as a dictionary.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            # Return default configuration
            return {}
    
    def speech_to_text(self, audio_data: BinaryIO, language: str = None) -> Dict:
        """Convert speech to text.
        
        Args:
            audio_data: The audio data as a file-like object.
            language: The language of the audio.
            
        Returns:
            A dictionary containing the recognized text and metadata.
        """
        if language is None:
            language = self.config.get("language", "ru")
        
        # Map internal language codes to speech recognition language codes
        language_mapping = {
            "ru": "ru-RU",
            "en": "en-US",
            "es": "es-ES",
            "fr": "fr-FR",
            "de": "de-DE",
            "zh": "zh-CN",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "ar": "ar-SA",
            "hi": "hi-IN"
        }
        
        recognition_language = language_mapping.get(language, "en-US")
        
        try:
            # Save the audio data to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(audio_data.read())
                temp_audio_path = temp_audio.name
            
            # Perform speech recognition
            with sr.AudioFile(temp_audio_path) as source:
                audio = self.recognizer.record(source)
                
                # Try to recognize the speech
                text = self.recognizer.recognize_google(audio, language=recognition_language)
                
                # Clean up the temporary file
                os.unlink(temp_audio_path)
                
                return {
                    "success": True,
                    "text": text,
                    "language": language
                }
        
        except sr.UnknownValueError:
            logger.warning("Speech could not be understood")
            return {
                "success": False,
                "error": "Speech could not be understood",
                "text": "",
                "language": language
            }
            
        except sr.RequestError as e:
            logger.error(f"Could not request results from speech recognition service: {e}")
            return {
                "success": False,
                "error": f"Could not request results from speech recognition service: {e}",
                "text": "",
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "language": language
            }
    
    def text_to_speech(self, text: str, language: str = None, voice_id: str = None, emotion: str = "neutral") -> Dict:
        """Convert text to speech with emotional expression.
        
        Args:
            text: The text to convert to speech.
            language: The language of the text.
            voice_id: The ID of the voice to use.
            emotion: The emotional tone (neutral, happy, sad, urgent, professional).
            
        Returns:
            A dictionary containing the audio data and metadata.
        """
        if language is None:
            language = self.config.get("language", "ru")
        
        # Use the specified voice ID or get one based on the language
        if voice_id is None:
            voice_id = self.voice_models.get(language, self.voice_models.get("en"))
        
        # Define stability and similarity boost based on emotion
        emotion_settings = {
            "neutral": {"stability": 0.5, "similarity_boost": 0.75},
            "happy": {"stability": 0.3, "similarity_boost": 0.8},
            "sad": {"stability": 0.7, "similarity_boost": 0.7},
            "urgent": {"stability": 0.2, "similarity_boost": 0.9},
            "professional": {"stability": 0.8, "similarity_boost": 0.5}
        }
        
        settings = emotion_settings.get(emotion, emotion_settings["neutral"])
        
        try:
            # Check if we have an ElevenLabs API key
            if self.elevenlabs_api_key:
                # Use ElevenLabs API for high-quality voice generation
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": self.elevenlabs_api_key
                }
                
                data = {
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": settings["stability"],
                        "similarity_boost": settings["similarity_boost"]
                    }
                }
                
                response = requests.post(url, json=data, headers=headers)
                response.raise_for_status()
                
                audio_data = response.content
                
                # Convert to base64 for easy transport
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                return {
                    "success": True,
                    "audio_base64": audio_base64,
                    "format": "mp3",
                    "language": language,
                    "voice_id": voice_id,
                    "emotion": emotion
                }
                
            elif self.openai_api_key:
                # Fall back to OpenAI's TTS if ElevenLabs is not available
                from openai import OpenAI
                
                client = OpenAI(api_key=self.openai_api_key)
                
                # Map voices to OpenAI's available voices
                openai_voices = {
                    "male": "alloy",
                    "female": "nova"
                }
                
                # Determine if the voice is male or female based on voice_id
                voice_type = "male" if "male" in voice_id else "female"
                openai_voice = openai_voices.get(voice_type, "alloy")
                
                response = client.audio.speech.create(
                    model="tts-1",
                    voice=openai_voice,
                    input=text
                )
                
                # Get the audio data
                audio_data = response.content
                
                # Convert to base64 for easy transport
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                return {
                    "success": True,
                    "audio_base64": audio_base64,
                    "format": "mp3",
                    "language": language,
                    "voice_id": openai_voice,
                    "emotion": "neutral"  # OpenAI doesn't support emotions yet
                }
                
            else:
                # No voice service available
                logger.warning("No voice service API keys available")
                return {
                    "success": False,
                    "error": "No voice service API keys available"
                }
                
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_image(self, image_data: BinaryIO) -> Dict:
        """Analyze an image and extract information from it.
        
        Args:
            image_data: The image data as a file-like object.
            
        Returns:
            A dictionary containing the analysis results.
        """
        try:
            # Check if we have an OpenAI API key for image analysis
            if not self.openai_api_key:
                return {
                    "success": False,
                    "error": "OpenAI API key not available for image analysis"
                }
            
            # Convert the image to base64
            image_data.seek(0)
            image_bytes = image_data.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Use OpenAI's Vision model to analyze the image
            from openai import OpenAI
            
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an image analysis assistant for the Lucky Train project. Analyze the image and provide relevant information about the content, especially if it relates to trains, virtual locations, NFTs, or blockchain-related items."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image? Provide a detailed analysis."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract key information from the analysis
            # In a real implementation, you would do more sophisticated processing here
            contains_train = "train" in analysis_text.lower() or "поезд" in analysis_text.lower()
            contains_nft = "nft" in analysis_text.lower() or "token" in analysis_text.lower() or "токен" in analysis_text.lower()
            contains_location = "location" in analysis_text.lower() or "place" in analysis_text.lower() or "локация" in analysis_text.lower() or "место" in analysis_text.lower()
            
            # Determine if it might be a QR code or AR marker
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(image_bytes))
            
            # Simplified check for QR-like patterns (not a real QR detector)
            # A real implementation would use a proper QR code library
            is_qr_code = False
            try:
                # Convert to greyscale
                img_grey = img.convert('L')
                # Check for high contrast pixels in a grid pattern (crude approximation)
                pixels = list(img_grey.getdata())
                width, height = img_grey.size
                
                if width > 20 and height > 20:
                    # Sample the image to check for high contrast patterns
                    contrast_score = 0
                    samples = 100
                    for i in range(samples):
                        x1, y1 = (i % 10) * width // 10, (i // 10) * height // 10
                        x2, y2 = ((i % 10) + 1) * width // 10, ((i // 10) + 1) * height // 10
                        
                        if abs(pixels[y1 * width + x1] - pixels[y2 * width + x2]) > 100:
                            contrast_score += 1
                    
                    is_qr_code = contrast_score > 50
            except Exception as e:
                logger.error(f"Error analyzing image for QR patterns: {e}")
            
            return {
                "success": True,
                "analysis": analysis_text,
                "contains_train": contains_train,
                "contains_nft": contains_nft,
                "contains_location": contains_location,
                "is_qr_code": is_qr_code
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_image(self, prompt: str, style: str = "realistic", size: str = "1024x1024") -> Dict:
        """Generate an image based on a text prompt.
        
        Args:
            prompt: The prompt describing the image to generate.
            style: The image style (realistic, cartoon, artistic, pixel_art).
            size: The image size (1024x1024, 512x512, 256x256).
            
        Returns:
            A dictionary containing the generated image data and metadata.
        """
        try:
            # Check if we have an OpenAI API key for image generation
            if not self.openai_api_key:
                return {
                    "success": False,
                    "error": "OpenAI API key not available for image generation"
                }
            
            # Enhance the prompt based on the style
            style_prompts = {
                "realistic": "Photorealistic, detailed, high resolution, sharp focus",
                "cartoon": "Cartoon style, colorful, simple shapes, clean lines",
                "artistic": "Digital art style, painterly, vibrant colors, detailed",
                "pixel_art": "Pixel art style, 8-bit, retro gaming aesthetic",
                "schematic": "Technical drawing, blueprint style, schematic, clean lines"
            }
            
            # Add Lucky Train branding to ensure consistency
            lucky_train_elements = "Include Lucky Train branding elements, trains, TON blockchain visual motifs"
            
            enhanced_prompt = f"{prompt}. {style_prompts.get(style, '')}. {lucky_train_elements}"
            
            # Use OpenAI's DALL-E model to generate the image
            from openai import OpenAI
            
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size=size,
                quality="standard",
                n=1
            )
            
            image_url = response.data[0].url
            
            # Download the image
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            
            # Convert to base64 for easy transport
            image_base64 = base64.b64encode(image_response.content).decode('utf-8')
            
            return {
                "success": True,
                "image_base64": image_base64,
                "format": "png",
                "prompt": prompt,
                "style": style,
                "size": size
            }
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_ar_preview(self, nft_data: Dict, background_image: BinaryIO = None) -> Dict:
        """Create an augmented reality preview for an NFT.
        
        Args:
            nft_data: Data about the NFT to preview.
            background_image: Optional background image (e.g., from camera).
            
        Returns:
            A dictionary containing the AR preview data and metadata.
        """
        try:
            # For a real implementation, you would use specialized AR libraries
            # This is a simplified version that creates a composite image
            
            # Generate an image of the NFT using the generate_image method
            nft_name = nft_data.get("name", "Lucky Train NFT")
            nft_description = nft_data.get("description", "A unique digital asset from the Lucky Train metaverse")
            
            prompt = f"A 3D visualization of the NFT named '{nft_name}'. {nft_description}"
            nft_image_result = self.generate_image(prompt, style="realistic")
            
            if not nft_image_result["success"]:
                return nft_image_result
            
            # If we have a background image, create a composite
            if background_image:
                try:
                    # Open the background image and NFT image
                    from PIL import Image, ImageDraw, ImageFont, ImageEnhance
                    import io
                    
                    # Get the NFT image from base64
                    nft_image_bytes = base64.b64decode(nft_image_result["image_base64"])
                    nft_img = Image.open(io.BytesIO(nft_image_bytes))
                    
                    # Get the background image
                    background_image.seek(0)
                    bg_img = Image.open(background_image)
                    
                    # Resize background if needed
                    if bg_img.size[0] < 512 or bg_img.size[1] < 512:
                        bg_img = bg_img.resize((1024, 1024))
                    
                    # Create a composite image
                    # Resize NFT image to fit nicely on the background
                    nft_width = int(bg_img.width * 0.6)
                    nft_height = int(nft_width * nft_img.height / nft_img.width)
                    nft_img = nft_img.resize((nft_width, nft_height))
                    
                    # Position the NFT image on the background
                    x_offset = (bg_img.width - nft_width) // 2
                    y_offset = (bg_img.height - nft_height) // 2
                    
                    # Add the NFT to the background with a slight transparency
                    composite = bg_img.copy()
                    nft_img_with_alpha = nft_img.convert("RGBA")
                    composite.paste(nft_img_with_alpha, (x_offset, y_offset), nft_img_with_alpha)
                    
                    # Add NFT information as text
                    draw = ImageDraw.Draw(composite)
                    
                    # Simplified font handling (in a real app, you'd have proper fonts)
                    try:
                        font = ImageFont.truetype("arial.ttf", 24)
                    except:
                        font = ImageFont.load_default()
                    
                    # Draw text with a background for visibility
                    text_y = y_offset + nft_height + 20
                    draw.rectangle([(x_offset, text_y), (x_offset + nft_width, text_y + 100)], fill=(0, 0, 0, 128))
                    draw.text((x_offset + 10, text_y + 10), nft_name, fill=(255, 255, 255), font=font)
                    draw.text((x_offset + 10, text_y + 40), f"Price: {nft_data.get('price', '100')} LTT", fill=(255, 255, 255), font=font)
                    draw.text((x_offset + 10, text_y + 70), "Tap to purchase", fill=(255, 255, 255), font=font)
                    
                    # Convert the composite to bytes and base64
                    buffer = io.BytesIO()
                    composite.save(buffer, format="PNG")
                    buffer.seek(0)
                    composite_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    
                    return {
                        "success": True,
                        "image_base64": composite_base64,
                        "format": "png",
                        "ar_type": "composite",
                        "nft_data": nft_data
                    }
                    
                except Exception as e:
                    logger.error(f"Error creating composite AR image: {e}")
                    # Fall back to just the NFT image
                    return nft_image_result
            
            # If no background, just return the NFT image
            return {
                "success": True,
                "image_base64": nft_image_result["image_base64"],
                "format": "png",
                "ar_type": "simple",
                "nft_data": nft_data
            }
            
        except Exception as e:
            logger.error(f"Error creating AR preview: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_location_preview(self, location_name: str, location_type: str = "station") -> Dict:
        """Generate a 3D preview of a metaverse location.
        
        Args:
            location_name: The name of the location.
            location_type: The type of location (station, district, landmark).
            
        Returns:
            A dictionary containing the preview data and metadata.
        """
        # For now, this is a simplified implementation that generates a 2D image
        # In a real implementation, this would generate or serve a WebGL 3D scene
        
        # Create a detailed prompt based on location type
        location_type_descriptions = {
            "station": "A railway station with platforms, waiting areas, and trains",
            "district": "An urban district with buildings, streets, and people",
            "landmark": "A distinctive landmark building or monument",
            "natural": "A natural landscape with trees, mountains, or water features",
            "entertainment": "An entertainment venue with stages, seating, and decorations"
        }
        
        location_description = location_type_descriptions.get(location_type, "A virtual location")
        
        prompt = f"A detailed 3D-style visualization of '{location_name}' in the Lucky Train metaverse. {location_description}, with trains, passengers, and futuristic TON blockchain elements. Wide angle view, detailed architecture."
        
        # Generate an image of the location
        return self.generate_image(prompt, style="realistic", size="1024x1024")
    
    def process_gesture(self, gesture_data: Dict) -> Dict:
        """Process gesture data from camera.
        
        Args:
            gesture_data: Data about the detected gesture.
            
        Returns:
            A dictionary containing the processed gesture and action to take.
        """
        # This is a placeholder for gesture recognition functionality
        # In a real implementation, this would process data from computer vision algorithms
        
        gesture_type = gesture_data.get("type", "unknown")
        confidence = gesture_data.get("confidence", 0.0)
        
        # Only process gestures with high confidence
        if confidence < 0.7:
            return {
                "success": False,
                "error": "Gesture confidence too low",
                "action": "none"
            }
        
        # Map gestures to actions
        gesture_actions = {
            "swipe_left": "previous",
            "swipe_right": "next",
            "swipe_up": "select",
            "swipe_down": "back",
            "zoom_in": "zoom_in",
            "zoom_out": "zoom_out",
            "pointing": "point_select",
            "wave": "greeting",
            "thumbs_up": "approve",
            "thumbs_down": "reject",
            "palm_open": "stop",
            "fist": "grab"
        }
        
        action = gesture_actions.get(gesture_type, "none")
        
        return {
            "success": True,
            "gesture": gesture_type,
            "confidence": confidence,
            "action": action
        }


# Example usage
if __name__ == "__main__":
    # Initialize the multimodal interface
    interface = MultimodalInterface()
    
    # Example: Generate an image of a Lucky Train location
    location_preview = interface.generate_location_preview("Central Station", "station")
    
    if location_preview["success"]:
        print("Successfully generated location preview")
        
        # Save the image to a file for testing
        try:
            image_data = base64.b64decode(location_preview["image_base64"])
            with open("central_station_preview.png", "wb") as f:
                f.write(image_data)
            print("Saved preview to central_station_preview.png")
        except Exception as e:
            print(f"Error saving preview: {e}")
    else:
        print(f"Error generating location preview: {location_preview.get('error')}")
    
    # Example: Generate speech from text
    speech_result = interface.text_to_speech("Добро пожаловать в метавселенную Lucky Train!", "ru", emotion="happy")
    
    if speech_result["success"]:
        print("Successfully generated speech")
        
        # Save the audio to a file for testing
        try:
            audio_data = base64.b64decode(speech_result["audio_base64"])
            with open("welcome_message.mp3", "wb") as f:
                f.write(audio_data)
            print("Saved audio to welcome_message.mp3")
        except Exception as e:
            print(f"Error saving audio: {e}")
    else:
        print(f"Error generating speech: {speech_result.get('error')}") 