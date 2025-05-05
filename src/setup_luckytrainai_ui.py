#!/usr/bin/env python3
"""
Setup script for LuckyTrainAI UI

This script prepares the LuckyTrainAI UI by:
1. Creating necessary directories
2. Converting and copying the logo file
3. Setting up the loading animation from the video
4. Generating additional UI assets
"""

import os
import sys
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

def ensure_directory(path):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def main():
    # Define paths
    base_dir = Path(__file__).parent
    project_root = base_dir.parent
    
    static_dir = base_dir / "web" / "static"
    img_dir = static_dir / "images"
    media_dir = static_dir / "media"
    
    # Create necessary directories
    ensure_directory(static_dir)
    ensure_directory(img_dir)
    ensure_directory(media_dir)
    
    # Process logo
    logo_path = project_root / "logo.jpg"
    if logo_path.exists():
        # Create a PNG version with transparency
        try:
            logo = Image.open(logo_path)
            # Resize if needed
            logo = logo.resize((200, 200))
            # Save as PNG with transparency
            logo_output = img_dir / "luckytrainai-logo.png"
            logo.save(logo_output, format="PNG")
            print(f"Processed logo: {logo_output}")
        except Exception as e:
            print(f"Error processing logo: {e}")
    else:
        print(f"Logo file not found: {logo_path}")
        # Create a placeholder logo
        create_placeholder_logo(img_dir / "luckytrainai-logo.png")
    
    # Process loading animation
    video_path = project_root / "019a9e99-d5cb-4e75-9888-c02dc720d123.mp4"
    if video_path.exists():
        # Copy the video to the media directory
        shutil.copy(video_path, media_dir / "loading-animation.mp4")
        print(f"Copied loading animation: {media_dir / 'loading-animation.mp4'}")
        
        # Also create a smaller version for the response loading
        shutil.copy(video_path, media_dir / "response-loading.mp4")
        print(f"Copied response loading animation: {media_dir / 'response-loading.mp4'}")
    else:
        print(f"Loading animation file not found: {video_path}")
        # Create a placeholder animation
        create_placeholder_animation(media_dir / "loading-animation.mp4")
        create_placeholder_animation(media_dir / "response-loading.mp4")
    
    # Create additional UI assets
    create_assistant_avatar(img_dir / "assistant-avatar.png")
    create_user_avatar(img_dir / "user-avatar.png")
    create_particles_image(img_dir / "particles.png")
    create_favicon(img_dir / "favicon.png")
    
    print("\nLuckyTrainAI UI setup complete!")
    print("You can now run the web interface with the new UI.")

def create_placeholder_logo(output_path):
    """Create a placeholder logo if the original isn't found."""
    try:
        # Create a 200x200 image with text
        img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a background circle
        draw.ellipse((10, 10, 190, 190), fill=(0, 255, 229, 100))
        
        # Draw letter L in the center - we can't use font_size in PIL's draw.text
        # Draw a simple L shape manually
        draw.rectangle((70, 70, 90, 130), fill=(255, 255, 255, 255))
        draw.rectangle((70, 130, 130, 150), fill=(255, 255, 255, 255))
        
        img.save(output_path, format="PNG")
        print(f"Created placeholder logo: {output_path}")
    except Exception as e:
        print(f"Error creating placeholder logo: {e}")

def create_placeholder_animation(output_path):
    """Create a simple placeholder video animation."""
    try:
        # We'll just create a text file indicating it's a placeholder
        with open(f"{output_path}.txt", "w") as f:
            f.write("Placeholder for loading animation. Please add a real video file.")
        print(f"Created placeholder note for animation: {output_path}.txt")
    except Exception as e:
        print(f"Error creating placeholder animation note: {e}")

def create_assistant_avatar(output_path):
    """Create an avatar for the AI assistant."""
    try:
        # Create a 100x100 avatar
        img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw circular background
        draw.ellipse((0, 0, 100, 100), fill=(0, 255, 229, 100))
        
        # Draw a robot-like face
        # Draw eyes
        draw.ellipse((25, 30, 40, 45), fill=(255, 255, 255, 255))
        draw.ellipse((60, 30, 75, 45), fill=(255, 255, 255, 255))
        
        # Draw mouth
        draw.rectangle((30, 65, 70, 75), fill=(255, 255, 255, 200))
        
        img.save(output_path, format="PNG")
        print(f"Created assistant avatar: {output_path}")
    except Exception as e:
        print(f"Error creating assistant avatar: {e}")

def create_user_avatar(output_path):
    """Create a default user avatar."""
    try:
        # Create a 100x100 avatar
        img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw circular background
        draw.ellipse((0, 0, 100, 100), fill=(153, 51, 255, 100))
        
        # Draw a simple user silhouette
        # Draw head
        draw.ellipse((35, 20, 65, 50), fill=(255, 255, 255, 220))
        
        # Draw body
        draw.ellipse((25, 60, 75, 110), fill=(255, 255, 255, 220))
        
        img.save(output_path, format="PNG")
        print(f"Created user avatar: {output_path}")
    except Exception as e:
        print(f"Error creating user avatar: {e}")

def create_particles_image(output_path):
    """Create a particles background image."""
    try:
        # Create a 500x500 transparent image with dots
        img = Image.new('RGBA', (500, 500), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw random particles
        colors = [(0, 255, 229, 50), (153, 51, 255, 50), (255, 0, 51, 50), (0, 255, 102, 50)]
        
        import random
        for _ in range(200):
            x = random.randint(0, 500)
            y = random.randint(0, 500)
            size = random.randint(1, 3)
            color = random.choice(colors)
            
            draw.ellipse((x, y, x+size, y+size), fill=color)
        
        img.save(output_path, format="PNG")
        print(f"Created particles image: {output_path}")
    except Exception as e:
        print(f"Error creating particles image: {e}")

def create_favicon(output_path):
    """Create a favicon for the site."""
    try:
        # Create a 32x32 favicon
        img = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw circular background
        draw.ellipse((0, 0, 32, 32), fill=(0, 255, 229, 200))
        
        # Draw letter L - manually since we can't use font_size
        draw.rectangle((10, 8, 14, 22), fill=(255, 255, 255, 255))
        draw.rectangle((10, 22, 22, 26), fill=(255, 255, 255, 255))
        
        img.save(output_path, format="PNG")
        print(f"Created favicon: {output_path}")
    except Exception as e:
        print(f"Error creating favicon: {e}")

if __name__ == "__main__":
    main() 