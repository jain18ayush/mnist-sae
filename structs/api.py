import torch
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import requests
import base64
import io
from PIL import Image
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json 
import numpy as np
import time

# Load environment variables
load_dotenv()

# ======= DATA HANDLING MODULE ========
class ImageExtractor:
    def __init__(self, dataset):
        # Initialize and load the datasets
        # self.train_dataset = datasets.MNIST(root='./data', train=True, download=True)
        # self.test_dataset = datasets.MNIST(root='./data', train=False, download=True)
        self.combined_dataset = dataset
    
    def extract_images(self, indices):
        """Extract images at the specified indices from the combined dataset"""
        extracted_images = []
        
        for idx in indices:
            if idx >= len(self.combined_dataset):
                print(f"Index {idx} is out of range. Max index is {len(self.combined_dataset)-1}")
                continue
                
            img, label = self.combined_dataset[idx]
            
            # Keep as grayscale - no need to convert to RGB since we'll handle this during encoding
            extracted_images.append({
                "index": idx,
                "label": label,
                "image": img
            })
        
        return extracted_images
    
    def visualize_images(self, extracted_images):
        """Display the extracted images"""
        for img_data in extracted_images:
            plt.figure(figsize=(2, 2))
            plt.imshow(img_data["image"], cmap='gray')  # Use grayscale colormap
            plt.title(f"Index: {img_data['index']}, Label: {img_data['label']}")
            plt.axis('off')
            plt.show()


# ======= IMAGE ENCODING MODULE ========
class ImageEncoder:
    @staticmethod
    def encode_to_base64(pil_image):
        """Convert a PIL image to base64 string"""
        # Convert to RGB for API compatibility
        if pil_image.mode == 'L':  # If it's grayscale
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            pil_image = rgb_image
            
        buffer = io.BytesIO()
        # Use PNG format instead of JPEG for better quality with simple images
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    @staticmethod
    def prepare_for_api(extracted_images):
        """Prepare images for API by encoding them to base64"""
        api_ready_images = []
        
        for img_data in extracted_images:
            base64_image = ImageEncoder.encode_to_base64(img_data["image"])
            api_ready_images.append({
                "index": img_data["index"],
                "label": img_data["label"],
                "base64": base64_image
            })
            
        return api_ready_images


# ======= API CLIENT MODULE ========
class GroqClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.getenv('GROQ_API_KEY')
            if not self.api_key:
                raise ValueError("API key is required. Set it directly or via GROQ_API_KEY environment variable.")
        self.api_base = "https://api.groq.com/openai/v1/chat/completions"
    
    def create_content_array(self, prompt, api_ready_images):
        """Create the content array for the API request"""
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        # Add images to content array
        for img_data in api_ready_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_data['base64']}"
                }
            })
            
        return content
    
    def send_request(self, content, model="llama-3.1-70b-versatile"):
        """Send the request to Groq API"""
        try:
            response = requests.post(
                self.api_base,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                },
                timeout=60  # Add timeout to prevent hanging requests
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            print(f"Response Text: {response.json()}")
            return {"error": str(e)}
