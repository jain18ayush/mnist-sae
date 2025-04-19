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
# Load environment variables
load_dotenv()

# ======= DATA HANDLING MODULE ========
class MnistImageExtractor:
    def __init__(self):
        # Initialize and load the datasets
        self.train_dataset = datasets.MNIST(root='./data', train=True, download=True)
        self.test_dataset = datasets.MNIST(root='./data', train=False, download=True)
        self.combined_dataset = ConcatDataset([self.train_dataset, self.test_dataset])
    
    def extract_images(self, indices):
        """Extract images at the specified indices from the combined dataset"""
        extracted_images = []
        
        for idx in indices:
            if idx >= len(self.combined_dataset):
                print(f"Index {idx} is out of range. Max index is {len(self.combined_dataset)-1}")
                continue
                
            img, label = self.combined_dataset[idx]
            
            # Convert to RGB (MNIST is grayscale)
            img_rgb = Image.new("RGB", img.size)
            img_rgb.paste(img)
            
            extracted_images.append({
                "index": idx,
                "label": label,
                "image": img_rgb
            })
        
        return extracted_images
    
    def visualize_images(self, extracted_images):
        """Display the extracted images"""
        for img_data in extracted_images:
            plt.figure(figsize=(2, 2))
            plt.imshow(img_data["image"])
            plt.title(f"Index: {img_data['index']}, Label: {img_data['label']}")
            plt.axis('off')
            plt.show()


# ======= IMAGE ENCODING MODULE ========
class ImageEncoder:
    @staticmethod
    def encode_to_base64(pil_image):
        """Convert a PIL image to base64 string"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
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
class OpenRouterClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key is required. Set it directly or via OPENROUTER_API_KEY environment variable.", self.api_key)
    
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
                    "url": f"data:image/jpeg;base64,{img_data['base64']}"
                }
            })
            
        return content
    
    def send_request(self, content, model="meta-llama/llama-4-scout"):
        """Send the request to OpenRouter API"""
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
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
            }
        )
        return response.json()


# ======= MAIN SCRIPT ========
path = '/Volumes/Ayush_Drive/mnist'
prefix = path if os.path.exists(path) else ''

neuron_set = set()
results = []
cache_file = os.path.join(path, "metrics", "neuron_results.jsonl")

extractor = MnistImageExtractor()
encoder = ImageEncoder()
client = OpenRouterClient(api_key=os.getenv('OPENROUTER_API_KEY'))

def extract_last_bracketed_line(response: str) -> str:
    lines = [line.strip() for line in response.strip().splitlines() if line.strip().startswith("[") and line.strip().endswith("]")]
    return lines[-1][1:-1].strip() if lines else None

def process_neuron(indices, activations): 
    prompt = f"""
    You are a meticulous AI researcher conducting an important investigation into patterns found in MNIST models. Your task is to analyze image examples and provide an interpretation that thoroughly encapsulates possible patterns a specific neuron is detecting. The dataset is simply a set of MNIST images (so the numbers from 0 to 9).
    Guidelines:
    You will be given a set of image examples where a specific neuron shows high activation. The activation strength for each image is listed in parentheses.
    Try to produce a concise final description. Describe the visual features that are common across these high-activating images, and what patterns you found.
    Focus on identifying specific visual elements, shapes, textures, objects, colors, spatial arrangements, or semantic concepts that might trigger this neuron.
    If the examples are uninformative or seem random, you don't need to force an interpretation. Simply note that no clear pattern is evident by putting "None" in the interpretation.
    Keep your interpretations short and concise.
    Do not make lists of possible interpretations.
    The last line of your response must be the formatted interpretation, using [interpretation]:

    The activations are " [{activations}] " and the images are attached in order. 
    """

    extracted_images = extractor.extract_images(indices)

    # Visualize images (optional)
    # extractor.visualize_images(extracted_images)

    # Prepare images for API
    api_ready_images = encoder.prepare_for_api(extracted_images)
    content = client.create_content_array(prompt, api_ready_images)

    response = client.send_request(content, model="meta-llama/llama-4-scout")
    return response 



prompt_tokens = 0 
completion_tokens = 0 

feature_obj = json.load(open(os.path.join(path, 'feature_evolution_output', 'all_features_indices.json')))

# Load existing cache if it exists
if os.path.exists(cache_file):
    with open(cache_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            neuron_set.add((entry['feature'], entry['depth']))

# Main loop
for feature in tqdm(feature_obj, desc="Processing features"):
    for depth in tqdm(feature_obj[feature]['depth_data'], desc=f"Processing depths for feature {feature}", leave=False):
        if depth == 1: 
            continue
        key = (feature, depth)
        # if key in neuron_set:
        #     continue

        try:
            response = process_neuron(
                feature_obj[feature]['depth_data'][depth]['indices'],
                feature_obj[feature]['depth_data'][depth]['activations']
            )
            content = response['choices'][0]['message']['content']
            interpretation = extract_last_bracketed_line(content)

            result = {
                "feature": feature_obj[feature]['depth_data'][depth]['feature'],
                "depth": depth,
                "response": interpretation,
                "text": content
            }

            # Write result immediately to cache
            with open(cache_file, 'a') as f:
                f.write(json.dumps(result) + "\n")

            # Track for memory (optional)
            results.append(result)
            neuron_set.add(key)

            prompt_tokens += response['usage']['prompt_tokens']
            completion_tokens += response['usage']['completion_tokens']

        except Exception as e:
            print(f"Error processing feature {feature}, depth {depth}: {e}")
    break 
# Print summary
print(f"Total Prompt Tokens: {prompt_tokens}")
print(f"Total Completion Tokens: {completion_tokens}")
# Save results to a df
import pandas as pd
df = pd.DataFrame(results)
df.to_parquet(os.path.join(path, "metrics", "neuron_results.parquet"), index=False)
