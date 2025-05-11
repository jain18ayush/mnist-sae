import sys
import os 
import pandas as pd
import json 
from tqdm import tqdm

sys.path.append('../../')

import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
 
from structs.api import ImageEncoder, GroqClient, ImageExtractor

path = '/Volumes/Ayush_Drive/mnist/'
embedding_path = 'embeddings/cifar100/'
import os 
import pandas as pd
import json 

sys.path.append('../../')

import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
 
from structs.api import ImageEncoder, GroqClient, ImageExtractor

path = '/Volumes/Ayush_Drive/mnist/'
embedding_path = 'embeddings/cifar100/'
if os.path.exists(path):
    prefix = path
else:
    prefix = ''

data_path = os.path.join(prefix, 'data')

# Load CIFAR100 datasets
# train_dataset = datasets.CIFAR100(data_path, train=True, download=True)
test_dataset = datasets.CIFAR100(data_path, train=False, download=True)

# Combine train and test datasets
# combined_dataset = ConcatDataset([train_dataset, test_dataset])


# Initialize components
extractor = ImageExtractor(dataset=test_dataset)
encoder = ImageEncoder()
client = GroqClient()  # Will use env variable

prompt_tokens = 0
completion_tokens = 0

def extract_last_bracketed_line(response: str) -> str:
    lines = [line.strip() for line in response.strip().splitlines() if line.strip().startswith("[") and line.strip().endswith("]")]
    return lines[-1][1:-1].strip() if lines else None
# ================================== PROCESS NEURON ========================================
def process_image_neuron(indices, activations): 
    prompt = f"""
    You are a meticulous AI researcher conducting an important investigation into patterns found in Image models. Your task is to analyze image examples and provide an interpretation that thoroughly encapsulates possible patterns a specific neuron is detecting.
    Guidelines:
    You will be given a set of image examples where a specific neuron shows high activation. The activation strength for each image is listed in parentheses.
    Try to produce a concise final description. Describe the visual features that are common across these high-activating images, and what patterns you found.
    Focus on identifying specific visual elements, shapes, textures, objects, colors, spatial arrangements, or semantic concepts that might trigger this neuron.
    Do not just list the images or their labels. Instead, analyze the images and provide a coherent interpretation of what the neuron is responding to.
    If the examples are uninformative or seem random, you don't need to force an interpretation. Simply note that no clear pattern is evident by putting "None" in the interpretation.
    Keep your interpretations short and concise.
    Do not make lists of possible interpretations.
    The last line of your response must be the formatted interpretation, using [interpretation]:

    The activations are " [{activations}] " and the images are attached in order. 
    """

    extracted_images = extractor.extract_images(indices)

    # Prepare images for API
    api_ready_images = encoder.prepare_for_api(extracted_images)
    content = client.create_content_array(prompt, api_ready_images)

    # Groq model selection (update with appropriate model that supports vision)
    response = client.send_request(content, model="meta-llama/llama-4-scout-17b-16e-instruct")
    
    # Check for API errors
    if "error" in response:
        print(f"Error from API: {response['error']}")
        return None
    
    if "choices" not in response or not response["choices"]:
        print("Invalid response from API: No choices found")
        return None
    
    return response

def process_text_neuron(labels, activations): 
    prompt = f"""
    You are a neural interpretability researcher analyzing automatically generated descriptions of neuron activations from a sparse autoencoder trained on image data.

    Your task is to synthesize these noisy or verbose interpretations into a clean, concise label that best captures the feature or visual pattern the latent responds to.

    Instructions:
    - Carefully read the provided interpretations.
    - Identify recurring visual features, shapes, objects, or spatial patterns.
    - Ignore vague or speculative content.
    - Return a **short phrase (1–5 words)** or a **single sentence** that best describes the latent.
    - If no coherent pattern emerges, return `"None"`.

    The last line of your response must be the formatted interpretation, using [interpretation]:

    Example input:
    Descriptions:
    - "Activates on images with vertical stripes on the left side"
    - "Strongly responds to narrow vertical patterns"
    - "Often seen in digits like '1' or '4' with upright strokes"

    Output:
   [left vertical line]

    
    Now process the following. I will provide you with a set of labels and their corresponding activations:

    The activations are " [{activations}] " and the labels are attached in order.
    The labels are " [{labels}] "
    """

    # extracted_images = extractor.extract_images(indices)

    # Prepare images for API
    # api_ready_images = encoder.prepare_for_api(extracted_images)
    content = client.create_content_array(prompt, [])

    # Groq model selection (update with appropriate model that supports vision)
    response = client.send_request(content, model="meta-llama/llama-4-scout-17b-16e-instruct")
    
    # Check for API errors
    if "error" in response:
        print(f"Error from API: {response['error']}")
        return None
    
    if "choices" not in response or not response["choices"]:
        print("Invalid response from API: No choices found")
        return None
    
    return response

# ================================= PROCESS LOOP ========================================
# read in data: 
depth = 2
neurons = pd.read_parquet(os.path.join(path, 'laion', 'analysis', f'sae_depth_{depth}_decoder_data_filtered.parquet'))
interpretations = 

neuron_set = set()
results = []

# Create metrics directory if it doesn't exist
os.makedirs(os.path.join(path, "metrics"), exist_ok=True)
cache_file = os.path.join(path, "metrics", f"neuron_results_laion_{depth}.jsonl")

if os.path.exists(cache_file):
    with open(cache_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            neuron_set.add((entry['feature'], entry['depth']))


for row in tqdm(neurons.itertuples(), desc="Processing neurons"):
    neuron_id = row.neuron_id
    activations = row.top_values
    indices = row.top_indices

    # maks out any activations and indices that are 0 
    mask = activations != 0
    activations = activations[mask]
    indices = indices[mask]

    # now pull the labels 

    # Now need to do the whole response stuff. 
    key = (neuron_id, depth)
    if key in neuron_set:
        print(f"Neuron {neuron_id} already processed.")
        continue

    try: 
        response = process_text_neuron(indices[:5], activations[:5])

        if not response: 
            print(f"Failed to process neuron {neuron_id}.")
            continue
        
        content = response['choices'][0]['message']['content']
        interpretation = extract_last_bracketed_line(content)

        result = {
            "feature": neuron_id,
            "depth": depth,
            "response": interpretation,
            "text": content
        }

        with open(cache_file, 'a') as f:
            f.write(json.dumps(result) + "\n")

        results.append(result)
        neuron_set.add(key)

        if 'usage' in response:
            prompt_tokens += response['usage'].get('prompt_tokens', 0)
            completion_tokens += response['usage'].get('completion_tokens', 0)
            print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
    
    except Exception as e:
        print(f"Error processing neuron {neuron_id}, depth {depth}: {e}")
        continue
