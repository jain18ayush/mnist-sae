import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import json
import csv
import logging
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_evolution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("feature_evolution")

def validate_activations(activations_by_depth: Dict[int, torch.Tensor]) -> None:
    """Validate activation tensors dictionary structure."""
    if not isinstance(activations_by_depth, dict):
        raise TypeError(f"Expected activations_by_depth to be a dictionary, got {type(activations_by_depth)}")
    
    for depth, activations in activations_by_depth.items():
        if not isinstance(depth, int):
            raise TypeError(f"Expected depth key to be an integer, got {type(depth)}")
        if not isinstance(activations, torch.Tensor):
            raise TypeError(f"Expected activations to be a torch.Tensor, got {type(activations)}")
        if activations.dim() != 2:
            raise ValueError(f"Expected activations to be 2D tensor [samples, features], got shape {activations.shape}")

def compute_feature_similarity_matrix(
    activations1: torch.Tensor, 
    activations2: torch.Tensor
) -> torch.Tensor:
    """
    Efficiently compute pairwise cosine similarity between all features in two activation sets.
    
    Args:
        activations1: Tensor of shape [num_samples, num_features1]
        activations2: Tensor of shape [num_samples, num_features2]
    
    Returns:
        Similarity matrix of shape [num_features1, num_features2]
    """
    # Input validation
    assert isinstance(activations1, torch.Tensor), f"Expected activations1 to be torch.Tensor, got {type(activations1)}"
    assert isinstance(activations2, torch.Tensor), f"Expected activations2 to be torch.Tensor, got {type(activations2)}"
    assert activations1.dim() == 2, f"Expected activations1 to be 2D tensor, got shape {activations1.shape}"
    assert activations2.dim() == 2, f"Expected activations2 to be 2D tensor, got shape {activations2.shape}" 
    assert activations1.shape[0] == activations2.shape[0], (
        f"Expected activations1 and activations2 to have the same number of samples, "
        f"got {activations1.shape[0]} and {activations2.shape[0]}"
    )
    
    try:
        # Normalize each feature vector for cosine similarity calculation
        norm1 = torch.norm(activations1, dim=0, keepdim=True)
        norm2 = torch.norm(activations2, dim=0, keepdim=True)
        
        # Replace zero norms with 1 to avoid division by zero
        norm1 = torch.where(norm1 == 0, torch.ones_like(norm1), norm1)
        norm2 = torch.where(norm2 == 0, torch.ones_like(norm2), norm2)
        
        normalized1 = activations1 / norm1
        normalized2 = activations2 / norm2
        
        # Compute cosine similarity matrix in one operation
        similarity_matrix = torch.mm(normalized1.t(), normalized2)
        
        # Sanity check to ensure values are in valid range for cosine similarity
        assert torch.all(similarity_matrix >= -1.01) and torch.all(similarity_matrix <= 1.01), (
            f"Similarity matrix contains values outside the expected range [-1, 1]: "
            f"min={similarity_matrix.min().item()}, max={similarity_matrix.max().item()}"
        )
        
        return similarity_matrix
    except Exception as e:
        logger.error(f"Error computing similarity matrix: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def identify_related_features_all_fast(
    activations_by_depth: Dict[int, torch.Tensor], 
    similarity_threshold: float = 0.3, 
    max_depth: int = 9,
    max_related_per_feature: int = 3
) -> Dict[int, Dict[int, List[int]]]:
    """
    Efficiently identify related features across all depths for all features at once.
    
    Args:
        activations_by_depth: Dictionary mapping depth to activation tensors
        similarity_threshold: Minimum cosine similarity to consider features related
        max_depth: Maximum depth to analyze
        max_related_per_feature: Maximum number of related features to track per feature
    
    Returns:
        Dictionary mapping feature_idx at depth 1 to related features at each depth
    """
    # Validate inputs
    validate_activations(activations_by_depth)
    assert 0 < similarity_threshold <= 1, f"similarity_threshold must be in (0, 1], got {similarity_threshold}"
    assert isinstance(max_depth, int) and max_depth > 0, f"max_depth must be a positive integer, got {max_depth}"
    assert isinstance(max_related_per_feature, int) and max_related_per_feature > 0, (
        f"max_related_per_feature must be a positive integer, got {max_related_per_feature}"
    )
    
    all_related_features = {}
    
    # Check if depth 1 exists
    if 1 not in activations_by_depth:
        logger.error("Depth 1 is missing from activations_by_depth")
        raise ValueError("activations_by_depth must contain depth 1")
    
    # Pre-compute similarity matrices between consecutive depths
    similarity_matrices = {}
    for depth in range(1, max_depth):
        if depth+1 not in activations_by_depth:
            logger.warning(f"Depth {depth+1} not found in activations_by_depth, skipping")
            continue
        if depth not in activations_by_depth:
            logger.warning(f"Depth {depth} not found in activations_by_depth, skipping")
            continue
        
        try:
            logger.info(f"Computing similarity matrix for depths {depth} → {depth+1}...")
            similarity_matrices[depth] = compute_feature_similarity_matrix(
                activations_by_depth[depth], 
                activations_by_depth[depth+1]
            )
            logger.info(f"Similarity matrix for depths {depth} → {depth+1} computed successfully")
        except Exception as e:
            logger.error(f"Failed to compute similarity matrix for depths {depth} → {depth+1}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    # For each feature at depth 1, find related features at all depths
    num_features_depth1 = activations_by_depth[1].shape[1]
    
    for feature_idx in tqdm(range(num_features_depth1), desc="Processing features"):
        try:
            related_by_depth = {1: [feature_idx]}  # Start with the feature itself at depth 1
            
            # For each subsequent depth, find related features based on previous depth
            for depth in range(1, max_depth):
                if depth+1 not in activations_by_depth or depth not in related_by_depth:
                    continue
                    
                # Get previous depth's related features
                prev_features = related_by_depth[depth]
                
                # Get similarity matrix for this depth transition
                if depth not in similarity_matrices:
                    logger.warning(f"No similarity matrix found for depth {depth} → {depth+1}, skipping")
                    continue
                
                sim_matrix = similarity_matrices[depth]
                
                # Find all features at next depth that have high similarity with any feature from current depth
                related_next_depth = set()
                for prev_idx in prev_features:
                    # Validate index
                    if prev_idx >= sim_matrix.shape[0]:
                        logger.warning(
                            f"Feature index {prev_idx} at depth {depth} is out of bounds "
                            f"(max: {sim_matrix.shape[0]-1}), skipping"
                        )
                        continue
                    
                    # Find indices where similarity exceeds threshold
                    related_indices = torch.where(sim_matrix[prev_idx] > similarity_threshold)[0].tolist()
                    
                    # Take only top N most similar features to avoid explosion of relations
                    if len(related_indices) > max_related_per_feature:
                        similarities = [(idx, sim_matrix[prev_idx, idx].item()) for idx in related_indices]
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        related_indices = [idx for idx, _ in similarities[:max_related_per_feature]]
                    
                    related_next_depth.update(related_indices)
                
                if related_next_depth:
                    related_by_depth[depth+1] = list(related_next_depth)
            
            all_related_features[feature_idx] = related_by_depth
            
        except Exception as e:
            logger.error(f"Error processing feature {feature_idx}: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue with next feature
    
    if not all_related_features:
        logger.warning("No related features found across depths")
    
    return all_related_features

def visualize_feature_evolution_single(
    feature_idx: int, 
    related_features: Dict[int, Dict[int, List[int]]], 
    activations_by_depth: Dict[int, torch.Tensor], 
    dataset: Any,
    output_dir: str, 
    max_depth: int = 9, 
    top_k: int = 5, 
    save_indices: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Visualize how a specific feature at depth 1 evolves through subsequent depths.
    
    Args:
        feature_idx: Index of the feature at depth 1 to track
        related_features: Dictionary mapping depth to list of related feature indices
        activations_by_depth: Dictionary mapping depth to activation tensors
        dataset: MNIST dataset containing images
        output_dir: Directory to save visualizations
        max_depth: Maximum depth to visualize
        top_k: Number of top activating images to show per feature
        save_indices: Whether to save the indices matrix
    
    Returns:
        Dictionary containing image indices and activation values for each depth
    """
    try:
        # Input validation
        if feature_idx not in related_features:
            logger.warning(f"Feature {feature_idx} not found in related_features, skipping")
            return None
        
        if 1 not in activations_by_depth:
            logger.error("Depth 1 not found in activations_by_depth")
            return None
        
        if dataset is None:
            logger.error("Dataset cannot be None")
            return None
        
        feature_data = related_features[feature_idx]
        
        # Dictionary to store indices and activation values
        indices_data = {
            "feature_idx": int(feature_idx),
            "depth_data": {}
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get feature at depth 1
        base_activations = activations_by_depth[1]
        if feature_idx >= base_activations.shape[1]:
            logger.error(f"Feature index {feature_idx} is out of bounds for depth 1 activations")
            return None
            
        base_feature = base_activations[:, feature_idx]
        
        # Create figure
        fig_width = 15
        fig_height = 2.5 * min(max_depth, 9)  # Limit height for very deep models
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Show top activating images for base feature
        if top_k > base_feature.shape[0]:
            logger.warning(
                f"top_k ({top_k}) is greater than the number of available samples ({base_feature.shape[0]}), "
                f"using {base_feature.shape[0]} instead"
            )
            top_k = base_feature.shape[0]
            
        base_top_indices = torch.topk(base_feature, top_k).indices.cpu()
        
        # Store indices and activation values for depth 1
        indices_data["depth_data"][1] = {
            "feature": int(feature_idx),
            "indices": [int(idx.item()) for idx in base_top_indices],
            "activations": [float(base_feature[idx.item()].item()) for idx in base_top_indices]
        }
        
        # Plot top activating images for depth 1
        try:
            for j, idx in enumerate(base_top_indices):
                ax = plt.subplot(max_depth, top_k, j+1)
                idx_item = idx.item()
                
                # Handle different dataset formats safely
                try:
                    # Attempt to get the image with expected format [3, 28, 28]
                    img = dataset[idx_item][0]
                    
                    # Check if the image is a tensor and has the expected shape
                    if isinstance(img, torch.Tensor):
                        if img.dim() == 3 and img.shape[0] in [1, 3]:
                            # Handle RGB (3 channels) or grayscale (1 channel)
                            if img.shape[0] == 3:
                                # RGB image, convert to numpy and permute dimensions
                                img = img.permute(1, 2, 0).cpu().numpy()
                                # Convert to grayscale for visualization
                                img = img.mean(axis=2)
                            else:
                                # Already grayscale, just squeeze and convert to numpy
                                img = img.squeeze(0).cpu().numpy()
                        else:
                            logger.warning(
                                f"Unexpected image shape: {img.shape}, using alternative display method"
                            )
                            img = img.cpu().numpy().astype(float)
                    else:
                        logger.warning(
                            f"Image is not a tensor, got {type(img)}, using alternative display method"
                        )
                        img = np.array(img, dtype=float)
                        
                except Exception as e:
                    logger.warning(f"Error processing image at index {idx_item}: {str(e)}")
                    # Use a blank image as fallback
                    img = np.zeros((28, 28), dtype=float)
                
                ax.imshow(img, cmap='gray')
                activation_value = base_feature[idx_item].item()
                ax.set_title(f'{activation_value:.2f}', fontsize=8)
                ax.axis('off')
        
            # Label for base feature
            plt.figtext(0.02, 1.0 - 0.5/fig_height, f'Depth 1\nFeature {feature_idx}', 
                        fontsize=10, ha='left', va='center')
        except Exception as e:
            logger.error(f"Error plotting images for depth 1, feature {feature_idx}: {str(e)}")
            logger.error(traceback.format_exc())
        
        # For each subsequent depth
        for depth in range(2, max_depth + 1):
            if depth not in feature_data or not feature_data[depth]:
                logger.debug(f"No related features found at depth {depth} for feature {feature_idx}")
                continue
            
            if depth not in activations_by_depth:
                logger.warning(f"Depth {depth} not found in activations_by_depth, skipping")
                continue
                
            try:
                # Display top activating images for the first related feature at this depth
                curr_feature = feature_data[depth][0]  # Take first related feature for visualization
                
                # Validate feature index
                curr_activations = activations_by_depth[depth]
                if curr_feature >= curr_activations.shape[1]:
                    logger.warning(
                        f"Feature index {curr_feature} at depth {depth} is out of bounds "
                        f"(max: {curr_activations.shape[1]-1}), skipping"
                    )
                    continue
                
                # Get top activating images for this feature
                if top_k > curr_activations.shape[0]:
                    adjusted_top_k = curr_activations.shape[0]
                    logger.warning(
                        f"top_k ({top_k}) is greater than the number of available samples "
                        f"({curr_activations.shape[0]}) at depth {depth}, "
                        f"using {adjusted_top_k} instead"
                    )
                else:
                    adjusted_top_k = top_k
                    
                curr_top_indices = torch.topk(curr_activations[:, curr_feature], adjusted_top_k).indices.cpu()
                
                # Store indices and activation values for this depth
                indices_data["depth_data"][depth] = {
                    "feature": int(curr_feature),
                    "indices": [int(idx.item()) for idx in curr_top_indices],
                    "activations": [float(curr_activations[idx.item(), curr_feature].item()) for idx in curr_top_indices]
                }
                
                for j, idx in enumerate(curr_top_indices):
                    # Calculate position in grid
                    pos = (depth - 1) * top_k + j + 1
                    if pos <= max_depth * top_k:  # Ensure we don't go out of bounds
                        ax = plt.subplot(max_depth, top_k, pos)
                        idx_item = idx.item()
                        
                        # Handle different dataset formats safely
                        try:
                            # Attempt to get the image with expected format [3, 28, 28]
                            img = dataset[idx_item][0]
                            
                            # Check if the image is a tensor and has the expected shape
                            if isinstance(img, torch.Tensor):
                                if img.dim() == 3 and img.shape[0] in [1, 3]:
                                    # Handle RGB (3 channels) or grayscale (1 channel)
                                    if img.shape[0] == 3:
                                        # RGB image, convert to numpy and permute dimensions
                                        img = img.permute(1, 2, 0).cpu().numpy()
                                        # Convert to grayscale for visualization
                                        img = img.mean(axis=2)
                                    else:
                                        # Already grayscale, just squeeze and convert to numpy
                                        img = img.squeeze(0).cpu().numpy()
                                else:
                                    logger.warning(
                                        f"Unexpected image shape: {img.shape}, using alternative display method"
                                    )
                                    img = img.cpu().numpy().astype(float)
                            else:
                                logger.warning(
                                    f"Image is not a tensor, got {type(img)}, using alternative display method"
                                )
                                img = np.array(img, dtype=float)
                                
                        except Exception as e:
                            logger.warning(f"Error processing image at index {idx_item}: {str(e)}")
                            # Use a blank image as fallback
                            img = np.zeros((28, 28), dtype=float)
                        
                        ax.imshow(img, cmap='gray')
                        activation_value = curr_activations[idx_item, curr_feature].item()
                        ax.set_title(f'{activation_value:.2f}', fontsize=8)
                        ax.axis('off')
                
                # Label for this feature
                plt.figtext(0.02, 1.0 - (depth-0.5)/max_depth, 
                          f'Depth {depth}\nFeature {curr_feature}', 
                          fontsize=10, ha='left', va='center')
            except Exception as e:
                logger.error(f"Error processing depth {depth} for feature {feature_idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.1)
        
        # Save the visualization
        plt_filename = os.path.join(output_dir, f'feature_{feature_idx}_evolution.png')
        try:
            plt.savefig(plt_filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {plt_filename}")
        except Exception as e:
            logger.error(f"Failed to save visualization to {plt_filename}: {str(e)}")
        finally:
            plt.close()
        
        # Save the indices and activation values
        if save_indices:
            try:
                # Save as JSON
                json_filename = os.path.join(output_dir, f'feature_{feature_idx}_indices.json')
                with open(json_filename, 'w') as f:
                    json.dump(indices_data, f, indent=2)
                logger.info(f"Saved indices to {json_filename}")
                
                # Also save as CSV for easier import into other tools
                csv_filename = os.path.join(output_dir, f'feature_{feature_idx}_indices.csv')
                with open(csv_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Depth', 'Feature', 'Position', 'Dataset_Index', 'Activation'])
                    
                    for depth, data in indices_data["depth_data"].items():
                        feature = data["feature"]
                        for pos, (idx, act) in enumerate(zip(data["indices"], data["activations"])):
                            writer.writerow([depth, feature, pos, idx, act])
                logger.info(f"Saved CSV to {csv_filename}")
            except Exception as e:
                logger.error(f"Failed to save indices for feature {feature_idx}: {str(e)}")
                logger.error(traceback.format_exc())
        
        return indices_data
    
    except Exception as e:
        logger.error(f"Error visualizing feature {feature_idx}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_batch_evolution(args):
    """Helper function for parallel processing"""
    feature_indices, related_features, activations_by_depth, dataset, output_dir, max_depth, top_k = args
    results = {}
    for idx in feature_indices:
        try:
            result = visualize_feature_evolution_single(
                idx, related_features, activations_by_depth, dataset, output_dir, max_depth, top_k
            )
            if result:
                results[idx] = result
        except Exception as e:
            logger.error(f"Error processing feature {idx} in batch: {str(e)}")
            logger.error(traceback.format_exc())
    return results

def visualize_all_features_evolution_parallel(
    related_features: Dict[int, Dict[int, List[int]]], 
    activations_by_depth: Dict[int, torch.Tensor], 
    dataset: Any,
    output_dir: str, 
    max_depth: int = 9, 
    top_k: int = 5, 
    num_workers: Optional[int] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Visualize all features in parallel using multiprocessing
    
    Args:
        related_features: Dictionary mapping feature_idx to related features by depth
        activations_by_depth: Dictionary mapping depth to activation tensors
        dataset: Dataset containing images
        output_dir: Directory to save visualizations
        max_depth: Maximum depth to visualize
        top_k: Number of top activating images to show per feature
        num_workers: Number of parallel workers (default: CPU count - 1)
    
    Returns:
        Dictionary with indices data for all features
    """
    try:
        # Input validation
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all feature indices at depth 1
        all_features = list(related_features.keys())
        if not all_features:
            logger.warning("No features found in related_features")
            return {}
        
        # Split features into batches for parallel processing
        batch_size = max(1, len(all_features) // num_workers)
        feature_batches = [all_features[i:i+batch_size] for i in range(0, len(all_features), batch_size)]
        
        # Prepare arguments for each worker
        args_list = [(batch, related_features, activations_by_depth, dataset, output_dir, max_depth, top_k) 
                    for batch in feature_batches]
        
        # Use ProcessPoolExecutor for parallel processing
        logger.info(f"Processing {len(all_features)} features using {num_workers} workers...")
        
        all_results = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Process batches in parallel
            futures = list(executor.map(process_batch_evolution, args_list))
            
            # Combine results from all batches
            for result_dict in futures:
                all_results.update(result_dict)
        
        # Save all indices in one combined file
        combined_json_path = os.path.join(output_dir, 'all_features_indices.json')
        try:
            with open(combined_json_path, 'w') as f:
                json.dump(all_results, f)
            logger.info(f"Saved combined indices to {combined_json_path}")
        except Exception as e:
            logger.error(f"Failed to save combined indices: {str(e)}")
            logger.error(traceback.format_exc())
        
        return all_results
    
    except Exception as e:
        logger.error(f"Error in parallel visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def analyze_and_visualize_feature_evolution(
    activations_by_depth: Dict[int, torch.Tensor], 
    dataset: Any,
    output_dir: str = "feature_evolutions",
    similarity_threshold: float = 0.3, 
    max_depth: int = 9, 
    top_k: int = 5, 
    num_workers: Optional[int] = None, 
    batch_processing: bool = True
) -> Tuple[Dict[int, Dict[int, List[int]]], Dict[int, Dict[str, Any]]]:
    """
    Analyze feature evolution and visualize all features efficiently
    
    Args:
        activations_by_depth: Dictionary mapping depth to activation tensors
        dataset: Dataset containing images
        output_dir: Directory to save visualizations
        similarity_threshold: Minimum similarity to consider features related
        max_depth: Maximum depth to analyze
        top_k: Number of top activating images to show per feature
        num_workers: Number of parallel workers (defaults to CPU count - 1)
        batch_processing: Whether to use batch processing (True) or process one feature at a time (False)
    
    Returns:
        Tuple of (related_features, indices_data)
    """
    try:
        # Input validation
        validate_activations(activations_by_depth)
        
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        if not isinstance(output_dir, str):
            raise TypeError(f"output_dir must be a string, got {type(output_dir)}")
        
        if not (0 < similarity_threshold <= 1):
            raise ValueError(f"similarity_threshold must be in (0, 1], got {similarity_threshold}")
        
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError(f"max_depth must be a positive integer, got {max_depth}")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        
        # Ensure activations are on CPU for multiprocessing compatibility
        for depth in activations_by_depth:
            if isinstance(activations_by_depth[depth], torch.Tensor) and activations_by_depth[depth].is_cuda:
                logger.info(f"Moving activations for depth {depth} to CPU for multiprocessing compatibility")
                activations_by_depth[depth] = activations_by_depth[depth].cpu()
        
        # Step 1: Efficiently identify related features across depths
        logger.info("Identifying related features across depths...")
        related_features = identify_related_features_all_fast(
            activations_by_depth, 
            similarity_threshold=similarity_threshold,
            max_depth=max_depth
        )
        logger.info(f"Found related features across depths for {len(related_features)} features")
        
        # Step 2: Visualize feature evolution
        logger.info("Generating visualizations...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a separate indices directory
        indices_dir = os.path.join(output_dir, 'indices')
        os.makedirs(indices_dir, exist_ok=True)
        
        indices_data = {}
        
        if batch_processing:
            # Process all features in parallel
            logger.info("Using batch processing for visualization")
            indices_data = visualize_all_features_evolution_parallel(
                related_features, 
                activations_by_depth, 
                dataset,
                output_dir, 
                max_depth=max_depth,
                top_k=top_k,
                num_workers=num_workers
            )
        else:
            # Process one feature at a time (less memory intensive)
            logger.info("Processing features one at a time")
            num_features = len(related_features)
            for idx in tqdm(related_features.keys(), desc="Visualizing features", total=num_features):
                try:
                    result = visualize_feature_evolution_single(
                        idx, 
                        related_features, 
                        activations_by_depth, 
                        dataset, 
                        output_dir, 
                        max_depth=max_depth,
                        top_k=top_k
                    )
                    if result:
                        indices_data[idx] = result
                except Exception as e:
                    logger.error(f"Error processing feature {idx}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
            
            # Save all indices in one combined file
            try:
                combined_json_path = os.path.join(output_dir, 'all_features_indices.json')
                with open(combined_json_path, 'w') as f:
                    json.dump(indices_data, f, indent=2)
                logger.info(f"Saved combined indices to {combined_json_path}")
            except Exception as e:
                logger.error(f"Failed to save combined indices: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Create indexing matrices
        try:
            # Calculate the max feature index to determine the size of the array
            if related_features:
                max_feature_idx = max(related_features.keys())
                
                # Create matrices with proper dimensions
                indexing_matrix = np.zeros((max_feature_idx + 1, max_depth, top_k), dtype=np.int32)
                activation_matrix = np.zeros((max_feature_idx + 1, max_depth, top_k), dtype=np.float32)
                
                # Fill the matrices with data from indices_data
                for feature_idx, data in indices_data.items():
                    feature_idx = int(feature_idx)  # Ensure it's an integer
                    if feature_idx > max_feature_idx:
                        logger.warning(f"Feature index {feature_idx} exceeds max_feature_idx {max_feature_idx}, skipping")
                        continue
                        
                    for depth_str, depth_data in data.get("depth_data", {}).items():
                        depth = int(depth_str)
                        if depth <= max_depth:
                            indices = depth_data.get("indices", [])
                            activations = depth_data.get("activations", [])
                            for k in range(min(len(indices), top_k)):
                                indexing_matrix[feature_idx, depth-1, k] = indices[k]
                                activation_matrix[feature_idx, depth-1, k] = activations[k]
                
                # Save the matrices
                np.save(os.path.join(output_dir, 'feature_evolution_indices_matrix.npy'), indexing_matrix)
                np.save(os.path.join(output_dir, 'feature_evolution_activations_matrix.npy'), activation_matrix)
                
                logger.info(f"Saved index and activation matrices to {output_dir}")
            else:
                logger.warning("No related features found, skipping matrix creation")
                
        except Exception as e:
            logger.error(f"Error creating indexing matrices: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info(f"Analysis complete. Visualizations saved to {output_dir}")
        return related_features, indices_data
    
    except Exception as e:
        logger.error(f"Error in analyze_and_visualize_feature_evolution: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}

def load_images_from_indices(
    indices_matrix_path: str, 
    dataset: Any, 
    feature_idx: int, 
    depth_idx: Optional[int] = None
) -> Dict[int, List[torch.Tensor]]:
    """
    Load images from the saved indices matrix for a specific feature.
    
    Args:
        indices_matrix_path: Path to the saved indices matrix
        dataset: Dataset containing images
        feature_idx: Index of the feature to load images for
        depth_idx: Optional depth index (0-based). If None, returns images for all depths
        
    Returns:
        Dictionary mapping depth to list of images
    """
    try:
        # Input validation
        if not os.path.exists(indices_matrix_path):
            logger.error(f"Indices matrix file not found: {indices_matrix_path}")
            return {}
        
        if dataset is None:
            logger.error("Dataset cannot be None")
            return {}
        
        if not isinstance(feature_idx, int) or feature_idx < 0:
            logger.error(f"feature_idx must be a non-negative integer, got {feature_idx}")
            return {}
        
        if depth_idx is not None and (not isinstance(depth_idx, int) or depth_idx < 0):
            logger.error(f"depth_idx must be a non-negative integer or None, got {depth_idx}")
            return {}
        
        # Load the indices matrix
        try:
            indices_matrix = np.load(indices_matrix_path)
            logger.info(f"Loaded indices matrix with shape {indices_matrix.shape}")
        except Exception as e:
            logger.error(f"Error loading indices matrix: {str(e)}")
            return {}
        
        # Validate feature index
        if feature_idx >= indices_matrix.shape[0]:
            logger.error(f"Feature index {feature_idx} is out of bounds for matrix with shape {indices_matrix.shape}")
            return {}
        
        # If no specific depth is requested, return all depths
        if depth_idx is None:
            depths_to_load = range(indices_matrix.shape[1])
        else:
            if depth_idx >= indices_matrix.shape[1]:
                logger.error(f"Depth index {depth_idx} is out of bounds for matrix with shape {indices_matrix.shape}")
                return {}
            depths_to_load = [depth_idx]
        
        images_by_depth = {}
        
        for depth in depths_to_load:
            images = []
            for k in range(indices_matrix.shape[2]):
                idx = indices_matrix[feature_idx, depth, k]
                if idx > 0:  # Skip zeros (no image)
                    try:
                        # Get the image tensor
                        img = dataset[idx][0]
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"Error loading image at index {idx}: {str(e)}")
                        continue
            
            # Convert to 1-based depth for consistency with other functions
            images_by_depth[depth+1] = images
        
        return images_by_depth
    
    except Exception as e:
        logger.error(f"Error in load_images_from_indices: {str(e)}")
        logger.error(traceback.format_exc())
        return {}


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from structs.utils import fc1_config, encoder_config, decoder_config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from structs.models import ColoredMNISTModel
from structs.models import EnhancedSAE, SimpleSAE

torch.manual_seed(42)

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

import os 
path = '/Volumes/Ayush_Drive/mnist'

if os.path.exists(path):
    prefix = path
else:
    prefix = ''

# load activations 
activation_dict = {}

for i in tqdm(range(1, 10), desc="loading activations"):
    activation_dict[i] = torch.load(os.path.join(path, 'embeddings', f'mnist_encoder_fc1_depth_{i}.pth'))

# load dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


class ToRGB:
    def __call__(self, img):
        return img.repeat(3, 1, 1)  # Repeat the grayscale channel 3 times

# Updated transforms for colored MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    ToRGB(),  # Convert to RGB
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307],  # Same normalization for each channel
                       std=[0.3081, 0.3081, 0.3081])
])

# Load datasets
train_dataset = datasets.MNIST(root=f'{prefix}/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=f'{prefix}/data', train=False, download=True, transform=transform)
dataset = ConcatDataset([train_dataset, test_dataset])

# Get total cores available
total_cores = multiprocessing.cpu_count()
print(f"Total logical cores: {total_cores}")

#determine num workers 
meta_sae_workers = max(1, total_cores // 2)

related_features, indices_data = analyze_and_visualize_feature_evolution(
    activations_by_depth=activation_dict,
    dataset=dataset,
    output_dir=os.path.join(path,"feature_evolution_output"),
    similarity_threshold=0.3,
    max_depth=5,
    top_k=5,
    num_workers=2,  # Adjust based on your CPU
    batch_processing=True  # Set to False for lower memory usage
)

# Example usage (commented out):
"""
# How to use this code:

# 1. Load your activations for each depth
activations_by_depth = {
    1: torch.load('embeddings/mnist_encoder_depth_1_FashionMNIST.pth'),
    2: torch.load('embeddings/mnist_encoder_depth_2_FashionMNIST.pth'),
    # ... load all depths
}

# 2. Load your dataset
from torchvision import datasets, transforms

class ToRGB:
    def __call__(self, img):
        return img.repeat(3, 1, 1)  # Repeat grayscale channel 3 times

transform = transforms.Compose([
    transforms.ToTensor(),
    ToRGB(),  # Convert to RGB
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
])

dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)

# 3. Run the analysis
related_features, indices_data = analyze_and_visualize_feature_evolution(
    activations_by_depth=activations_by_depth,
    dataset=dataset,
    output_dir="feature_evolution_output",
    similarity_threshold=0.3,
    max_depth=9,
    top_k=5,
    num_workers=4,  # Adjust based on your CPU
    batch_processing=True  # Set to False for lower memory usage
)

# 4. Later, to load images for a specific feature:
images = load_images_from_indices(
    indices_matrix_path='feature_evolution_output/feature_evolution_indices_matrix.npy',
    dataset=dataset,
    feature_idx=42  # The feature you're interested in
)
"""