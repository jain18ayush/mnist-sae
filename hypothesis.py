import torch 
torch.manual_seed(42)

import os 
path = '/Volumes/Sid_Drive/mnist/'

if os.path.exists(path):
    prefix = path
else:
    prefix = ''

import torch
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, dendrogram
import os

class CrossDatasetAnalyzer:
    def __init__(self, dataset_names, max_depth=9, prefix=''):
        """
        Args:
            prefix: Path prefix for loading files
            dataset_names: List of dataset names to analyze
            max_depth: Maximum depth to analyze
        """
        self.prefix = prefix
        self.dataset_names = dataset_names
        self.max_depth = max_depth
        os.makedirs('plots/hypothesis', exist_ok=True)
    
    def load_depth_embeddings(self, depth, dataset_name):
        """Load embeddings for a specific depth and dataset"""
        path = f'{self.prefix}embeddings/mnist_encoder_{dataset_name}_depth_{depth}.pth'
        return torch.load(path)

    def analyze_activation_patterns(self, activations):
        """Analyze activation patterns"""
        metrics = {}
        
        # 1. Activation Statistics
        metrics['mean_activation'] = torch.mean(activations).item()
        metrics['activation_std'] = torch.std(activations).item()
        metrics['sparsity'] = (activations == 0).float().mean().item()
        
        # 2. Active Feature Count
        threshold = activations.mean() + activations.std()
        active_features = (activations > threshold).sum(dim=1)
        metrics['avg_active_features'] = active_features.float().mean().item()
        
        # 3. Feature Utilization
        feature_usage = (activations > threshold).float().mean(dim=0)
        metrics['feature_utilization'] = feature_usage.mean().item()
        metrics['feature_utilization_std'] = feature_usage.std().item()
        
        # 4. Activation Distribution
        normalized = torch.nn.functional.softmax(activations, dim=1)
        activation_entropy = entropy(normalized.numpy(), axis=1)
        metrics['activation_entropy'] = np.mean(activation_entropy)
        
        return metrics

    def analyze_feature_overlap(self, activations_by_depth, batch_size=1000):
        """
        Measure how much features overlap between consecutive depths using batched processing
        to avoid memory issues.
        
        Args:
            activations_by_depth: List of activation tensors at different depths
            batch_size: Size of batches to process at once
        """
        overlaps = []
        
        for i in range(len(activations_by_depth)-1):
            current = activations_by_depth[i]
            next_depth = activations_by_depth[i+1]
            n = current.shape[0]
            
            # Process in batches to compute max similarities
            max_similarities = torch.zeros(n)
            
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                current_batch = current[batch_start:batch_end].unsqueeze(1)  # [batch_size, 1, dim]
                
                batch_max_similarities = torch.zeros(batch_end - batch_start)
                
                # Process next_depth in batches too
                for next_batch_start in range(0, n, batch_size):
                    next_batch_end = min(next_batch_start + batch_size, n)
                    next_batch = next_depth[next_batch_start:next_batch_end].unsqueeze(0)  # [1, batch_size, dim]
                    
                    # Compute similarities for this batch pair
                    batch_similarities = F.cosine_similarity(current_batch, next_batch, dim=2)
                    
                    # Update max similarities for current batch
                    batch_max_similarities = torch.max(
                        batch_max_similarities,
                        batch_similarities.max(dim=1)[0]
                    )
                
                max_similarities[batch_start:batch_end] = batch_max_similarities
                
            overlaps.append({
                'depth': i + 1,
                'mean_overlap': max_similarities.mean().item(),
                'std_overlap': max_similarities.std().item()
            })
        
        return overlaps

    def measure_feature_specificity(self, activations):
        """Measure how specifically features respond to inputs"""
        # Calculate activation distributions
        mean_activations = activations.mean(dim=0)
        std_activations = activations.std(dim=0)
        
        # Calculate peakedness (kurtosis) of activation distributions
        kurtosis = ((activations - mean_activations)**4).mean(dim=0) / (std_activations**4 + 1e-10)
        
        return {
            'mean_kurtosis': kurtosis.mean().item(),
            'std_kurtosis': kurtosis.std().item()
        }

    def analyze_feature_hierarchy(self, activations):
        """Analyze hierarchical relationships between features with proper handling of edge cases"""
        # Add small epsilon to avoid zero variance
        eps = 1e-8
        activations = activations + eps
        
        # Remove features with zero or near-zero variance
        variances = torch.var(activations, dim=0)
        valid_features = variances > eps
        
        if not torch.any(valid_features):
            return {
                'n_clusters': 1,
                'avg_cluster_size': activations.shape[1],
                'linkage_matrix': None,
                'error': 'No valid features found'
            }
        
        # Filter activations to only include valid features
        filtered_activations = activations[:, valid_features]
        
        try:
            # Compute feature similarity matrix
            feature_similarities = torch.corrcoef(filtered_activations.T)
            
            # Convert to numpy and handle any remaining NaN values
            feature_similarities_np = feature_similarities.numpy()
            feature_similarities_np = np.nan_to_num(feature_similarities_np)
            
            # Ensure the matrix is symmetric and contains only finite values
            np.fill_diagonal(feature_similarities_np, 1.0)
            feature_similarities_np = (feature_similarities_np + feature_similarities_np.T) / 2
            
            # Convert similarities to distances
            distances = 1 - np.abs(feature_similarities_np)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distances, method='ward')
            
            # Analyze cluster structure
            n_clusters = len(np.unique(linkage_matrix[:, -1]))
            avg_cluster_size = filtered_activations.shape[1] / n_clusters
            
            return {
                'n_clusters': n_clusters,
                'avg_cluster_size': avg_cluster_size,
                'linkage_matrix': linkage_matrix,
                'error': None
            }
        
        except Exception as e:
            print(f"Error in hierarchical clustering: {str(e)}")
            return {
                'n_clusters': 1,
                'avg_cluster_size': activations.shape[1],
                'linkage_matrix': None,
                'error': str(e)
            }
        
    def analyze_depth_patterns(self, activations):
        """Analyze activation patterns at a specific depth"""
        # Count unique activation patterns
        discretized = (activations > activations.mean()).float()
        unique_patterns = torch.unique(discretized, dim=0).shape[0]
        
        # Measure average activation sparsity
        sparsity = (activations == 0).float().mean()
        
        # Calculate activation entropy
        normalized = F.softmax(activations, dim=1)
        entropy = -(normalized * torch.log(normalized + 1e-10)).sum(1).mean()
        
        return {
            'unique_patterns': unique_patterns,
            'sparsity': sparsity.item(),
            'entropy': entropy.item()
        }

    def analyze_feature_composition(self, prev_activations, curr_activations):
        """Analyze how features are composed across depths"""
        # Fit linear regression to see how current features are composed
        regression = torch.linalg.lstsq(prev_activations, curr_activations)
        
        # Analyze composition weights
        weight_sparsity = (regression.solution == 0).float().mean()
        weight_distribution = regression.solution.std(dim=0)
        
        return {
            'weight_sparsity': weight_sparsity.item(),
            'weight_distribution': weight_distribution.mean().item()
        }

    def compare_datasets(self):
        """Compare activation patterns across datasets and depths"""
        results = {}
        
        for dataset_name in tqdm(self.dataset_names, desc="Processing datasets"):
            dataset_results = {
                'basic_metrics': [],
                'feature_specificity': [],
                'hierarchy': [],
                'patterns': [],
                'compositions': []
            }
            activations_by_depth = []
            
            for depth in tqdm(range(1, self.max_depth + 1), desc=f"Processing depth for {dataset_name}"):
                try:
                    # Load embeddings for this depth
                    activations = self.load_depth_embeddings(depth, dataset_name)
                    activations_by_depth.append(activations)
                    
                    # # Basic metrics
                    # basic_metrics = self.analyze_activation_patterns(activations)
                    # dataset_results['basic_metrics'].append(basic_metrics)
                    
                    # # Feature specificity
                    # specificity = self.measure_feature_specificity(activations)
                    # dataset_results['feature_specificity'].append(specificity)
                    
                    # # Hierarchy analysis
                    # hierarchy = self.analyze_feature_hierarchy(activations)
                    # dataset_results['hierarchy'].append(hierarchy)
                    
                    # # Pattern analysis
                    # patterns = self.analyze_depth_patterns(activations)
                    # dataset_results['patterns'].append(patterns)
                    
                    # # Feature composition (skip first depth)
                    # if depth > 1:
                    #     composition = self.analyze_feature_composition(
                    #         activations_by_depth[-2], activations)
                    #     dataset_results['compositions'].append(composition)
                    
                except FileNotFoundError:
                    print(f"No embeddings found for {dataset_name} at depth {depth}")
                    break
            
            print("Finished gathering results")
            # Analyze feature overlap across all depths
            if len(activations_by_depth) > 1:
                dataset_results['overlaps'] = self.analyze_feature_overlap(activations_by_depth)
            
            results[dataset_name] = dataset_results
            print("Finished processing dataset")
        return results

    def plot_metrics_across_depths(self, results):
        """Plot comprehensive metrics across depths for each dataset"""
        print("plotting metrics")
        # Set the style
        plt.style.use('default')  # clean, professional look
        
        for dataset_name, dataset_results in results.items():
            # Create directory for this dataset's plots
            save_dir = f"plots/hypothesis/{dataset_name}"
            os.makedirs(save_dir, exist_ok=True)
            
            # 1. Basic Activation Metrics
            if dataset_results['basic_metrics']:
                metrics = dataset_results['basic_metrics'][0].keys()
                for metric in metrics:
                    plt.figure(figsize=(10, 6))
                    values = [m[metric] for m in dataset_results['basic_metrics']]
                    plt.plot(range(1, len(values) + 1), values, 'o-', linewidth=2, markersize=8)
                    plt.xlabel('Depth', fontsize=12)
                    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
                    plt.title(f'{metric.replace("_", " ").title()} vs Depth for {dataset_name}', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{save_dir}/{metric}.png", dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 2. Feature Specificity Plot
            if dataset_results['feature_specificity']:
                plt.figure(figsize=(12, 6))
                mean_kurtosis = [m['mean_kurtosis'] for m in dataset_results['feature_specificity']]
                std_kurtosis = [m['std_kurtosis'] for m in dataset_results['feature_specificity']]
                depths = range(1, len(mean_kurtosis) + 1)
                
                plt.plot(depths, mean_kurtosis, 'b-o', label='Mean Kurtosis', linewidth=2, markersize=8)
                plt.fill_between(depths, 
                            [m - s for m, s in zip(mean_kurtosis, std_kurtosis)],
                            [m + s for m, s in zip(mean_kurtosis, std_kurtosis)],
                            alpha=0.2, color='blue')
                plt.xlabel('Depth', fontsize=12)
                plt.ylabel('Feature Specificity (Kurtosis)', fontsize=12)
                plt.title(f'Feature Specificity vs Depth for {dataset_name}', fontsize=14)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{save_dir}/feature_specificity.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Hierarchical Structure Plot
            if dataset_results['hierarchy']:
                plt.figure(figsize=(12, 6))
                n_clusters = []
                avg_sizes = []
                depths = []
                
                for i, h in enumerate(dataset_results['hierarchy']):
                    if h['error'] is None:  # Only include results without errors
                        n_clusters.append(h['n_clusters'])
                        avg_sizes.append(h['avg_cluster_size'])
                        depths.append(i + 1)
                
                if depths:  # Only create plot if we have valid data
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Number of clusters
                    ax1.plot(depths, n_clusters, 'r-o', linewidth=2, markersize=8)
                    ax1.set_xlabel('Depth', fontsize=12)
                    ax1.set_ylabel('Number of Clusters', fontsize=12)
                    ax1.set_title('Cluster Count vs Depth', fontsize=14)
                    ax1.grid(True, alpha=0.3)
                    
                    # Average cluster size
                    ax2.plot(depths, avg_sizes, 'g-o', linewidth=2, markersize=8)
                    ax2.set_xlabel('Depth', fontsize=12)
                    ax2.set_ylabel('Average Cluster Size', fontsize=12)
                    ax2.set_title('Cluster Size vs Depth', fontsize=14)
                    ax2.grid(True, alpha=0.3)
                    
                    plt.suptitle(f'Hierarchical Structure Analysis for {dataset_name}', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(f"{save_dir}/hierarchy_analysis.png", dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 4. Pattern Analysis Plot
            if dataset_results['patterns']:
                plt.figure(figsize=(15, 5))
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                depths = range(1, len(dataset_results['patterns']) + 1)
                
                # Unique patterns
                unique_patterns = [p['unique_patterns'] for p in dataset_results['patterns']]
                ax1.plot(depths, unique_patterns, 'b-o', linewidth=2, markersize=8)
                ax1.set_xlabel('Depth', fontsize=12)
                ax1.set_ylabel('Unique Patterns', fontsize=12)
                ax1.set_title('Pattern Diversity', fontsize=14)
                ax1.grid(True, alpha=0.3)
                
                # Sparsity
                sparsity = [p['sparsity'] for p in dataset_results['patterns']]
                ax2.plot(depths, sparsity, 'r-o', linewidth=2, markersize=8)
                ax2.set_xlabel('Depth', fontsize=12)
                ax2.set_ylabel('Sparsity', fontsize=12)
                ax2.set_title('Activation Sparsity', fontsize=14)
                ax2.grid(True, alpha=0.3)
                
                # Entropy
                entropy = [p['entropy'] for p in dataset_results['patterns']]
                ax3.plot(depths, entropy, 'g-o', linewidth=2, markersize=8)
                ax3.set_xlabel('Depth', fontsize=12)
                ax3.set_ylabel('Entropy', fontsize=12)
                ax3.set_title('Activation Entropy', fontsize=14)
                ax3.grid(True, alpha=0.3)
                
                plt.suptitle(f'Activation Pattern Analysis for {dataset_name}', fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{save_dir}/pattern_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 5. Feature Overlap Plot
            if 'overlaps' in dataset_results and dataset_results['overlaps']:
                plt.figure(figsize=(10, 6))
                mean_overlaps = [o['mean_overlap'] for o in dataset_results['overlaps']]
                std_overlaps = [o['std_overlap'] for o in dataset_results['overlaps']]
                depths = range(1, len(mean_overlaps) + 1)
                
                plt.plot(depths, mean_overlaps, 'p-', label='Mean Overlap', linewidth=2, markersize=8)
                plt.fill_between(depths,
                            [m - s for m, s in zip(mean_overlaps, std_overlaps)],
                            [m + s for m, s in zip(mean_overlaps, std_overlaps)],
                            alpha=0.2)
                plt.xlabel('Depth', fontsize=12)
                plt.ylabel('Feature Overlap', fontsize=12)
                plt.title(f'Feature Overlap Analysis for {dataset_name}', fontsize=14)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{save_dir}/feature_overlap.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 6. Feature Composition Analysis
            if dataset_results['compositions']:
                plt.figure(figsize=(12, 6))
                sparsity = [c['weight_sparsity'] for c in dataset_results['compositions']]
                distribution = [c['weight_distribution'] for c in dataset_results['compositions']]
                depths = range(2, len(sparsity) + 2)  # Start from depth 2
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.plot(depths, sparsity, 'm-o', linewidth=2, markersize=8)
                ax1.set_xlabel('Depth', fontsize=12)
                ax1.set_ylabel('Weight Sparsity', fontsize=12)
                ax1.set_title('Composition Sparsity', fontsize=14)
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(depths, distribution, 'c-o', linewidth=2, markersize=8)
                ax2.set_xlabel('Depth', fontsize=12)
                ax2.set_ylabel('Weight Distribution', fontsize=12)
                ax2.set_title('Composition Distribution', fontsize=14)
                ax2.grid(True, alpha=0.3)
                
                plt.suptitle(f'Feature Composition Analysis for {dataset_name}', fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{save_dir}/feature_composition.png", dpi=300, bbox_inches='tight')
                plt.close()

analyzer = CrossDatasetAnalyzer(['MNIST', 'CIFAR100', 'EMNIST_letter', 'EMNIST'], max_depth=9, prefix=prefix)
results = analyzer.compare_datasets()

import pickle
with open('analysis_results.pkl', 'wb') as f:
    pickle.dump(results, f)

analyzer.plot_metrics_across_depths(results)