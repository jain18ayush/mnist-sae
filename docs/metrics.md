# Meta-SAE Atomicity Analysis: Experimental Design & Mathematical Framework

## Research Question

**Do meta-SAEs continuously become more atomic with increasing depth?**

This document outlines the experimental design and mathematical framework for analyzing whether Sparse Autoencoders (SAEs) trained in a cascaded manner (meta-SAEs) learn progressively more atomic features as depth increases. We present four key metrics that together provide strong evidence regarding this research question.

## Experimental Setup

### Required Components

1. **Base Model**: An MNIST classifier model
2. **Meta-SAE Collection**: A series of SAEs where:
   - SAE at depth 1 is trained on activations from the base model
   - SAE at depth n (n > 1) is trained on latent activations from SAE at depth n-1

3. **Saved Data**:
   - Activation matrices for each depth (including base model)
   - Trained SAE models for each depth
   - MNIST dataset with labels

### Analysis Pipeline

For each metric, we will:
1. Load the saved activations and models
2. Compute the metric for each depth
3. Plot the metric values against depth
4. Test for monotonic increase using Spearman's rank correlation

## Metrics

We propose four complementary metrics that together provide comprehensive evidence about feature atomicity:

### 1. Feature Reconstruction Independence (FRI)

**Intuition**: Truly atomic features should independently contribute to reconstruction without relying on other features.

**Mathematical Framework**:

For each depth $d$ and example $i$:
1. Let $\mathbf{z}_i^d$ be the latent activations at depth $d$
2. Let $\mathbf{a}_i^0$ be the original activations in the base model
3. For each active feature $j$ in $\mathbf{z}_i^d$:
   - Create an isolated representation $\hat{\mathbf{z}}_{i,j}^d$ where only feature $j$ is non-zero
   - Reconstruct back to the base model: $\hat{\mathbf{a}}_{i,j}^0 = g_1(g_2(...g_d(\hat{\mathbf{z}}_{i,j}^d)))$ where $g_d$ is the decoder of SAE at depth $d$
   - Compute the reconstruction error: $E_{i,j} = \|\hat{\mathbf{a}}_{i,j}^0 - \mathbf{a}_i^0\|_2^2$
4. Compute the full reconstruction error: $E_i^{\text{full}} = \|\hat{\mathbf{a}}_i^0 - \mathbf{a}_i^0\|_2^2$ where $\hat{\mathbf{a}}_i^0$ is reconstructed using all features
5. Calculate the relative independence score: $I_{i,j} = \frac{E_{i,j} - E_i^{\text{full}}}{|\text{active features}| - 1}$

The FRI score for depth $d$ is:
$$\text{FRI}(d) = -\frac{1}{N} \sum_{i=1}^{N} \frac{1}{|\text{active features in }\mathbf{z}_i^d|} \sum_{j \in \text{active}} I_{i,j}$$

Higher FRI scores indicate more independent features.

**Implementation**:

```python
def compute_fri(sae_models, activations_by_depth, depth):
    """Compute Feature Reconstruction Independence score at a specific depth."""
    # Get activations for this depth
    depth_activations = activations_by_depth[depth]
    original_activations = activations_by_depth[0]  # Base model activations
    
    # Track independence scores
    independence_scores = []
    
    # For each example (limit to 1000 for efficiency)
    num_examples = min(1000, depth_activations.shape[0])
    
    for example_idx in range(num_examples):
        # Get active features for this example
        example_activations = depth_activations[example_idx:example_idx+1]
        active_indices = torch.where(torch.abs(example_activations) > 1e-5)[1]
        
        if len(active_indices) <= 1:
            continue  # Skip examples with only one active feature
        
        # Baseline full reconstruction
        full_reconstructed = example_activations.clone()
        for d in range(depth, 0, -1):
            full_reconstructed = sae_models[d].decode(full_reconstructed)
        
        full_error = torch.mean((full_reconstructed - original_activations[example_idx:example_idx+1]) ** 2).item()
        
        # Test reconstruction error when each feature is isolated
        feature_errors = []
        
        for feature_idx in active_indices:
            # Create mask with only this feature active
            isolated_activations = torch.zeros_like(example_activations)
            isolated_activations[0, feature_idx] = example_activations[0, feature_idx]
            
            # Reconstruct from isolated feature
            isolated_reconstructed = isolated_activations.clone()
            for d in range(depth, 0, -1):
                isolated_reconstructed = sae_models[d].decode(isolated_reconstructed)
            
            # Compute error
            isolated_error = torch.mean((isolated_reconstructed - original_activations[example_idx:example_idx+1]) ** 2).item()
            
            # Relative independence score (lower is better)
            relative_error = (isolated_error - full_error) / max(len(active_indices) - 1, 1)
            feature_errors.append(relative_error)
        
        # Average feature independence for this example
        if feature_errors:
            independence_scores.append(np.mean(feature_errors))
    
    # Average independence score across all examples (lower is better)
    if independence_scores:
        fri_score = -np.mean(independence_scores)  # Negate to make higher score = better
    else:
        fri_score = float('-inf')
    
    return fri_score
```

### 2. Concept Selectivity Progression (CSP)

**Intuition**: More atomic features should be more selective for specific digit classes rather than activating across multiple classes.

**Mathematical Framework**:

For each depth $d$ and feature $j$:
1. Compute the mean activation for each digit class $c$: $\mu_{j,c}^d = \frac{1}{|C_c|}\sum_{i \in C_c} z_{i,j}^d$ where $C_c$ is the set of examples from class $c$
2. Compute feature selectivity: $S_j^d = \frac{\max_c \mu_{j,c}^d - \text{second}_c \mu_{j,c}^d}{\max_c \mu_{j,c}^d}$ if $\max_c \mu_{j,c}^d > 0$, otherwise $S_j^d = 0$
3. Average across all features: $\text{CSP}(d) = \frac{1}{F_d} \sum_{j=1}^{F_d} S_j^d$ where $F_d$ is the number of features at depth $d$

Higher CSP scores indicate more class-selective features.

**Implementation**:

```python
def compute_csp(activations_by_depth, labels, depth):
    """Compute Concept Selectivity Progression score at a specific depth."""
    # Get activations for this depth
    activations = activations_by_depth[depth]
    
    # Compute class-conditional activation distributions
    num_classes = 10  # For MNIST
    class_activations = [activations[labels == c] for c in range(num_classes)]
    
    # Compute selectivity for each feature
    num_features = activations.shape[1]
    feature_selectivity = torch.zeros(num_features)
    
    for feature_idx in range(num_features):
        # Get mean activation per class for this feature
        class_means = torch.tensor([
            class_act[:, feature_idx].mean() if len(class_act) > 0 else 0.0
            for class_act in class_activations
        ])
        
        # Compute selectivity as (max_class_mean - second_max_class_mean) / max_class_mean
        sorted_means, _ = torch.sort(class_means, descending=True)
        if sorted_means[0] > 0:
            feature_selectivity[feature_idx] = (sorted_means[0] - sorted_means[1]) / sorted_means[0]
        else:
            feature_selectivity[feature_idx] = 0
    
    # Average selectivity across all features
    csp_score = feature_selectivity.mean().item()
    
    return csp_score
```

### 3. Progressive Disentanglement Score (PDS)

**Intuition**: More atomic features should be more statistically independent from each other, showing lower cross-correlation.

**Mathematical Framework**:

For each depth $d$:
1. Compute the normalized covariance matrix: $\Sigma^d_{i,j} = \frac{\text{Cov}(z_i^d, z_j^d)}{\sqrt{\text{Var}(z_i^d) \cdot \text{Var}(z_j^d)}}$
2. Compute the off-diagonal mass: $\text{ODM}(d) = \frac{1}{F_d(F_d-1)} \sum_{i \neq j} |\Sigma^d_{i,j}|$
3. Calculate the progressive improvement: $\text{PDS}(d) = \text{ODM}(d-1) - \text{ODM}(d)$ for $d > 1$

Positive PDS values indicate improving disentanglement at each depth.

**Implementation**:

```python
def compute_pds(activations_by_depth, depth):
    """Compute Progressive Disentanglement Score between depth-1 and depth."""
    if depth <= 1:
        return 0.0  # No previous depth to compare with
    
    # Get activations for current and previous depth
    act_current = activations_by_depth[depth]
    act_prev = activations_by_depth[depth-1]
    
    # Compute normalized covariance matrices
    def compute_normalized_covariance(activations):
        # Center the activations
        centered = activations - activations.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        n_samples = activations.shape[0]
        cov = (centered.T @ centered) / (n_samples - 1)
        
        # Normalize by diagonal elements
        diag = torch.diag(cov)
        outer_sqrt_diag = torch.sqrt(diag).unsqueeze(1) @ torch.sqrt(diag).unsqueeze(0)
        norm_cov = cov / (outer_sqrt_diag + 1e-8)
        
        return norm_cov
    
    norm_cov_current = compute_normalized_covariance(act_current)
    norm_cov_prev = compute_normalized_covariance(act_prev)
    
    # Compute off-diagonal mass
    def compute_off_diagonal_mass(norm_cov):
        n = norm_cov.shape[0]
        mask = torch.ones(n, n) - torch.eye(n)
        return (torch.abs(norm_cov) * mask).sum() / (n * (n-1))
    
    odm_current = compute_off_diagonal_mass(norm_cov_current)
    odm_prev = compute_off_diagonal_mass(norm_cov_prev)
    
    # Progressive disentanglement score (positive means improvement)
    pds_score = odm_prev.item() - odm_current.item()
    
    return pds_score
```

### 4. Latent Traversal Effect Size (LTES)

**Intuition**: Modifying an atomic feature should produce clean, consistent effects on the reconstructed representation.

**Mathematical Framework**:

For each depth $d$ and feature $j$:
1. Create a baseline reconstruction: $\hat{\mathbf{a}}_i^0 = g_1(g_2(...g_d(\mathbf{z}_i^d)))$
2. Create a modified representation: $\tilde{\mathbf{z}}_i^d$ where feature $j$ is scaled by a factor $\alpha$ (e.g., $\alpha = 2$)
3. Reconstruct the modified representation: $\tilde{\mathbf{a}}_i^0 = g_1(g_2(...g_d(\tilde{\mathbf{z}}_i^d)))$
4. Compute the difference: $\mathbf{d}_i^j = \tilde{\mathbf{a}}_i^0 - \hat{\mathbf{a}}_i^0$
5. Measure:
   - Sparsity: $\text{Sp}_j^d = 1 - \frac{1}{N} \sum_{i=1}^N \frac{|\{k : |d_{i,k}^j| > \tau \cdot \sigma_i\}|}{D}$ where $\tau$ is a threshold, $\sigma_i$ is the standard deviation of $\hat{\mathbf{a}}_i^0$, and $D$ is the dimensionality
   - Consistency: $\text{Co}_j^d = \frac{1}{N} \sum_{i=1}^N \frac{\mathbf{d}_i^j \cdot \bar{\mathbf{d}}^j}{|\mathbf{d}_i^j| \cdot |\bar{\mathbf{d}}^j|}$ where $\bar{\mathbf{d}}^j$ is the mean difference vector
6. Combine: $\text{LTES}_j^d = \frac{\text{Sp}_j^d + \text{Co}_j^d}{2}$

The overall LTES score for depth $d$ is the average across all active features:
$$\text{LTES}(d) = \frac{1}{|J_d|} \sum_{j \in J_d} \text{LTES}_j^d$$

where $J_d$ is the set of frequently active features.

**Implementation**:

```python
def compute_ltes(sae_models, activations_by_depth, depth):
    """Compute Latent Traversal Effect Size score at a specific depth."""
    # Skip if this is the base model
    if depth == 0:
        return 0.0
        
    # Get representative activations
    depth_activations = activations_by_depth[depth][:100]  # Use 100 examples for efficiency
    
    # Track effect sizes for all latents in this depth
    effect_sizes = []
    
    # For each potentially active latent
    num_latents = depth_activations.shape[1]
    for latent_idx in range(num_latents):
        # Skip if this latent is rarely active
        if torch.sum(torch.abs(depth_activations[:, latent_idx]) > 1e-5) < 10:
            continue
            
        # Get baseline activations
        baseline_reconstructed = depth_activations.clone()
        for d in range(depth, 0, -1):
            baseline_reconstructed = sae_models[d].decode(baseline_reconstructed)
            
        # Create modified activations with scaled latent
        modified_activations = depth_activations.clone()
        modified_activations[:, latent_idx] *= 2.0  # Double the activation
        
        # Reconstruct modified activations
        modified_reconstructed = modified_activations.clone()
        for d in range(depth, 0, -1):
            modified_reconstructed = sae_models[d].decode(modified_reconstructed)
            
        # Compute difference
        diff = modified_reconstructed - baseline_reconstructed
        
        # Compute effect size measures:
        # 1. Sparsity of effect (what % of dimensions are significantly affected)
        diff_normalized = torch.abs(diff) / (torch.std(baseline_reconstructed, dim=1, keepdim=True) + 1e-8)
        significant_changes = (diff_normalized > 0.5).float().mean(dim=1)
        sparsity = 1.0 - significant_changes.mean().item()  # Higher means more sparse/focused effect
        
        # 2. Consistency of effect (how consistent the direction of change is across examples)
        mean_diff = torch.mean(diff, dim=0)
        cosine_sims = []
        for i in range(len(diff)):
            sim = torch.nn.functional.cosine_similarity(diff[i:i+1], mean_diff.unsqueeze(0), dim=1).item()
            if not np.isnan(sim):
                cosine_sims.append(sim)
        consistency = np.mean(cosine_sims) if cosine_sims else 0  # Higher means more consistent effect
        
        # Combine into a single effect size score (higher is better)
        effect_size = (sparsity + consistency) / 2
        effect_sizes.append(effect_size)
    
    # Average effect size across all active latents
    ltes_score = np.mean(effect_sizes) if effect_sizes else 0
    
    return ltes_score
```

## Main Analysis Function

The following code combines all four metrics into a single analysis function:

```python
def analyze_meta_sae_atomicity(sae_models, activations_by_depth, labels, num_depths):
    """
    Analyze whether meta-SAEs become more atomic with depth.
    
    Args:
        sae_models: Dictionary mapping depths to SAE models
        activations_by_depth: Dictionary mapping depths to activation matrices
        labels: MNIST labels corresponding to examples
        num_depths: Number of depths to analyze
        
    Returns:
        results: Dictionary of results for each metric and depth
    """
    # Initialize results dictionary
    results = {
        'FRI': {},
        'CSP': {},
        'PDS': {},
        'LTES': {}
    }
    
    # Compute metrics for each depth
    for depth in range(1, num_depths):
        print(f"Analyzing depth {depth}...")
        
        # Feature Reconstruction Independence
        results['FRI'][depth] = compute_fri(sae_models, activations_by_depth, depth)
        
        # Concept Selectivity Progression
        results['CSP'][depth] = compute_csp(activations_by_depth, labels, depth)
        
        # Progressive Disentanglement Score
        results['PDS'][depth] = compute_pds(activations_by_depth, depth)
        
        # Latent Traversal Effect Size
        results['LTES'][depth] = compute_ltes(sae_models, activations_by_depth, depth)
    
    # Test for monotonic increase using Spearman's rank correlation
    correlations = {}
    for metric in results:
        depths = list(results[metric].keys())
        scores = [results[metric][d] for d in depths]
        rho, p_value = spearmanr(depths, scores)
        correlations[metric] = {
            'rho': rho,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Add correlations to results
    results['correlations'] = correlations
    
    return results
```

## Visualization Functions

```python
def plot_atomicity_metrics(results):
    """Plot all four atomicity metrics against depth."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = list(results.keys())
    metrics.remove('correlations')
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        depths = list(results[metric].keys())
        scores = [results[metric][d] for d in depths]
        
        ax.plot(depths, scores, 'o-', linewidth=2, markersize=8)
        ax.set_title(f"{metric} vs Depth")
        ax.set_xlabel("Depth")
        ax.set_ylabel(f"{metric} Score")
        
        # Add correlation information
        rho = results['correlations'][metric]['rho']
        p_val = results['correlations'][metric]['p_value']
        ax.annotate(f"ρ = {rho:.3f} (p = {p_val:.3f})", 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   ha='left', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        
        # Add trend line
        z = np.polyfit(depths, scores, 1)
        p = np.poly1d(z)
        ax.plot(depths, p(depths), "--", color='red', alpha=0.7)
    
    plt.tight_layout()
    return fig
```

## Expected Outcomes

If meta-SAEs do continuously become more atomic with depth, we expect:

1. **FRI scores** to increase with depth, indicating features contribute more independently to reconstruction
2. **CSP scores** to increase with depth, indicating features become more selective for specific digit classes
3. **PDS scores** to be consistently positive, indicating improved disentanglement at each depth
4. **LTES scores** to increase with depth, indicating cleaner, more consistent feature effects

The strength of our conclusion will depend on how many of these metrics show the expected pattern and the statistical significance of their trends.

## Significance Testing

For each metric, we will:
1. Compute Spearman's rank correlation between depth and the metric score
2. Test the null hypothesis that there is no monotonic relationship
3. Consider p < 0.05 as evidence for rejecting the null hypothesis

Additionally, we will compare the observed metrics to appropriate null models (e.g., randomly initialized SAEs) to ensure the patterns we observe are not artifacts of the network architecture or analysis method.

## Conclusion

This experimental design provides a comprehensive framework for investigating whether meta-SAEs continuously become more atomic with depth. By using four complementary metrics that capture different aspects of atomicity, we can build a strong case for or against the hypothesis, giving us confidence in our findings regardless of the outcome.