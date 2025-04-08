import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import torch
import math

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def visualize_2d_vectors():
    """Visualize weight vectors in 2D and their angles"""
    # Create some 2D weight vectors
    vectors = np.array([
        [1.0, 0.0],    # Vector pointing along x-axis
        [0.0, 1.0],    # Vector pointing along y-axis (orthogonal to first)
        [0.7, 0.7],    # Vector at 45 degrees
        [-0.5, 0.5]    # Vector at 135 degrees
    ])
    
    # Compute pairwise cosine similarities
    norms = np.sqrt(np.sum(vectors**2, axis=1, keepdims=True))
    normalized = vectors / norms
    cosine_similarities = np.dot(normalized, normalized.T)
    angles_degrees = np.arccos(np.clip(cosine_similarities, -1.0, 1.0)) * 180 / np.pi
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    
    # Plot vectors
    colors = ['blue', 'red', 'green', 'purple']
    labels = ['Vector 1', 'Vector 2', 'Vector 3', 'Vector 4']
    
    for i, (vec, color, label) in enumerate(zip(vectors, colors, labels)):
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.1, fc=color, ec=color, label=label)
    
    # Add angle annotations for some pairs
    # Vector 1 and 2 (orthogonal)
    ax.annotate(f"90°", xy=(0.3, 0.3), xytext=(0.4, 0.4), 
                arrowprops=dict(arrowstyle="->", color='black'))
    
    # Vector 1 and 3
    ax.annotate(f"45°", xy=(0.85, 0.35), xytext=(0.9, 0.5), 
                arrowprops=dict(arrowstyle="->", color='black'))
    
    # Show angle matrix
    angle_table = ""
    for i, row_label in enumerate(labels):
        angle_table += f"{row_label:<10}"
        for j, col_label in enumerate(labels):
            angle_table += f"{angles_degrees[i,j]:6.1f}° "
        angle_table += "\n"
    
    ax.text(1.1, 0.5, "Angle Matrix (degrees):\n" + angle_table, 
            transform=ax.transAxes, fontfamily='monospace')
    
    # Set plot limits and labels
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Weight Vectors and Their Angles in 2D')
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def visualize_3d_vectors():
    """Visualize weight vectors in 3D and their angles"""
    # Create some 3D weight vectors
    vectors = np.array([
        [1.0, 0.0, 0.0],    # Vector pointing along x-axis
        [0.0, 1.0, 0.0],    # Vector pointing along y-axis
        [0.0, 0.0, 1.0],    # Vector pointing along z-axis
        [0.7, 0.7, 0.0],    # Vector in xy-plane
        [0.6, 0.0, 0.8]     # Vector in xz-plane
    ])
    
    # Compute pairwise cosine similarities
    norms = np.sqrt(np.sum(vectors**2, axis=1, keepdims=True))
    normalized = vectors / norms
    cosine_similarities = np.dot(normalized, normalized.T)
    angles_degrees = np.arccos(np.clip(cosine_similarities, -1.0, 1.0)) * 180 / np.pi
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vectors
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    labels = ['Vector 1', 'Vector 2', 'Vector 3', 'Vector 4', 'Vector 5']
    
    for i, (vec, color, label) in enumerate(zip(vectors, colors, labels)):
        a = Arrow3D([0, vec[0]], [0, vec[1]], [0, vec[2]], 
                    mutation_scale=20, lw=2, arrowstyle="-|>", color=color, label=label)
        ax.add_artist(a)
        # Add text label near arrow tip
        ax.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, label, color=color)
    
    # Show angle matrix (in plot title instead of text because 3D plot)
    angle_table = "Angle Matrix (degrees):\n"
    for i, row_label in enumerate(labels):
        angle_table += f"{row_label:<10}"
        for j, col_label in enumerate(labels):
            angle_table += f"{angles_degrees[i,j]:6.1f}° "
        angle_table += "\n"
    
    fig.text(0.5, 0.01, angle_table, ha='center', fontfamily='monospace')
    
    # Set plot limits and labels
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('Weight Vectors and Their Angles in 3D')
    
    # Create a custom legend
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot([0], [0], [0], color=color, label=label)
    ax.legend()
    
    return fig

def visualize_high_dim_projection():
    """Visualize a projection of high-dimensional weight vectors"""
    # Create some higher-dimensional vectors (e.g., 10D)
    dim = 10
    np.random.seed(42)  # For reproducibility
    
    # Create vectors with different characteristics
    vectors = []
    
    # 1. Orthogonal vectors
    v1 = np.zeros(dim)
    v1[0] = 1.0
    
    v2 = np.zeros(dim)
    v2[1] = 1.0
    
    # 2. Somewhat aligned vectors
    v3 = np.random.randn(dim)
    v3 = v3 / np.linalg.norm(v3)
    
    v4 = v3 * 0.8 + np.random.randn(dim) * 0.2
    v4 = v4 / np.linalg.norm(v4)
    
    # 3. Anti-aligned vector
    v5 = -v3 * 0.9 + np.random.randn(dim) * 0.1
    v5 = v5 / np.linalg.norm(v5)
    
    vectors = np.vstack([v1, v2, v3, v4, v5])
    
    # Compute pairwise cosine similarities
    cosine_similarities = np.dot(vectors, vectors.T)
    angles_degrees = np.arccos(np.clip(cosine_similarities, -1.0, 1.0)) * 180 / np.pi
    
    # Use PCA to project to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    
    # Plot vectors
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    labels = ['Orthogonal 1', 'Orthogonal 2', 'Random', 'Similar to 3', 'Opposite to 3']
    
    for i, (vec, color, label) in enumerate(zip(vectors_2d, colors, labels)):
        # Scale to unit length in the 2D space
        vec = vec / np.linalg.norm(vec)
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.1, fc=color, ec=color, label=label)
    
    # Show angle matrix
    angle_table = ""
    for i, row_label in enumerate(labels):
        angle_table += f"{row_label:<12}"
        for j, col_label in enumerate(labels):
            angle_table += f"{angles_degrees[i,j]:6.1f}° "
        angle_table += "\n"
    
    ax.text(1.1, 0.5, "Original High-Dim Angle Matrix:\n" + angle_table, 
            transform=ax.transAxes, fontfamily='monospace')
    
    # Set plot limits and labels
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')
    ax.set_title('PCA Projection of High-Dimensional Weight Vectors')
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def visualize_sae_weights_simplified():
    """Create a simplified visualization of SAE weight matrix angles"""
    # Create a simplified SAE weight matrix (encoder weights)
    # In real SAEs, this would be much larger, e.g., [2304, 128] for your first layer
    np.random.seed(42)
    
    # For depth 1: less orthogonal weights
    hidden_dim1, input_dim1 = 6, 4
    weights1 = np.random.randn(hidden_dim1, input_dim1)
    # Make them less orthogonal by adding a common component
    common = np.random.randn(1, input_dim1)
    weights1 = weights1 + np.tile(common, (hidden_dim1, 1)) * 1.0
    
    # For depth 5: more orthogonal weights
    hidden_dim2, input_dim2 = 6, 4
    weights2 = np.random.randn(hidden_dim2, input_dim2)
    # Just keep them as random vectors - they'll be more orthogonal
    
    # Normalize rows to unit length
    weights1 = weights1 / np.sqrt(np.sum(weights1**2, axis=1, keepdims=True))
    weights2 = weights2 / np.sqrt(np.sum(weights2**2, axis=1, keepdims=True))
    
    # Compute pairwise cosine similarities
    cosine_sim1 = np.dot(weights1, weights1.T)
    angles1 = np.arccos(np.clip(cosine_sim1, -1.0, 1.0)) * 180 / np.pi
    
    cosine_sim2 = np.dot(weights2, weights2.T)
    angles2 = np.arccos(np.clip(cosine_sim2, -1.0, 1.0)) * 180 / np.pi
    
    # Create a figure for the side-by-side angle matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot the angle matrices
    im1 = ax1.imshow(angles1, cmap='viridis')
    ax1.set_title('Angles Between Weight Vectors (Depth 1)\nLess Orthogonal')
    fig.colorbar(im1, ax=ax1, label='Degrees')
    ax1.set_xlabel('Neuron Index')
    ax1.set_ylabel('Neuron Index')
    
    im2 = ax2.imshow(angles2, cmap='viridis')
    ax2.set_title('Angles Between Weight Vectors (Depth 5)\nMore Orthogonal')
    fig.colorbar(im2, ax=ax2, label='Degrees')
    ax2.set_xlabel('Neuron Index')
    ax2.set_ylabel('Neuron Index')
    
    # Add some statistics as text
    np.fill_diagonal(angles1, np.nan)  # Ignore self-similarities
    np.fill_diagonal(angles2, np.nan)
    
    mean1 = np.nanmean(angles1)
    median1 = np.nanmedian(angles1)
    
    mean2 = np.nanmean(angles2)
    median2 = np.nanmedian(angles2)
    
    stats_text = f"""
    Depth 1 Statistics:
    Mean Angle: {mean1:.2f}°
    Median Angle: {median1:.2f}°
    
    Depth 5 Statistics:
    Mean Angle: {mean2:.2f}°
    Median Angle: {median2:.2f}°
    
    As depth increases:
    - Mean angle increases by {mean2-mean1:.2f}°
    - Angles closer to 90° suggest more orthogonality
    - More orthogonal = more monosemantic features
    """
    
    fig.text(0.5, 0.01, stats_text, ha='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for the text
    
    return fig

# Create all visualizations
fig1 = visualize_2d_vectors()
fig2 = visualize_3d_vectors()
fig3 = visualize_high_dim_projection()
fig4 = visualize_sae_weights_simplified()

# Display them
plt.show()