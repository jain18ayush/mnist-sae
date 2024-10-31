import torch
import torch.nn as nn
import torch.optim as optim
import math 
import torch.nn.functional as F

#* This is the transform for mnist dataset to coloredMnist (ie add 3 channel from 1)
# # Transform MNIST to RGB
# class ToRGB:
#     def __call__(self, img):
#         return img.repeat(3, 1, 1)  # Repeat the grayscale channel 3 times

# # Updated transforms for colored MNIST
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     ToRGB(),  # Convert to RGB
#     transforms.Normalize(mean=[0.1307, 0.1307, 0.1307],  # Same normalization for each channel
#                        std=[0.3081, 0.3081, 0.3081])
# ])

class ColoredMNISTModel(nn.Module):
    def __init__(self):
        super(ColoredMNISTModel, self).__init__()
        self.fc1 = nn.Linear(3 * 28 * 28, 256)  # Input layer (3x28x28 pixels for RGB)
        self.fc2 = nn.Linear(256, 128)          # Larger hidden layer
        self.fc3 = nn.Linear(128, 10)           # Output layer (10 classes)

        # Initialize a cache to store lists of activations
        self.activation_cache = {
            'fc1': [],
            'fc2': [],
            'fc3': []
        }

    def forward(self, x, cache_activations=True):
        x = x.view(-1, 3 * 28 * 28)  # Flatten the input image (accounting for 3 channels)
        
        # Pass through fc1 and cache activations if needed
        fc1_out = torch.relu(self.fc1(x))
        if cache_activations:
            self.activation_cache['fc1'].append(fc1_out.detach().clone())
        
        # Pass through fc2 and cache activations if needed
        fc2_out = torch.relu(self.fc2(fc1_out))
        if cache_activations:
            self.activation_cache['fc2'].append(fc2_out.detach().clone())
        
        # Pass through fc3 and cache activations if needed
        fc3_out = self.fc3(fc2_out)
        if cache_activations:
            self.activation_cache['fc3'].append(fc3_out.detach().clone())

        return fc3_out

    def get_cached_activations(self, layer_name):
        return torch.cat(self.activation_cache[layer_name]) if layer_name in self.activation_cache else None

    def clear_cache(self):
        self.activation_cache = {
            'fc1': [],
            'fc2': [],
            'fc3': []
        }

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (28x28 pixels)
        self.fc2 = nn.Linear(128, 64)       # Hidden layer
        self.fc3 = nn.Linear(64, 10)        # Output layer (10 classes)

        # Initialize a cache to store lists of activations
        self.activation_cache = {
            'fc1': [],
            'fc2': [],
            'fc3': []
        }

    def forward(self, x, cache_activations=True):
        print(x.shape)
        x = x.view(-1, 28 * 28)  # Flatten the input image
        
        # Pass through fc1 and cache activations if needed
        fc1_out = torch.relu(self.fc1(x))
        if cache_activations:
            # Split batch into individual samples and extend the list
            individual_samples = fc1_out.detach().clone().split(1)  # Split into tensors of size 1
            self.activation_cache['fc1'].extend(individual_samples)  # Add each sample separately

        
        # Pass through fc2 and cache activations if needed
        fc2_out = torch.relu(self.fc2(fc1_out))
        if cache_activations:
            self.activation_cache['fc2'].append(fc2_out.detach().clone())  # Append fc2 activations
        
        # Pass through fc3 and cache activations if needed
        fc3_out = self.fc3(fc2_out)
        if cache_activations:
            self.activation_cache['fc3'].append(fc3_out.detach().clone())  # Append fc3 activations

        return fc3_out

    # Method to retrieve cached activations for a specified layer
    def get_cached_activations(self, layer_name):
        return torch.cat(self.activation_cache[layer_name]) if layer_name in self.activation_cache else None

    # Method to clear the cache
    def clear_cache(self):
        self.activation_cache = {
            'fc1': [],
            'fc2': [],
            'fc3': []
        }

class SimpleSAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, l1_coeff=0.1, seed=42):
        super(SimpleSAE, self).__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.l1_coeff = l1_coeff  # L1 regularization coefficient for sparsity
        
        # Initialize a cache to store lists of activations
        self.activation_cache = {
            'encoder': [],
            'decoder': []
        }

    def forward(self, x, cache_activations=False):
        # Encoder: Reduce the dimensionality
        encoded = torch.relu(self.encoder(x))
        if cache_activations:
            self.activation_cache['encoder'].append(encoded.detach().clone())  # Append encoder activations

        # Decoder: Reconstruct the original input
        decoded = self.decoder(encoded)
        if cache_activations:
            self.activation_cache['decoder'].append(decoded.detach().clone())  # Append decoder activations

        return encoded, decoded

    def compute_loss(self, x, decoded, encoded):
        # Reconstruction Loss (MSE)
        recon_loss = nn.MSELoss()(decoded, x)

        # L1 Sparsity Loss (L1 regularization on encoded activations)
        l1_loss = self.l1_coeff * torch.sum(torch.abs(encoded))

        # Combine losses: L = MSE + λ * L1
        loss = recon_loss + l1_loss

        return loss

    # Method to retrieve cached activations for a specified layer
    def get_cached_activations(self, layer_name):
        return torch.cat(self.activation_cache[layer_name]) if layer_name in self.activation_cache else None

    # Method to clear the cache
    def clear_cache(self):
        self.activation_cache = {
            'encoder': [],
            'decoder': []
        }

class EnhancedSAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, l1_coeff=0.1, seed=42):
        super(EnhancedSAE, self).__init__()
        self.l1_coeff = l1_coeff  # L1 regularization coefficient for sparsity
        torch.manual_seed(seed)

        # Encoder and decoder with Kaiming initialization
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        self.encoder.bias.data.zero_()
        self.decoder.bias.data.zero_()

        self.activation_cache = {
            'encoder': [],
            'decoder': []
        }


    def forward(self, x, cache_activations=True):
        # Center the input
        x_centered = x - self.decoder.bias
        # Encoder: Reduce the dimensionality
        encoded = F.relu(self.encoder(x_centered))
        
        if cache_activations:
            self.activation_cache['encoder'].append(encoded.detach().clone())  # Append encoder activations

        # Decoder: Reconstruct the original input
        decoded = self.decoder(encoded)
        if cache_activations:
            self.activation_cache['decoder'].append(decoded.detach().clone())  # Append decoder activations

        return encoded, decoded

    def compute_loss(self, x, decoded, encoded):
        # Reconstruction Loss (MSE)
        recon_loss = nn.MSELoss()(decoded, x)

        # L1 Sparsity Loss (L1 regularization on encoded activations)
        l1_loss = self.l1_coeff * torch.sum(torch.abs(encoded))

        # Combine losses: L = MSE + λ * L1
        loss = recon_loss + l1_loss

        return loss

    def get_cached_activations(self, layer_name):
        return torch.cat(self.activation_cache[layer_name]) if layer_name in self.activation_cache else None


    def clear_cache(self):
        self.activation_cache = {
            'encoder': [],
            'decoder': []
        }
