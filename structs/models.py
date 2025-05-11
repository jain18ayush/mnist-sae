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

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # This is specifically modified for CIFAR: 
        # Smaller initial conv with 3x3 kernel instead of 7x7
        # No initial max pooling to preserve spatial information
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, cache_activations=False):
        # If caching is enabled, we'll store activations
        activations = {}
        
        out = torch.relu(self.bn1(self.conv1(x)))
        if cache_activations:
            activations['conv1'] = out.detach().clone()
            
        out = self.layer1(out)
        if cache_activations:
            activations['layer1'] = out.detach().clone()
            
        out = self.layer2(out)
        if cache_activations:
            activations['layer2'] = out.detach().clone()
            
        out = self.layer3(out)
        if cache_activations:
            activations['layer3'] = out.detach().clone()
            
        out = self.layer4(out)
        if cache_activations:
            activations['layer4'] = out.detach().clone()
            
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        fc_features = out
        if cache_activations:
            activations['fc_features'] = fc_features.detach().clone()
            
        out = self.linear(out)
        if cache_activations:
            activations['output'] = out.detach().clone()
            
        if cache_activations:
            return out, activations
        return out

# Create ResNet18 model
def ResNet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
#* This is the model for CIFAR100 dataset

class CIFAR100Model(nn.Module):
    def __init__(self):
        super(CIFAR100Model, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)  # After 3 pooling operations: 32x32 -> 4x4
        self.fc2 = nn.Linear(1024, 512)         # Hidden layer good for SAE analysis
        self.fc3 = nn.Linear(512, 100)          # 100 classes for CIFAR100
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Initialize a cache to store activations
        self.activation_cache = {
            'conv1': [],
            'conv2': [],
            'conv3': [],
            'fc1': [],
            'fc2': [],
            'fc3': []
        }

    def forward(self, x, cache_activations=True):
        # Convolutional layers with ReLU, batch norm, and pooling
        conv1_out = self.pool(F.relu(self.bn1(self.conv1(x))))
        if cache_activations:
            self.activation_cache['conv1'].append(conv1_out.detach().clone())
        
        conv2_out = self.pool(F.relu(self.bn2(self.conv2(conv1_out))))
        if cache_activations:
            self.activation_cache['conv2'].append(conv2_out.detach().clone())
        
        conv3_out = self.pool(F.relu(self.bn3(self.conv3(conv2_out))))
        if cache_activations:
            self.activation_cache['conv3'].append(conv3_out.detach().clone())
        
        # Flatten the output for the fully connected layer
        flattened = conv3_out.view(-1, 256 * 4 * 4)
        
        # Fully connected layers
        fc1_out = F.relu(self.fc1(flattened))
        if cache_activations:
            self.activation_cache['fc1'].append(fc1_out.detach().clone())
        
        fc1_out = self.dropout(fc1_out)
        
        fc2_out = F.relu(self.fc2(fc1_out))
        if cache_activations:
            self.activation_cache['fc2'].append(fc2_out.detach().clone())
        
        fc2_out = self.dropout(fc2_out)
        
        fc3_out = self.fc3(fc2_out)
        if cache_activations:
            self.activation_cache['fc3'].append(fc3_out.detach().clone())
        
        return fc3_out

    def get_cached_activations(self, layer_name):
        """Return all cached activations for a specific layer"""
        if layer_name not in self.activation_cache or not self.activation_cache[layer_name]:
            return None
        return torch.cat(self.activation_cache[layer_name])

    def clear_cache(self):
        """Clear all cached activations"""
        self.activation_cache = {
            'conv1': [],
            'conv2': [],
            'conv3': [],
            'fc1': [],
            'fc2': [],
            'fc3': []
        }

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

    def forward(self, x, cache_activations=True):
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
