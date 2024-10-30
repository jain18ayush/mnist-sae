import numpy as np
from torch.utils.data import Dataset
import torch 


class dSpritesDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file, allow_pickle=True, encoding='latin1')
        self.images = data['imgs']  # 2D images (64x64)
        self.latents_values = data['latents_values']  # Latent variables
        self.latents_classes = data['latents_classes']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = torch.tensor(self.latents_classes[idx][1], dtype=torch.long) #shape of the image

        if self.transform:
            img = self.transform(img)
            
        return img, label  # You can return `idx` as a placeholder label for now

    def get_data_by_label(self, label):
        filtered_images = [self.transform(img) for img, latents in zip(self.images, self.latents_classes) if latents[1] == label]
        filtered_labels = [label] * len(filtered_images)
        return list(zip(filtered_images, filtered_labels))  # Return as a list of (img, label)
