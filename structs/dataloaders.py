import numpy as np
from torch.utils.data import Dataset

class dSpritesDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file, allow_pickle=True, encoding='latin1')
        self.images = data['imgs']  # 2D images (64x64)
        self.latents_values = data['latents_values']  # Latent variables
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)
            
        return img, idx  # You can return `idx` as a placeholder label for now
