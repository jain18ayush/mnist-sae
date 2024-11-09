from PIL import Image

def convert_to_rgb(img):
    return img.convert("RGB")

class LayerConfig:
    def __init__(self, name, input_dim):
        self.name = name
        self.input_dim = input_dim

