from PIL import Image

def convert_to_rgb(img):
    return img.convert("RGB")

class LayerConfig:
    def __init__(self, name, input_dim):
        self.name = name
        self.input_dim = input_dim
        
# Create instances for each layer
fc1_config = LayerConfig('fc1', 256)
fc2_config = LayerConfig('fc2', 128)
fc3_config = LayerConfig('fc3', 10)
encoder_config = LayerConfig('encoder', 2304)
decoder_config = LayerConfig('decoder', 128)