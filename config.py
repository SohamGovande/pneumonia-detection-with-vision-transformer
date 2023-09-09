
from pathlib import Path

from funcyou.utils import DotDict

data_dir = Path('./dataset')
train_dir = data_dir/'train'
test_dir = data_dir/'test'
val_dir = data_dir/'val'


# Create a DotDict instance and initialize it with your configuration
config = DotDict()
#data configurations
config.image_size = 512  # should be mutiple of patch_size
config.batch_size = 8
config.device = 'cuda'
config.test = True
#model configurations
config.num_layers = 4
config.resnet_layers = 2
config.hidden_dim = 120  # should be mutiple of num_heads
config.mlp_dim = 2048
config.num_heads = 12
config.dropout_rate = 0.1
config.patch_size = 32   # should be mutiple of 8
config.num_patches = int(config.image_size**2 / config.patch_size**2)
config.num_channels = 3
config.patching_elements = (config.num_channels*config.image_size**2 )//config.num_patches
config.final_resnet_output_dim = 2048
config.num_classes = 2
config.learning_rate = 2e-4

config.model_path = 'models/vit2.pth'
config.data_dir = 'dataset/'

config.num_epochs = 10

config.save_toml('config.toml')
