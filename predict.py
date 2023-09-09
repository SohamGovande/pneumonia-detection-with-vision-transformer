# predict.py
import argparse
import os

import numpy as np
import pandas as pd
import torch
from funcyou.utils import DotDict
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data import ImageFolderWithPaths
from model import VisionTransformer  # Import your model class


def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image

def predict_directory(directory_path, model, transform, batch_size=8, device='cpu'):
    model.eval()

    image_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]

    dataset = ImageFolderWithPaths(directory_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    predictions = []
    confidence_scores = []
    image_paths_list = []

    with torch.no_grad():
        for images, _, paths in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs, _ = model(images)
            outputs = torch.sigmoid(outputs).cpu().numpy()

            predictions.extend(np.round(outputs))
            confidence_scores.extend(outputs.max(axis=1))
            image_paths_list.extend(paths)

    return pd.DataFrame(
        {
            "Image_Path": image_paths_list,
            "Predicted_Class": predictions,
            "Confidence_Score": confidence_scores,
        }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pneumonia Detection using Vision Transformer')
    parser.add_argument('image_directory', type=str, help='Path to the directory containing test images')
    args = parser.parse_args()

    config = DotDict.from_toml('config.toml')
    model = VisionTransformer(config)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the trained model weights
    model.load_state_dict(torch.load(config.model_path))
    model.to(config.device)
    model.eval()
    
    # Define transformations for test images (should be the same as used during training)
    test_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor()
    ])
    
    image_directory = Path(args.image_directory) 
    
    result_df = predict_directory(image_directory, model, test_transform)
    
    print(result_df)  # Display the first few rows of the DataFrame
    result_df.to_csv('output.csv')
    print('Output saved in output.csv!')