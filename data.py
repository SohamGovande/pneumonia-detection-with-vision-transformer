import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from utils import calculate_class_weights_from_directory

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from collections import defaultdict

class_indices = {
    # 'normal': 0,
    'viral': 0,
    'bacterial': 1,
    # 'covid': 3,
}

def compute_accuracies(predictions: torch.Tensor, labels: torch.Tensor):
    if len(class_indices) == 2:
        return {"Exact Match": (predictions.to(torch.int32) == labels.to(torch.int32)).sum().item() / len(labels)}
    features = {
        class_indices['bacterial']: {
            'is_sick': True,
            'is_bacterial': True,
            'is_pneumonia': True,
        },
        class_indices['viral']: {
            'is_sick': True,
            'is_bacterial': False,
            'is_pneumonia': False,
        },
        class_indices['normal']: {
            'is_sick': False,
            'is_bacterial': None,
            'is_pneumonia': None,
        },
        class_indices['covid']: {
            'is_sick': True,
            'is_bacterial': None,
            'is_pneumonia': False,
        },
    }
    
    buildup = defaultdict(lambda: {"total": 0, "correct": 0})

    for prediction, label in zip(predictions, labels):
        real_feature = features[label.item()]
        pred_feature = features[prediction.item()]

        for feature_name, feature_value in real_feature.items():
            if feature_value is not None:
                buildup[feature_name]["total"] += 1
                if feature_value == pred_feature[feature_name]:
                    buildup[feature_name]["correct"] += 1

    accuracies = {}
    
    for feature_name, feature_data in buildup.items():
        accuracies[feature_name] = feature_data["correct"] / feature_data["total"] if feature_data["total"] > 0 else 0


    accuracies['Total'] = (predictions.to(torch.int32) == labels.to(torch.int32)).sum().item() / len(labels)
    return accuracies

def load_image_data(data_dir:Path, image_size, batch_size=1, num_workers=2, transform=None, use_sampler=False, shuffle=False, allowed_classes=['viral', 'bacterial']):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    # Load data directly from the directory, only allowing specified classes
    dataset = ImageFolder(root=data_dir, transform=transform)
    
    # Ensure that the class indices for allowed classes are consecutive
    allowed_class_indices = [class_indices[class_name] for class_name in allowed_classes if class_name in class_indices]
    if sorted(allowed_class_indices) != list(range(min(allowed_class_indices), max(allowed_class_indices) + 1)):
        raise ValueError("The class indices for allowed classes are not consecutive.")
    
    # Filter the dataset to only include allowed classes
    old_allowed_class_indices = [dataset.class_to_idx[class_name] for class_name in allowed_classes if class_name in dataset.class_to_idx]
    old_class_idx_to_new_class_idx = {old_idx: class_indices[class_name] for class_name, old_idx in dataset.class_to_idx.items() if class_name in allowed_classes}
    dataset.samples = [(path, old_class_idx_to_new_class_idx[target]) for path, target in dataset.samples if target in old_allowed_class_indices]
    dataset.targets = [old_class_idx_to_new_class_idx[target] for target in dataset.targets if target in old_allowed_class_indices]
    dataset.class_to_idx = {k: v for k, v in class_indices.items() if k in allowed_classes}
    
    print(f"Dataset class indices: {dataset.class_to_idx}")
    
    if not use_sampler:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    _, class_weights, _ = calculate_class_weights_from_directory(data_dir, allowed_classes=allowed_classes)
    sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )
         

if __name__ == "__main__":
    print('hello world!')
