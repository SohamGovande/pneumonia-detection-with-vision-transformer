import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from utils import calculate_class_weights_from_directory


def load_data(data_dir, image_size, batch_size, num_workers, test=False):
    # Define data augmentation transformations for X-ray images

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.3)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(), 
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor()
    ])

    # Train data
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)

    class_dist, class_weights, bincount = calculate_class_weights_from_directory(train_dir)
    train_sampler = torch.utils.data.WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler, num_workers=2)

    if test:
        # Test data
        test_dataset = ImageFolder(root=test_dir, transform=test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_dataloader, test_dataloader
    else:
        return train_dataloader
    
def get_class_mapping(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    classes = sorted(os.listdir(train_dir))
    return {cls_name: idx for idx, cls_name in enumerate(classes)}

class ImageFolderWithPaths(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        super(ImageFolderWithPaths, self).__init__(root, transform, target_transform, loader)

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path, _ = self.samples[index]
        return (original_tuple + (path,))



if __name__ == "__main__":
    print('hello world!')
