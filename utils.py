# Vision Transformer (ViT)
import os
import random
import time
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm


def make_cm(ytrue, ypred, title=None):
    # Create a confusion matrix
    cm = confusion_matrix(ytrue, ypred)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the confusion matrix in the first subplot
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_xlabel("Predicted Labels")
    axes[0].set_ylabel("True Labels")
    axes[0].set_title("Confusion Matrix" if title is None else title)

    # Calculate and display performance metrics in the second subplot
    accuracy = accuracy_score(ytrue, ypred)
    precision = precision_score(ytrue, ypred, average='binary')
    recall = recall_score(ytrue, ypred, average='binary')
    f1 = f1_score(ytrue, ypred, average='binary')

    # Set the x and y coordinates for each text label
    x_pos = 0.2  # Adjust these coordinates as needed
    y_pos = 0.6

    # Print the text on the plot without x-label and y-label
    axes[1].text(x_pos, y_pos, f"Accuracy: {accuracy:.4f}", fontsize=12)
    axes[1].text(x_pos, y_pos - 0.1, f"Precision: {precision:.4f}", fontsize=12)
    axes[1].text(x_pos, y_pos - 0.2, f"Recall: {recall:.4f}", fontsize=12)
    axes[1].text(x_pos, y_pos - 0.3, f"F1 Score: {f1:.4f}", fontsize=12)

    # Remove x-label and y-label from the second subplot
    axes[1].axes.get_xaxis().set_visible(False)
    axes[1].axes.get_yaxis().set_visible(False)

    # Adjust the layout of subplots
    plt.tight_layout()

    # Show the plot
    plt.show()



def evaluate_model(model, dataloader, device="cpu"):
    model.to(device)

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            outputs = outputs.squeeze()
            predictions = torch.round(outputs)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return y_true, y_pred



def validate_model(
    model,
    dataloader,
    loss_function,
    device="cpu",
):
    model.to(device)
    print(f"Validating {model.__class__.__name__} on: {device}")

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.inference_mode():
        validation_progress = tqdm(
            dataloader, desc="Validation"
        )

        for batch in validation_progress:
            image, label = batch
            image = image.to(device)
            label = label.to(device)

            outputs, _ = model(image)
            outputs = outputs.squeeze()
            predictions = torch.round(outputs)

            loss = loss_function(outputs, label.to(outputs.dtype))
            
            total_loss += loss.item()
            correct_predictions += (predictions.to(torch.int32) == label.to(torch.int32)).sum().item()
            total_samples += label.size()[0]

            formatted_loss = f"{loss.item():.8f}"
            accuracy = (correct_predictions / total_samples) * 100
            formatted_accuracy = f"{accuracy:.2f}%"

            validation_progress.set_postfix(
                {"Loss": formatted_loss, "Accuracy": formatted_accuracy}
            )

    average_loss = total_loss / len(dataloader)
    accuracy = (correct_predictions / (total_samples + 1e-7)) * 100

    print(f"Validation - Average Loss: {average_loss:.8f} - Accuracy: {accuracy:.2f}%")
    print()




def calculate_class_weights_from_directory(directory_path):
  """
  This function calculates the class weights for a given directory of images.

  Args:
    directory_path (str): The path to the directory of images.

  Returns:
    (list, list, np.ndarray): The class distribution, class weights, and bincount.
  """

  # Create a Path object for the directory
  directory = Path(directory_path)

  # Get a list of subdirectories (classes)
  classes = sorted([d.name for d in directory.iterdir() if d.is_dir()])

  # Initialize class_dist to store class labels
  class_dist = []

  # Iterate through each class directory
  for idx, class_name in enumerate(classes):
    class_path = directory / class_name
    num_files = len(list(class_path.glob('*')))
    class_dist.extend([idx] * num_files)

  # Calculate the class counts using bincount
  bincount = np.bincount(class_dist)

  # Calculate class weights as 1 / bincount for each class
  class_weights = 1.0 / bincount[class_dist]

  return class_dist, class_weights, bincount



def set_seed(seed_value=42):
    """
    Set seed for reproducibility in PyTorch, NumPy, and Python's random library.

    Args:
        seed_value (int): The seed value to set.

    Returns:
        None
    """
    # Set seed for NumPy
    np.random.seed(seed_value)

    # Set seed for Python's random library
    random.seed(seed_value)

    # Set seed for PyTorch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed for CuDNN (if available)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the environment variable for better reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed_value)