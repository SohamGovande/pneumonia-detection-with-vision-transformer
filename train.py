import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from funcyou.utils import DotDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_data  # Import data-related functions
from model import VisionTransformer  # Import your VIT model class
from utils import evaluate_model, validate_model


# Training function
def train_model(
    model,
    dataloader,
    optimizer,
    loss_function,
    num_epochs=10,
    device="cpu",
    data_percent=1.0,
    steps_per_epoch=None,
    save_on_every_n_epochs=5,
    model_path=None,
):
    model.to(device)
    print(f"{model.__class__.__name__} Running on: {device}")

    data_size = int(data_percent * len(dataloader)) if steps_per_epoch is None else steps_per_epoch

    Path(model_path).parent.mkdir(exist_ok=True)
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct_predictions = 0
        total_samples = 0

        epoch_progress = tqdm(
            dataloader, desc=f"Epoch [{epoch + 1:2}/{num_epochs:2}]"
        )
        
        last_update_time = time.time() - 1.0  

        for j, batch in enumerate(epoch_progress):
            image, label = batch
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            outputs, _ = model(image)
            outputs = outputs.squeeze()
            predictions = torch.round(outputs)

            loss = loss_function(outputs, label.to(outputs.dtype))
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()
            total_correct_predictions += (predictions.to(torch.int32) == label.to(torch.int32)).sum().item()
            total_samples += label.size()[0]

            formatted_loss = f"{loss.item():.8f}"
            accuracy = (total_correct_predictions / total_samples) * 100
            formatted_accuracy = f"{accuracy:.2f}%"
            
            
            current_time = time.time()
            if current_time - last_update_time > epoch_progress.mininterval:
                epoch_progress.set_postfix(
                    {"Loss": formatted_loss, "Accuracy": formatted_accuracy}
                )
                last_update_time = current_time

            if steps_per_epoch is not None and j + 1 >= steps_per_epoch:
                break
        
        average_loss = total_loss / data_size
        average_accuracy = (total_correct_predictions / (total_samples + 1e-7)) * 100

        print(
            f"\nEpoch [{epoch + 1:2}/{num_epochs:2}] - Average Loss: {average_loss:.8f} - Average Accuracy: {average_accuracy:.2f}%"
        )
        print()

        if (epoch+1) % save_on_every_n_epochs == 0 and model_path is not None:
            torch.save(model.state_dict(), model_path)


# Main training function
def main():
    config = DotDict.from_toml('config.toml')  # Load configuration
    config.device= 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if config.test:
        train_dataloader, test_dataloader = load_data(config.data_dir, config.image_size, config.batch_size, num_workers=2, test=config.test)
    else:
        train_dataloader = load_data(config.data_dir, config.image_size, config.batch_size, num_workers=2, test=config.test)
        
    # Initialize your VIT model and optimizer
    model = VisionTransformer(config)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Optional: Load pretrained model weights
    model_path = Path(config.model_path)
    model_path.parent.mkdir(exist_ok=True)
    
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded pretrained model weights from {model_path}")
    except FileNotFoundError:
        print('No saved model weights found.')
    except Exception as e:
        raise e
    finally:
        # Train the model
        train_model(model, train_dataloader, optimizer, loss_function, num_epochs=config.num_epochs, device=config.device, model_path=model_path)
        torch.save(model.state_dict(), config.model_path)
        
    if config.test:
        # validate on test data
        validate_model(model, test_dataloader, loss_function, config.device)
        
        # plotting
        y_true, y_pred = evaluate_model(model, test_dataloader, device=config.device)
        


if __name__ == "__main__":
    main()
