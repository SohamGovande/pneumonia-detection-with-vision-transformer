# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st
import torch
from funcyou.utils import DotDict
from PIL import Image
from torchvision import transforms

from model import VisionTransformer  # Import your model class


def load_and_preprocess_image(image, transform):
    image = Image.open(image).convert("RGB")
    image = transform(image)
    return image

def predict_image(image, model, transform, device='cpu'):
    model.eval()
    image = image.to(device)
    outputs, attention_weights = model(image.unsqueeze(0))
    prediction = torch.sigmoid(outputs).cpu().detach().numpy()
    attention_weights = [aw.squeeze().cpu().detach().numpy() for aw in attention_weights]  # Convert attention weights to NumPy arrays
    return prediction, attention_weights

def main():
    st.title("Pneumonia Detection using Vision Transformer")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        config = DotDict.from_toml('config.toml')
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = VisionTransformer(config)
        model.load_state_dict(torch.load(config.model_path))
        model.to(config.device)
        model.eval()

        test_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor()
        ])

        preprocessed_image = load_and_preprocess_image(uploaded_image, test_transform)
        prediction, attention_weights = predict_image(preprocessed_image, model, test_transform, config.device)

        # Display prediction
        st.write("Prediction:")
        st.write(f"Pneumonia Probability: {prediction[0][0]:.2%}")


if __name__ == "__main__":
    main()
