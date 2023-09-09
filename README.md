# Vision Transformer for Pneumonia Detection

## Introduction

This project implements a Vision Transformer (ViT) for the detection of pneumonia in chest X-ray images. The ViT model is a state-of-the-art deep learning architecture that has shown promising results in various computer vision tasks, including image classification.

## Dataset

The dataset used for this project consists of chest X-ray images containing both pneumonia and non-pneumonia cases. The dataset is divided into training, validation, and test sets, enabling the development and evaluation of the model's performance.

Dataset here [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/lasaljaywardena/pneumonia-chest-x-ray-dataset)

## Model

The Vision Transformer (ViT) architecture is employed to classify chest X-ray images into pneumonia or non-pneumonia cases. The model is trained to extract meaningful features from the images and make predictions based on these features.

## Getting Started

Follow the steps below to get started with this project:

### Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- Python 3.11
- PyTorch
- torchvision
- NumPy
- pandas
- Streamlit (for running the web app)

### Training

To train the Vision Transformer model on the provided dataset, follow these steps:

1. Clone this repository.
    ```
    git clone https://github.com/tikendraw/pneumonia-detection-with-vision-transformer.git
    ```
2. Prepare the dataset and configure the  `config.py` then execute the config.py.

    > folder structure must be
    ```     
    ├── train
    │   ├── normal
    │   │   ├── ...jpeg
    │   │   └── ...jpeg
    │   └── pneumonia
    │       ├── ...jpeg
    │       └── ...jpeg
    │       
    └── test
        ├── normal
        │   ├── ...jpeg
        │   └── ...jpeg
        └── pneumonia
            ├── ...jpeg
            └── ...jpeg
    ```
    ```
    python3 config.py
    ```
3. Run the training script using the `train.py` file.
    ```
    python3 train.py 
    ```

### Prediction

After training the model, you can make predictions on new chest X-ray images using the following steps:

1. Place the test images in a directory.
    ```
    python3 predict.py directory/to/a/folder/that/contains/images/to/test
    ```

2. Run the prediction script using the `predict.py` file.
    ```
    python3 predict.py directory/to/a/folder/that/contains/images/to/test
    ```

### Running the Streamlit App

To deploy a Streamlit web app for interactive pneumonia detection, follow these steps:

1. Configure the Streamlit app in the `app.py` file.
2. Run the app using the `streamlit run app.py` command.

## Contributing

Contributions to this project are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## Acknowledgements

- Special thanks to the creators and maintainers of the Vision Transformer (ViT) architecture.
- The dataset used in this project is from [provide dataset source].

