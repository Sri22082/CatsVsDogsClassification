# 🐱🐶 Cats vs Dogs Image Classification

### Project Overview

This project focuses on building a robust image classification model to distinguish between images of cats and dogs using deep learning techniques. We leverage a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras, trained on a well-known dataset from Kaggle.

![project_cnn](https://github.com/Sri22082/CatsVsDogsClassification/assets/92198693/16063bce-c05a-4482-8269-63a6407c8253)


### Dataset

The dataset used for training and evaluating the model is the [Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats) from Kaggle. It consists of 25,000 labeled images of cats and dogs.

### Model Architecture

The Convolutional Neural Network (CNN) model architecture includes:
- **Convolutional Layers:** Extract features from images
- **MaxPooling Layers:** Reduce dimensionality and computation
- **BatchNormalization Layers:** Normalize activations
- **Dropout Layers:** Prevent overfitting
- **Fully Connected (Dense) Layers:** Classification

### Results

The model achieved the following performance:
- **Training Accuracy:** 97%
- **Validation Accuracy:** 80%

The detailed training process and evaluation metrics can be found in the Jupyter Notebook.

### Getting Started

To replicate the results or further experiment with the model, follow these steps:

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/cats-vs-dogs-classification.git
    cd cats-vs-dogs-classification
    ```

2. **Download the Dataset**
    Follow the instructions in the Jupyter Notebook to download and prepare the dataset using the Kaggle API.

3. **Run the Jupyter Notebook**
    Open and run the `cats_v_dogs_classification.ipynb` notebook to train and evaluate the model:
    ```bash
    jupyter notebook cats_v_dogs_classification.ipynb
    ```

### Contributing

We welcome contributions! If you have ideas for improvements, please fork the repository and submit a pull request.



### Acknowledgements

- **Kaggle:** For providing the dataset.
- **TensorFlow and Keras:** For the deep learning libraries.
- **Community:** For continuous support and contributions.
