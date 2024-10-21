# 🧠 Deep Neural Network for Image Classification: Application

Welcome to the **Deep Neural Network for Image Classification** repository! This project implements a multi-layer neural network to classify images (e.g., cats vs. non-cats) using Python. 🐱📸

Special thanks to [DeepLearning.AI](https://www.deeplearning.ai/courses/deep-learning-specialization/) for their fantastic **Deep Learning Specialization** course, which inspired this project! 💡

## 🚀 Getting Started

Before you dive in, make sure to install all the necessary dependencies. Here's a quick overview of the project and its structure.

### 🔧 Packages
This project uses the following Python libraries:
- **NumPy** for efficient numerical operations
- **h5py** for handling datasets
- **Matplotlib** for plotting and data visualization
- **PIL** and **SciPy** for image processing
- **dnn_app_utils_v3** for utility functions

```python
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *
```
### 📊 Dataset

We use a small dataset consisting of 64x64 RGB images to classify whether an image is a cat or not. 🐾

- **Training set**: 209 images
- **Test set**: 50 images

```python
# Load the dataset
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Reshape the training and test datasets
train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # (num_px * num_px * 3, num_train)
test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T      # (num_px * num_px * 3, num_test)

# Normalize the image data
train_x = train_x / 255.0
test_x = test_x / 255.0
```

### 🏗️ Model Architecture

The deep neural network consists of several layers, including:
- **Input layer**: Takes in the flattened image data
- **Hidden layers**: Multiple dense layers with activation functions to capture complex patterns
- **Output layer**: Uses a sigmoid activation function for binary classification

Here’s a basic overview of the model architecture:

```python
def model(X_train, Y_train, X_test, Y_test, n_h, num_epochs=2000, learning_rate=0.005, print_cost=True):
    # Initialize parameters
    n_x = X_train.shape[0]  # size of input layer
    n_y = Y_train.shape[0]  # size of output layer
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)
    for i in range(num_epochs):
        # Forward propagation
        A3, cache = forward_propagation(X_train, parameters)
        
        # Cost function
        cost = compute_cost(A3, Y_train)

        # Backward propagation
        grads = backward_propagation(X_train, Y_train, cache)
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 epochs
        if print_cost and i % 100 == 0:
            print(f"Cost after epoch {i}: {cost:.6f}")

    return parameters
```

### 📈 Training the Model

To train the model, we call the `model` function with our training data:

```python
# Set hyperparameters
n_h = 7  # Number of hidden units
parameters = model(train_x, train_y, test_x, test_y, n_h, num_epochs=2500, learning_rate=0.01, print_cost=True)
```

### ✅ Evaluating the Model

After training, we can evaluate the model's performance on the test dataset:

```python
def predict(X, parameters):
    A3, _ = forward_propagation(X, parameters)
    predictions = (A3 > 0.5)  # Convert probabilities to binary output
    return predictions

# Make predictions
predictions = predict(test_x, parameters)

# Calculate accuracy
accuracy = np.mean(predictions == test_y) * 100
print(f"Accuracy: {accuracy:.2f}%")
```

### 📚 Conclusion

This project demonstrates a fundamental implementation of a deep neural network for image classification. Feel free to experiment with different hyperparameters, layers, and datasets to explore the capabilities of deep learning further! 

### 💬 Contributions

If you have suggestions, improvements, or any contributions to this project, feel free to fork the repository and submit a pull request!

Happy coding! 🚀
