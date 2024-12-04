# CNN-Cancer_Detection
Cancer Detection using CNN

# Histopathologic Cancer Detection with CNN

This repository contains a Python implementation of a Convolutional Neural Network (CNN) to detect cancerous tissue in histopathologic images. The dataset used is from the [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/overview) competition on Kaggle.

## Overview

The goal of this project was to build a deep learning model capable of distinguishing between cancerous and non-cancerous tissue from images. Using a CNN architecture, we achieved an **accuracy of 89.06%** on the training set and **83.93%** on the validation set.

## Dataset

The dataset consists of histopathologic images, each labeled as either cancerous (1) or non-cancerous (0). Images are 96x96 pixels, but we resized them to **60x60** for computational efficiency. The dataset was preprocessed to normalize pixel values and split into training and validation sets.

## Model Architecture

The CNN model was built using Keras' Sequential API with the following layers:

1. **Input Layer**: Input shape defined as `(60, 60, 3)`.
2. **Convolutional Layers**: 
   - `Conv2D` layers with 32, 64, and 128 filters, kernel size `(3, 3)`, and ReLU activation.
   - `MaxPooling2D` layers to down-sample the feature maps.
3. **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
4. **Fully Connected Layers**:
   - Dense layer with 256 neurons and ReLU activation.
   - Dropout layer with a rate of 0.5 to prevent overfitting.
5. **Output Layer**: Single neuron with a sigmoid activation function for binary classification.

### Model Summary:
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Performance**:
  - Training: **Accuracy** = 89.06%, **Loss** = 0.3066
  - Validation: **Accuracy** = 83.93%, **Loss** = 0.3725

## Key Learnings and Takeaways

1. **Data Preprocessing Matters**: Resizing and normalizing images significantly influenced training stability and model performance.
2. **Model Regularization**: Adding a dropout layer reduced overfitting, leading to better generalization on the validation set.
3. **Balancing Performance**: While the model performs well, there is room for improvement, particularly in validation accuracy.

## Potential Improvements

1. **Data Augmentation**: Applying techniques like rotation, flipping, and zooming could improve model generalization by increasing the diversity of the training dataset.
2. **Hyperparameter Tuning**: Experimenting with learning rates, batch sizes, and the number of filters in the convolutional layers.
3. **Advanced Architectures**: Using pre-trained models like VGG16, ResNet, or EfficientNet with transfer learning for better performance.
4. **Custom Loss Function**: Trying a weighted loss function to address any class imbalance.
5. **Larger Input Images**: Using the original image size (96x96) instead of resizing to 60x60 might preserve more information and improve accuracy.
6. **Cross-validation**: Performing k-fold cross-validation to ensure robustness of the model.

## Conclusion

This project demonstrates the feasibility of using CNNs for cancer detection in histopathologic images, achieving promising accuracy and insights for further improvement. The techniques and findings here can be a stepping stone for deploying more sophisticated models for medical image analysis.

---

Feel free to clone the repository, experiment with the code, and suggest improvements. Contributions are welcome!

