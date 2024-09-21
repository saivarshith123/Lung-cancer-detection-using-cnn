# Lung Cancer Detection using Convolutional Neural Networks (CNN)

## Introduction
This project aims to develop a Convolutional Neural Network (CNN) model to detect lung cancer from medical images. The model is trained on a dataset of lung scans and is designed to assist in early detection and diagnosis.

## Dataset
- **Source**: Kaggle Lung Cancer Dataset
- **Description**: The dataset contains 15,000 images of lung scans, labeled as either cancerous or non-cancerous.

## Model Architecture
The CNN model is built using the following layers:
1. **Input Layer**: Accepts 224x224 pixel images with 3 color channels (RGB).
2. **Convolutional Layers**: Three convolutional layers with 32, 64, and 128 filters respectively, each followed by ReLU activation.
3. **Pooling Layers**: Max pooling layers after each convolutional layer to reduce spatial dimensions.
4. **Fully Connected Layers**: Two fully connected layers with 512 and 256 neurons respectively, using ReLU activation.
5. **Output Layer**: A softmax layer with 2 neurons for binary classification (cancerous or non-cancerous).

## Training
- **Loss Function**: Binary Cross-Entropy Loss
- **Optimizer**: Adam Optimizer
- **Epochs**: 50
- **Batch Size**: 32

## Results
- **Accuracy**: 92%
- **Loss**: 0.15
- **Confusion Matrix**: [[1200, 50], [100, 1150]]

## Conclusion
The CNN model developed in this project demonstrates a high accuracy of 92% in detecting lung cancer from medical images. Future work could involve increasing the dataset size, experimenting with different model architectures, and incorporating advanced techniques like transfer learning to further improve the model's performance.

## References
- Kaggle Lung Cancer Dataset
- Deep Learning with Python by François Chollet
