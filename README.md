# EN3160 Assignment 3: Neural Networks

This repository contains the Jupyter Notebook and source code implementations for EN3160 Assignment 3 on Neural Networks as detailed in the reference document "task_sheet". The assignment focuses on the mathematical foundations of backpropagation, building Convolutional Neural Networks (CNNs), and applying transfer learning using PyTorch. 

## Key Theories and Technologies

### 1. Multilayer Perceptrons and Manual Backpropagation
* **Network Architecture:** Expanding a single dense layer network into a multilayer architecture by adding a middle layer with 100 nodes and a sigmoid activation function.
* **Loss Functions:** Transitioning from Mean Squared Error to cross-entropy loss to better optimize multi-class classification tasks.

### 2. Convolutional Neural Networks (CNNs)
* **LeNet-5:** Implementation of the classic LeNet-5 CNN architecture for digit recognition.
* **Modern CNN Design:** Building a contemporary 3-layer convolutional network that utilizes modern stabilization and regularization techniques, specifically batch normalization and dropout.
* **Performance Evaluation:** Utilizing confusion matrices to visualize classification accuracy across classes on test data.

### 3. Transfer Learning
* **Pre-trained Models:** Leveraging ResNet18, previously trained on the ImageNet1K dataset, to solve new classification problems.
* **Fine-Tuning vs. Feature Extraction:** Comparing two primary transfer learning paradigms: fine-tuning the entire network's weights versus freezing the base layers and using the network strictly as a feature extractor for a new classification head.

## Project Roadmap

The repository is structured around the following core tasks:

* **Task 1: Manual Network Modification:** Modify the baseline manual PyTorch code to include a hidden layer, implement cross-entropy loss, and train the model on the CIFAR-10 dataset for 10 epochs to extract training and test accuracies.
* **Task 2: PyTorch CNNs for MNIST:** Create and train both a LeNet-5 network and a modern 3-layer CNN on the MNIST dataset for 10 epochs. Log and plot the per-epoch training and test accuracy and loss, and display a final confusion matrix for each model.
* **Task 3: Transfer Learning with ResNet18:** Classify the hymenoptera dataset utilizing the pre-trained ResNet18 model. Perform and report the results of both full-network fine-tuning and feature extraction methodologies.

## Technologies Used
* **Python**
* **PyTorch (`torch`, `torch.nn`, `torch.optim`)**
* **Torchvision (`torchvision.datasets`, `torchvision.transforms`)**
* **Matplotlib (`matplotlib.pyplot`)**

## Author
**Bashitha Anthony**  
B.Sc. Hons in Electronics and Telecommunication Engineering  
University of Moratuwa, Sri Lanka