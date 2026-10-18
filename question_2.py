import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Data Loading (MNIST)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

batch_size = 64

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = tuple(str(i) for i in range(10))

# ==========================================
# 2. Part (a): LeNet-5 Architecture
# ==========================================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Input is 1x28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # out: 6x24x24
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # out: 6x12x12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # out: 16x8x8
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # out: 16x4x4
        
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 3. Part (b): Modern Small CNN Architecture
# ==========================================
class ModernCNN(nn.Module):
    def __init__(self):
        super(ModernCNN, self).__init__()
        # 3-layer convolutional network with batch normalization and dropout
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # out: 16x14x14
            
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # out: 32x7x7
            
            # Layer 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # out: 64x3x3 (integer division of 7/2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 3 * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 4. Part (c): Training, Logging, and Plotting
# ==========================================
def train_and_evaluate(model, model_name, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': []
    }
    
    print(f"\n--- Training {model_name} ---")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = 100. * correct / total
        
        # Testing Phase
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_test_loss = running_loss / total
        epoch_test_acc = 100. * correct / total
        
        # Log Metrics
        history['train_loss'].append(epoch_train_loss)
        history['test_loss'].append(epoch_test_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_acc'].append(epoch_test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")
        
    return history, all_labels, all_preds

def plot_results(history, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['test_loss'], label='Test Loss', marker='s')
    plt.title(f'{model_name} - Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs, history['test_acc'], label='Test Acc', marker='s')
    plt.title(f'{model_name} - Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(labels, preds, model_name):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    # Train and evaluate LeNet-5
    lenet = LeNet5()
    lenet_history, lenet_labels, lenet_preds = train_and_evaluate(lenet, "LeNet-5", epochs=10)
    plot_results(lenet_history, "LeNet-5")
    plot_confusion_matrix(lenet_labels, lenet_preds, "LeNet-5")

    # Train and evaluate Modern CNN
    modern_cnn = ModernCNN()
    modern_history, modern_labels, modern_preds = train_and_evaluate(modern_cnn, "Modern Small CNN", epochs=10)
    plot_results(modern_history, "Modern Small CNN")
    plot_confusion_matrix(modern_labels, modern_preds, "Modern Small CNN")
