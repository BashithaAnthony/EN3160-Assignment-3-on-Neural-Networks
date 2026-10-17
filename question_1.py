import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Dataloading
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 50

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Define Network Parameters
Din = 3 * 32 * 32 # Input size (flattened CIFAR-10 image size)
H = 100           # Middle layer size (100 nodes)
K = 10            # Output size (number of classes in CIFAR-10)

# Initialize weights and biases
# Changed std from 1e-5 to 1e-2; this is a necessary change because 1e-5 
# is too small for a deeper network to propagate meaningful gradients.
std = 1e-2 
w1 = torch.randn(Din, H) * std
b1 = torch.zeros(H)
w2 = torch.randn(H, K) * std
b2 = torch.zeros(K)

# Hyperparameters
iterations = 10    # Run the network for 10 epochs
lr = 0.1           # Necessary change: learning rate increased to 0.1 for Cross-Entropy 
lr_decay = 0.9     # Learning rate decay
reg = 0            # Regularization
loss_history = []

# 3. Training Loop
for t in range(iterations):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get inputs and labels
        inputs, labels = data
        Ntr = inputs.shape[0] # Batch size
        x_train = inputs.view(Ntr, -1) # Flatten input to (Ntr, Din)
        y_train_onehot = nn.functional.one_hot(labels, K).float() # Convert labels to one hot

        # Forward pass (Layer 1 -> Sigmoid -> Layer 2)
        z1 = x_train.mm(w1) + b1
        h1 = torch.sigmoid(z1)
        y_pred = h1.mm(w2) + b2 # Raw logits
        
        # Softmax computation (with shift for numerical stability)
        y_pred_shifted = y_pred - torch.max(y_pred, dim=1, keepdim=True).values
        exp_scores = torch.exp(y_pred_shifted)
        probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

        # Cross-Entropy Loss calculation
        loss = -torch.sum(y_train_onehot * torch.log(probs + 1e-8)) / Ntr
        loss += reg * (torch.sum(w1**2) + torch.sum(w2**2))
        
        loss_history.append(loss.item())
        running_loss += loss.item()

        # Backpropagation
        # 1. Gradient of cross entropy with softmax
        dy_pred = (probs - y_train_onehot) / Ntr
        
        # 2. Gradients for output layer (w2, b2)
        dw2 = h1.t().mm(dy_pred) + reg * w2
        db2 = dy_pred.sum(dim=0)
        
        # 3. Backpropagate error to hidden layer
        dh1 = dy_pred.mm(w2.t())
        
        # 4. Derivative of Sigmoid: a * (1 - a)
        dz1 = dh1 * h1 * (1.0 - h1)
        
        # 5. Gradients for hidden layer (w1, b1)
        dw1 = x_train.t().mm(dz1) + reg * w1
        db1 = dz1.sum(dim=0)

        # Parameter update
        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2

    # Print loss for every epoch
    print(f"Epoch {t+1}/{iterations}, Loss: {running_loss / len(trainloader)}")
    
    # Learning rate decay
    lr *= lr_decay

# 4. Plotting the Loss History
plt.plot(loss_history)
plt.title("Loss History")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# 5. Calculate Accuracy on Training Set
correct_train = 0
total_train = 0
with torch.no_grad():
    for data in trainloader:
        inputs, labels = data
        Ntr = inputs.shape[0]
        x_train = inputs.view(Ntr, -1)
        
        # Forward pass
        z1 = x_train.mm(w1) + b1
        h1 = torch.sigmoid(z1)
        y_train_pred = h1.mm(w2) + b2
        
        predicted_train = torch.argmax(y_train_pred, dim=1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

train_acc = 100 * correct_train / total_train
print(f"Training accuracy: {train_acc:.2f}%")

# 6. Calculate Accuracy on Test Set
correct_test = 0
total_test = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        Nte = inputs.shape[0]
        x_test = inputs.view(Nte, -1)
        
        # Forward pass
        z1 = x_test.mm(w1) + b1
        h1 = torch.sigmoid(z1)
        y_test_pred = h1.mm(w2) + b2
        
        predicted_test = torch.argmax(y_test_pred, dim=1)
        total_test += labels.size(0)
        correct_test += (predicted_test == labels).sum().item()

test_acc = 100 * correct_test / total_test
print(f"Test accuracy: {test_acc:.2f}%")
