
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split



# Define paths to datasets
train_data_path = "inaturalist_12K/train"
test_data_path = "inaturalist_12K/val"

# Define transformations for training and testing datasets
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.CenterCrop((224,224)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load datasets and split into train, validation, and test sets
train_dataset = datasets.ImageFolder(train_data_path, transform=transform_train)
test_dataset = datasets.ImageFolder(test_data_path, transform=transform_test)

# Split train dataset into train and validation sets
validation_ratio = 0.2
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(validation_ratio * num_train))
np.random.shuffle(indices)
train_idx, eval_idx = indices[split:], indices[:split]
train_set = torch.utils.data.Subset(train_dataset, train_idx)
eval_set = torch.utils.data.Subset(train_dataset, eval_idx)

# DataLoader parameters
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained ResNet-50 model and modify the final fully connected layer
resnet = torchvision.models.resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

# Replace the final fully connected layer for 10 output classes
in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, 10)

# Print trainable parameters
for name, param in resnet.named_parameters():
    if param.requires_grad:
        print(f"Trainable parameter: {name}, shape: {param.shape}")

# Move model to GPU if available
resnet = resnet.to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adamax(resnet.parameters(), lr=1e-4)

# Function to evaluate model performance on a dataset
def evaluate_model(dataloader, model, loss_fn):
    total, correct = 0, 0
    loss_epoch_arr = []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss_epoch_arr.append(loss.item())
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    accuracy = 100 * correct / total
    average_loss = sum(loss_epoch_arr) / len(loss_epoch_arr)
    return accuracy, average_loss

# Training loop
max_epochs = 5
train_loss_arr = []
val_loss_arr = []
train_acc_arr = []
val_acc_arr = []

for epoch in range(max_epochs):
    epoch_train_loss = []
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = resnet(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_train_loss.append(loss.item())
    
    train_loss = np.mean(epoch_train_loss)
    train_acc, _ = evaluate_model(train_loader, resnet, loss_fn)
    val_acc, val_loss = evaluate_model(eval_loader, resnet, loss_fn)
    
    train_loss_arr.append(train_loss)
    val_loss_arr.append(val_loss)
    train_acc_arr.append(train_acc)
    val_acc_arr.append(val_acc)

    print(f"Epoch [{epoch+1}/{max_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Plotting the training and validation metrics
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(train_loss_arr)
axs[0, 0].set_title("Training Loss")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Loss")

axs[0, 1].plot(val_loss_arr, 'tab:orange')
axs[0, 1].set_title("Validation Loss")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Loss")

axs[1, 0].plot(train_acc_arr, 'tab:green')
axs[1, 0].set_title("Training Accuracy")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Accuracy (%)")

axs[1, 1].plot(val_acc_arr, 'tab:red')
axs[1, 1].set_title("Validation Accuracy")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()
