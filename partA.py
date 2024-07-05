# Build a small CNN model consisting of 5 convolution layers. Each
# convolution layer would be followed by an activation and a max-
# pooling layer.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim


import wandb
wandb.login(key='f2fe2cd24a91733ff0c552e3956ff0372e874226')

class CNN(nn.Module):
    def __init__(self, num_filters, filter_sizes, activation, batch_norm, dropout, dense_neurons, dense_activation, mystride=2):
        super(CNN, self).__init__()
    
        activations_dict={'relu':nn.ReLU(),'silu':nn.SiLU(),'gelu':nn.GELU(),'mish':nn.Mish()}
        in_channel = 3

        i = 0
        self.conv1 = nn.Conv2d(in_channel, num_filters[i], filter_sizes[i])
        self.act1 = activations_dict[activation]
        self.batch1 = nn.BatchNorm2d(num_filters[i]) if batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel = 2, stride = mystride)

        i+=1
        self.conv2 = nn.Conv2d(num_filters[i-1], num_filters[i], filter_sizes[i])
        self.act2 = activations_dict[activation]
        self.batch2 = nn.BatchNorm2d(num_filters[i]) if batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel = 2, stride = mystride)

        i+=1
        self.conv3 = nn.Conv2d(num_filters[i-1], num_filters[i], filter_sizes[i])
        self.act3 = activations_dict[activation]
        self.batch3 = nn.BatchNorm2d(num_filters[i]) if batch_norm else nn.Identity()
        self.pool3 = nn.MaxPool2d(kernel = 2, stride = mystride)

        i+=1
        self.conv4 = nn.Conv2d(num_filters[i-1], num_filters[i], filter_sizes[i])
        self.act4 = activations_dict[activation]
        self.batch4 = nn.BatchNorm2d(num_filters[i]) if batch_norm else nn.Identity()
        self.pool4 = nn.MaxPool2d(kernel = 2, stride = mystride)

        i+=1
        self.conv5 = nn.Conv2d(num_filters[i-1], num_filters[i], filter_sizes[i])
        self.act5 = activations_dict[activation]
        self.batch5 = nn.BatchNorm2d(num_filters[i]) if batch_norm else nn.Identity()
        self.pool5 = nn.MaxPool2d(kernel = 2, stride = mystride)
        
        input_shape = (3, 224, 224)
        dummy_input = torch.randn(1, *input_shape)
        output_size = self.get_output_size(dummy_input)

        self.flat = nn.Flatten()

        self.dense_layer = nn.Linear(output_size, dense_neurons)
        self.dense_acti = activations_dict[dense_activation]
        self.drop = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dense_neurons, 10)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.batch1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.batch2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.batch3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.act4(x)
        x = self.batch4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.batch5(x)
        x = self.pool5(x)

        x = self.flat(x)
        x = self.dense_layer(x)
        x = self.dense_acti(x)
        x = self.drop(x)
        x = self.output_layer(x)
        return x
    
    def get_output_size(self, random_image):
        x = self.pool1(self.act1(self.conv1(random_image)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.pool4(self.act4(self.conv4(x)))
        x = self.pool5(self.act5(self.conv5(x)))
        return x.shape[1]*x.shape[2]*x.shape[3]
    


# input_shape = (3, 224, 224)  # Assuming images from iNaturalist dataset in the format (channels, height, width)
num_filters = [64, 64, 64, 64, 64]  # Number of filters in each convolutional layer
filter_sizes = [(3, 3)] * 5 # Filter sizes for each convolutional layer
activation = 'mish'   # Activation functions for each convolutional layer
dense_neurons = 256  # Number of neurons in the dense layer
batch_norm = True 
dropout = 0.1  # Dropout rate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def load_data():
    traindata = "inaturalist_12K/train"
    testdata = "inaturalist_12K/val"
    return traindata, testdata

def evaluate_model(dataloader, model, loss_function):
    total_samples, correct_predictions = 0, 0
    loss_epoch_list = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss_epoch_list.append(loss.item())
            _, predictions = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predictions == labels).sum().item()
    
    accuracy = 100 * correct_predictions / total_samples
    average_loss = sum(loss_epoch_list) / len(loss_epoch_list)
    return accuracy, average_loss

def train_cnn(num_filters=[64, 64, 64, 64, 64], filter_sizes=[3, 3, 3, 3, 3], activation='mish', augment_data=True, batch_norm=True, dropout_rate=0.1, dense_size=256, dense_activation='mish', stride=2, max_epochs=10):
    
    train_path, val_path = load_data()
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    if augment_data:
        train_dataset = datasets.ImageFolder(train_path, transform=transform_train)
        val_dataset = datasets.ImageFolder(val_path, transform=transform)
    else:
        train_dataset = datasets.ImageFolder(train_path, transform=transform)
        val_dataset = datasets.ImageFolder(val_path, transform=transform)
        
    num_val_samples = int(np.floor(0.2 * len(train_dataset)))
    num_train_samples = len(train_dataset) - num_val_samples
    train_dataset, eval_dataset = random_split(train_dataset, [num_train_samples, num_val_samples])
    val_dataset, _ = random_split(val_dataset, [len(val_dataset), 0])
    
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    cnn_model = CNN(num_filters, filter_sizes, activation, batch_norm, dropout_rate, dense_size, dense_activation, stride).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(cnn_model.parameters(), lr=1e-4)
    
    for epoch in range(max_epochs):
        epoch_loss_list = []
        for i, data in enumerate(train_loader, 0):
            if i % 50 == 0:
                print(f"Epoch {epoch}, Batch {i}")
                
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = cnn_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss_list.append(loss.item())
            
        train_loss = sum(epoch_loss_list) / len(epoch_loss_list)
        train_accuracy, _ = evaluate_model(train_loader, cnn_model, loss_fn)
        val_accuracy, val_loss = evaluate_model(eval_loader, cnn_model, loss_fn)
        print(f'Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')
        wandb.log({'Train Loss': train_loss, 'Train Accuracy': train_accuracy, 'Val Loss': val_loss, 'Val Accuracy': val_accuracy})

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "Val Accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "max_epochs": {
            "values": [5, 7, 9]
        },
        "num_filters": {
            "values": [[64, 64, 64, 64, 64], [32, 64, 128, 256, 512], [32, 32, 32, 32, 32], [512, 256, 128, 64, 32]]
        },
        "filter_sizes": {
            "values": [[3, 3, 3, 3, 3], [5, 5, 5, 5, 5], [11, 9, 7, 5, 3]]
        },
        "activation": {
            "values": ['relu', 'gelu', 'mish', 'silu']
        },
        "augment_data": {
            "values": [True, False]
        },
        "batch_norm": {
            "values": [True, False]
        },
        "dense_activation": {
            "values": ['relu', 'gelu', 'mish', 'silu']
        },
        "dropout_rate": {
            "values": [0.1, 0.2, 0.3]
        },
        "dense_size": {
            "values": [128, 256, 512]
        },
        "stride": {
            "values": [2, 3, 5]
        }
    }
}

def train_with_sweep():
    wandb.init()
    config = wandb.config
    train_cnn(
        num_filters=config.num_filters,
        filter_sizes=config.filter_sizes,
        activation=config.activation,
        augment_data=config.augment_data,
        batch_norm=config.batch_norm,
        dropout_rate=config.dropout_rate,
        dense_size=config.dense_size,
        dense_activation=config.dense_activation,
        stride=config.stride,
        max_epochs=config.max_epochs
    )
    wandb.finish()

sweep_id = wandb.sweep(sweep=sweep_config, project="CNN_Training")
wandb.agent(sweep_id, function=train_with_sweep, count=40)

wandb.finish()

## Now We will use our test data to test our trained model PardA Question 4


cnn_model = CNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adamax(cnn_model.parameters(),lr=1e-4)


trainset, testset =load_data()
batch_size = 32 #you better know the importamce of batchsize especially with respect to GPU memory
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
testset = datasets.ImageFolder(testset,transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def test_cnn(num_filters=[64, 64, 64, 64, 64], filter_sizes=[3, 3, 3, 3, 3], activation='mish', augment_data=True, batch_norm=True, dropout_rate=0.1, dense_size=256, dense_activation='mish', stride=2, max_epochs=10):
    
    trainset,testset=load_data()
    

    
    #my transformation on the trainset
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.CenterCrop((224,224)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    if augment_data:
        train_dataset = datasets.ImageFolder(trainset,transform_train)
        test_dataset = datasets.ImageFolder(testset,transform)
    else:
        train_dataset = datasets.ImageFolder(trainset,transform)
        test_dataset = datasets.ImageFolder(testset,transform)
        
    num_val_samples = int(np.floor(0.2 * len(train_dataset)))
    num_train_samples = len(train_dataset) - num_val_samples
    train_dataset, eval_dataset = random_split(train_dataset,[num_train_samples,num_val_samples])
    test_dataset, test_dataset2 = random_split(test_dataset,[len(test_dataset),0])
     
    
    batch_size = 32  #you better know the importamce of batchsize especially with respect to GPU memory
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    
    
    for epoch in range(max_epochs):
        loss_epoch_list = []
        for i, data in enumerate(trainloader, 0):                
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = cnn_model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_epoch_list.append(loss.item())
            
        loss_train=sum(loss_epoch_list)/len(loss_epoch_list)
        train_acc,_=evaluate_model(trainloader,cnn_model,loss_function)
        eval_acc,loss_eval=evaluate_model(evalloader,cnn_model,loss_function)
        print(f' epoch:- {epoch } train loss:- {loss_train} train acc:- {train_acc} val loss:- {loss_eval} val acc:- {eval_acc} ')
    print("Testing my model...")
    test_acc,test_loss=evaluate_model(testloader,cnn_model,loss_function)
    print("CNN test loss",test_loss,"CNN test acc",test_acc)
    
            
test_cnn([512,256,128,64,32],[3,3,3,3,3],'gelu',False,True,0.1,256,'mish',2,5)




import torch
import torchvision
import matplotlib.pyplot as plt

def predictions(dataloader=testloader, model=cnn_model):
    # Set model to evaluation mode
    wandb.init()
    model.eval()
    plot = []

    # Create figure for plotting images
    fig, axs = plt.subplots(10, 3, figsize=(10, 30))

    # Iterate over batches in dataloader
    for i, batch in enumerate(dataloader):
        # Get batch of images and labels
        if i>=1:
          break
        images, labels = batch
        images,labels=images.to(device),labels.to(device)

        # Make predictions with model
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

        # Plot images with actual and predicted labels as titles
        for j in range(images.size()[0]):
            image = images[j]
            label = labels[j]
            pred = preds[j]
            mylabel=classes[label.item()]
            mypred=classes[pred.item()]
            std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            image = np.multiply(image.cpu(), std_correction) + mean_correction

            # Calculate the row and column indices of the current subplot
            row_idx = (i * 3 + j) // 3
            col_idx = (i * 3 + j) % 3
            if row_idx*3+col_idx>29:
                break

            axs[row_idx, col_idx].imshow(torchvision.utils.make_grid(image, nrow=1).permute(1, 2, 0))
            plot.append(wandb.Image(image,caption= 'True='+ mylabel +', Predicted='+mypred))
            axs[row_idx, col_idx].set_title('Actual: {} \nPredicted: {}'.format(mylabel, mypred))
            axs[row_idx, col_idx].axis('off')

            # Check if we have displayed all 30 images
            if (i+1)*30 + j == len(dataloader.dataset):
                break

        # Check if we have displayed all 30 images
        if (i+1)*30 == len(dataloader.dataset):
            break

    plt.tight_layout()
    plt.show()
    wandb.log({"images":plot})
classes=('Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia')
# sweep_id=wandb.sweep(sweep=sweep_config,project="DeepLearn2")
# wandb.agent(sweep_id,predictions,count=1)


