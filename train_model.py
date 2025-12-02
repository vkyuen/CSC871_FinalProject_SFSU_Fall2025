# %% [markdown]
# # Creating and training models
# This file/notebook will create and train different models to see how different models will act.  Once the model has been trained, it will save the models to a folder for future use.

# %%
from data_module import get_data_loaders
import os

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# %% [markdown]
# ## Check for GPU

# %%
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU")
else:
    print("Using CPU")

# %% [markdown]
# ## Loading data loaders
# Putting the loaders in a dictionary.  Since this file focuses on the models, I really only need the training and validation sets, but since all three loaders are given, it wouldn't hurt to add it to the dictionary.

# %%
train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)
dataloaders = {"TRAIN": train_loader, "VAL": val_loader, "TEST": test_loader}

# %% [markdown]
# ## Define a training function
# Followed a tutorial that trains and validates the model at each epoch. With each training, the model, optimizer, criterion, number of epochs and the early stopping critera can be set. 
# 
# At the end of each epoch, the statistics is printed to the console, before checking to see if the stopping critera has been met, and if the current model the is being tested is the best model.  If it is, then it is saved to be returned.

# %%
def train_model(model, optimizer, criterion, num_epochs=50, early_stopping=None):
    model.train()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        
        for phase in ["TRAIN", "VAL"]:
            if phase == "TRAIN":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            # running_accuracy = 0.0
            running_correct = 0
            running_total = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "TRAIN"):
                    outputs = model(inputs)  # Shape: (batch_size, 1), dtype: float
                    loss = criterion(outputs, labels)  # ✅ Convert labels to float
                    prediction = torch.sigmoid(outputs).round().long()  # For accuracy calculation
                    _, predicted = torch.max(outputs.data, 1)
                
                # Backward pass and optimization
                if phase == "TRAIN":
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                # running_accuracy += torch.sum(prediction == labels)
                running_total += labels.size(0)
                running_correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / running_total
            # epoch_acc = running_accuracy.double() / len(dataloaders[phase].dataset)
            epoch_acc = (running_correct/running_total) * 100

            print(f'{phase} Loss: {epoch_loss:.2f} Acc: {epoch_acc:.2f}')

            # Early stopping
            if phase == "VAL":
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    model.load_state_dict(best_model_wts)
                    return model

            # Save the best model
            if phase == "VAL" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val Acc: {best_acc:.2f}')
    model.load_state_dict(best_model_wts)
    return model

# %% [markdown]
# ## Early stopping
# This method checks to make sure the training stops in time to prevent overfitting to the training data.

# %%
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

# %% [markdown]
# ## Generate the different models
# Create instances of the different model architectures

# %%
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# ### CNN

# %%
def cnn_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 56 * 56, 128),
        nn.ReLU(),
        nn.Linear(128, 1)  # Binary classification
    )
    return model 

# %%
modelCNN = cnn_model().to(device)
optimizerCNN = optim.SGD(cnn_model().parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopCNN = EarlyStopping(patience=7, delta=0.01)
modelCNN = modelCNN.to(device)

trainedCNN = train_model(modelCNN, optimizerCNN, criterion, early_stopping=early_stopCNN)
pathCNN = os.path.join("saved_models", "cnn.pth")

checkpointCNN = {'model_state_dict': trainedCNN.state_dict(),
                 'optimizer_state_dict': optimizerCNN.state_dict()}
torch.save(checkpointCNN, pathCNN)

# %% [markdown]
# ### ResNet18

# %%
modelResNet18Wts = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = modelResNet18Wts.fc.in_features

for param in modelResNet18Wts.parameters():
    param.requires_grad = False

modelResNet18Wts.fc = nn.Linear(num_ftrs, 1)  # Binary classification
optimizerResNet18Wts = optim.SGD(modelResNet18Wts.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopResNet18Wts = EarlyStopping(patience=7, delta=0.01)
modelResNet18Wts = modelResNet18Wts.to(device)

trainedResNet18Wts = train_model(modelResNet18Wts, optimizerResNet18Wts, criterion, early_stopping=early_stopResNet18Wts)

pathResNet18Wts = os.path.join("saved_models", "resnet18_weights.pth")

checkpointResNet18Wts = {'model_state_dict': trainedResNet18Wts.state_dict(),
                         'optimizer_state_dict': optimizerResNet18Wts.state_dict()}

torch.save(checkpointResNet18Wts, pathResNet18Wts)

# %%
modelResNet18 = models.resnet18()
num_ftrs = modelResNet18.fc.in_features

for param in modelResNet18.parameters():
    param.requires_grad = False

modelResNet18.fc = nn.Linear(num_ftrs, 1)  # Binary classification
optimizerResNet18 = optim.SGD(modelResNet18.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopResNet18 = EarlyStopping(patience=7, delta=0.01)
modelResNet18 = modelResNet18.to(device)

trainedResNet18 = train_model(modelResNet18, optimizerResNet18, criterion, early_stopping=early_stopResNet18)

pathResNet18 = os.path.join("saved_models", "resnet18.pth")

checkpointResNet18 = {'model_state_dict': trainedResNet18.state_dict(),
                      'optimizer_state_dict': optimizerResNet18.state_dict()}

torch.save(checkpointResNet18, pathResNet18)

# %% [markdown]
# ### ResNet50

# %%
modelResNet50Wts = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = modelResNet50Wts.fc.in_features

for param in modelResNet50Wts.parameters():
    param.requires_grad = False

modelResNet50Wts.fc = nn.Linear(num_ftrs, 1)  # Binary classification
optimizerResNet50Wts = optim.SGD(modelResNet50Wts.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopResNet50Wts = EarlyStopping(patience=7, delta=0.01)
modelResNet50Wts = modelResNet50Wts.to(device)

trainedResNet50Wts = train_model(modelResNet50Wts, optimizerResNet50Wts, criterion, early_stopping=early_stopResNet50Wts)

pathResNet50Wts = os.path.join("saved_models", "resnet50_weights.pth")
torch.save(trainedResNet50Wts.state_dict(), pathResNet50Wts)

checkpointResNet50Wts = {'model_state_dict': trainedResNet50Wts.state_dict(),
                         'optimizer_state_dict': optimizerResNet50Wts.state_dict()}

torch.save(checkpointResNet50Wts, pathResNet50Wts)

# %%
modelResNet50 = models.resnet50()
num_ftrs = modelResNet50.fc.in_features

for param in modelResNet50.parameters():
    param.requires_grad = False

modelResNet50.fc = nn.Linear(num_ftrs, 1)  # Binary classification
optimizerResNet50 = optim.SGD(modelResNet50.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopResNet50 = EarlyStopping(patience=7, delta=0.01)
modelResNet50 = modelResNet50.to(device)

trainedResNet50 = train_model(modelResNet50, optimizerResNet50, criterion, early_stopping=early_stopResNet50)

pathResNet50 = os.path.join("saved_models", "resnet50.pth")

checkpointResNet50 = {'model_state_dict': trainedResNet50.state_dict(),
                      'optimizer_state_dict': optimizerResNet50.state_dict()}

torch.save(checkpointResNet50, pathResNet50)

# %% [markdown]
# ### VGG16

# %%
modelVGG16Wts = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

for param in modelVGG16Wts.parameters():
    param.requires_grad = False

num_features = modelVGG16Wts.classifier[6].in_features
features = list(modelVGG16Wts.classifier.children())[:-1]
features.extend([nn.Linear(num_features, 1)])  # Binary classification
modelVGG16Wts.classifier = nn.Sequential(*features)

optimizerVGG16Wts = optim.SGD(modelVGG16Wts.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopVGG16Wts = EarlyStopping(patience=7, delta=0.01)
modelVGG16Wts = modelVGG16Wts.to(device)

trainedVGG16Wts = train_model(modelVGG16Wts, optimizerVGG16Wts, criterion, early_stopping=early_stopVGG16Wts)

pathVGG16Wts = os.path.join("saved_models", "vgg16_weights.pth")

checkpointVGG16Wts = {'model_state_dict': trainedVGG16Wts.state_dict(),
                      'optimizer_state_dict': optimizerVGG16Wts.state_dict()}

torch.save(checkpointVGG16Wts, pathVGG16Wts)

# %%
modelVGG16 = models.vgg16()

for param in modelVGG16.parameters():
    param.requires_grad = False

num_features = modelVGG16.classifier[6].in_features
features = list(modelVGG16.classifier.children())[:-1]
features.extend([nn.Linear(num_features, 1)])  # Binary classification
modelVGG16.classifier = nn.Sequential(*features)

optimizerVGG16 = optim.SGD(modelVGG16.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopVGG16 = EarlyStopping(patience=7, delta=0.01)
modelVGG16 = modelVGG16.to(device)

trainedVGG16 = train_model(modelVGG16, optimizerVGG16, criterion, early_stopping=early_stopVGG16)

pathVGG16 = os.path.join("saved_models", "vgg16.pth")

checkpointVGG16 = {'model_state_dict': trainedVGG16.state_dict(),
                   'optimizer_state_dict': optimizerVGG16.state_dict()}
torch.save(checkpointVGG16, pathVGG16)

# %% [markdown]
# ### DenseNet

# %%
modelDensenetWts = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
num_ftrs = modelDensenetWts.classifier.in_features

for param in modelDensenetWts.parameters():
    param.requires_grad = False

modelDensenetWts.classifier = nn.Linear(num_ftrs, 2)  # Binary classification
optimizerDensenetWts = optim.SGD(modelDensenetWts.classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopDensenetWts = EarlyStopping(patience=7, delta=0.01)
modelDensenetWts = modelDensenetWts.to(device)

trainedDensenetWts = train_model(modelDensenetWts, optimizerDensenetWts, criterion, early_stopping=early_stopDensenetWts)

pathDensenetWts = os.path.join("saved_models", "densenet121_weights.pth")

chckpointDensenetWts = {'model_state_dict': trainedDensenetWts.state_dict(),
                        'optimizer_state_dict': optimizerDensenetWts.state_dict()}
torch.save(chckpointDensenetWts, pathDensenetWts)

# %%
modelDensenet = models.densenet121()
num_ftrs = modelDensenet.classifier.in_features

for param in modelDensenet.parameters():
    param.requires_grad = False

modelDensenet.classifier = nn.Linear(num_ftrs, 2)  # Binary classification
optimizerDensenet = optim.SGD(modelDensenet.classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopDensenet = EarlyStopping(patience=7, delta=0.01)
modelDensenet = modelDensenet.to(device)
trainedDensenet = train_model(modelDensenet, optimizerDensenet, criterion, early_stopping=early_stopDensenet)

pathDensenet = os.path.join("saved_models", "densenet121.pth")

checkpointDensenet = {'model_state_dict': trainedDensenet.state_dict(),
                      'optimizer_state_dict': optimizerDensenet.state_dict()}
torch.save(checkpointDensenet, pathDensenet)

# %% [markdown]
# ### Efficient

# %%
modelEfficientWts = models.efficientnet_b0(weights= models.EfficientNet_B0_Weights.DEFAULT)
num_ftrs = modelEfficientWts.classifier[1].in_features

for param in modelEfficientWts.parameters():
    param.requires_grad = False

modelEfficientWts.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification
optimizerEfficientWts = optim.SGD(modelEfficientWts.classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopEfficientWts = EarlyStopping(patience=7, delta=0.01)
modelEfficientWts = modelEfficientWts.to(device)

trainedEfficientWts = train_model(modelEfficientWts, optimizerEfficientWts, criterion, early_stopping=early_stopEfficientWts)

pathEfficientWts = os.path.join("saved_models", "efficientnet_b0_weights.pth")

checkpointEfficientWts = {'model_state_dict': trainedEfficientWts.state_dict(),
                          'optimizer_state_dict': optimizerEfficientWts.state_dict()}
torch.save(checkpointEfficientWts, pathEfficientWts)

# %%
modelEfficient = models.efficientnet_b0()
num_ftrs = modelEfficient.classifier[1].in_features

for param in modelEfficient.parameters():
    param.requires_grad = False

modelEfficient.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification
optimizerEfficient = optim.SGD(modelEfficient.classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

early_stopEfficient = EarlyStopping(patience=7, delta=0.01)
modelEfficient = modelEfficient.to(device)

trainedEfficient = train_model(modelEfficient, optimizerEfficient, criterion, early_stopping=early_stopEfficient)

pathEfficient = os.path.join("saved_models", "efficientnet_b0.pth")

checkpointEfficient = {'model_state_dict': trainedEfficient.state_dict(),
                       'optimizer_state_dict': optimizerEfficient.state_dict()}
torch.save(checkpointEfficient, pathEfficient)


