# %% [markdown]
# # Creating models and saving to an array 
# Assuming the models are pre-set, and there is a folder named saved_models that holds all the models. Since the saved state dictionary were processed using GPU, speed up, the device should 
# 
# The get method will return arras wtith the model name, the models, and the optimizer.  The criterion is not included because the same one is used across all the models.

# %%
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# %% [markdown]
# ### CNN
# This is the only model that we have to create, all the other models are pre-trained in the library.

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
isTroubleshoot = True

# %% [markdown]
# ## Load models
# This class will take in the path to the folder that has all the models. Assuming all the file names are the same.

# %%
class loadModels:
    # modelNames = ['cnn', 'resnet18_weights', 'resnet18', 'resnet50_weights', 'resnet50', 
    #               'vgg16_weights', 'vgg16', 'densenet121_weights', 'densenet121', 
    #               'efficientnet_b0_weights', 'efficientnet_b0']
    modelNames = ['cnn', 'resnet18_weights', 'resnet18', 'resnet50_weights', 'resnet50', 
                  'densenet121_weights', 'densenet121', 'efficientnet_b0_weights', 
                  'efficientnet_b0']
    models = []

    path = ""
    device = ""
    def __init__(self, path, device):
        self.path = path
        self.device = device


    def get_files_in_folder(self, folder_path):
        # Check if the provided path is a valid directory
        if os.path.isdir(folder_path):
            for entry in os.listdir(folder_path):
                full_path = os.path.join(folder_path, entry)
                # Check if the entry is a file
                if os.path.isfile(full_path):
                    entry = self.remove_file_extension(entry)
                    self.modelNames.append(entry)
        else:
            print(f"Error: '{folder_path}' is not a valid directory.")
    
    def remove_file_extension(self, filepath):
        filename_without_extension, _ = os.path.splitext(filepath)
        return filename_without_extension
    
    def load_models(self):
        # Load CNN model
        modelCNN = cnn_model().to(self.device)

        checkpointCNN = torch.load(os.path.join(self.path, "cnn.pth"), map_location=self.device)
        modelCNN.load_state_dict(checkpointCNN['model_state_dict'])
        self.models.append(modelCNN)
        
        if(isTroubleshoot):
            print("Loaded CNN model.")
        
        # Load ResNet18 with weights
        modelResNet18Wts = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        num_ftrs = modelResNet18Wts.fc.in_features

        for param in modelResNet18Wts.parameters():
            param.requires_grad = False
        modelResNet18Wts.fc = nn.Linear(num_ftrs, 1)  # Binary classification
        
        checkpointResNet18Wts = torch.load(os.path.join(self.path, "resnet18_weights.pth"), map_location=self.device)
        modelResNet18Wts.load_state_dict(checkpointResNet18Wts['model_state_dict'])
        self.models.append(modelResNet18Wts)
        
        if(isTroubleshoot):
            print("Loaded ResNet18 with weights model.")

        # Load ResNet18 without weights
        modelResNet18 = models.resnet18().to(self.device)
        num_ftrs = modelResNet18.fc.in_features

        for param in modelResNet18.parameters():
            param.requires_grad = False
        modelResNet18.fc = nn.Linear(num_ftrs, 1)  # Binary classification

        checkpointResNet18 = torch.load(os.path.join(self.path, "resnet18.pth"), map_location=self.device)
        modelResNet18.load_state_dict(checkpointResNet18['model_state_dict'])
        self.models.append(modelResNet18)
        
        if(isTroubleshoot):
            print("Loaded ResNet18 without weights model.")

        # Load ResNet50 with weights
        modelResNet50Wts = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        num_ftrs = modelResNet50Wts.fc.in_features

        for param in modelResNet50Wts.parameters():
            param.requires_grad = False
        modelResNet50Wts.fc = nn.Linear(num_ftrs, 1)  # Binary classification

        checkpointResNet50Wts = torch.load(os.path.join(self.path, "resnet50_weights.pth"), map_location=self.device)
        modelResNet50Wts.load_state_dict(checkpointResNet50Wts['model_state_dict'])
        self.models.append(modelResNet50Wts)

        if(isTroubleshoot):
            print("Loaded ResNet50 with weights model.")

        # Load ResNet50 without weights
        modelResNet50 = models.resnet50().to(self.device)
        num_ftrs = modelResNet50.fc.in_features

        for param in modelResNet50.parameters():
            param.requires_grad = False
        modelResNet50.fc = nn.Linear(num_ftrs, 1)  # Binary classification

        checkpointResNet50 = torch.load(os.path.join(self.path, "resnet50.pth"), map_location=self.device)
        modelResNet50.load_state_dict(checkpointResNet50['model_state_dict'])
        self.models.append(modelResNet50)

        if(isTroubleshoot):
            print("Loaded ResNet50 without weights model.")

        # Load VGG16 with weights
        # modelVGG16Wts = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(self.device)

        # for param in modelVGG16Wts.parameters():
        #     param.requires_grad = False

        # num_features = 4096
        # features = list(modelVGG16Wts.classifier.children())[:-1]
        # features.extend([nn.Linear(num_features, 1)])  # Binary classification
        # modelVGG16Wts.classifier = nn.Sequential(*features)

        # checkpointVGG16Wts = torch.load(os.path.join(self.path, "vgg16_weights.pth"), map_location=self.device)
        # modelVGG16Wts.load_state_dict(checkpointVGG16Wts['model_state_dict'])
        # self.models.append(modelVGG16Wts)

        # if(isTroubleshoot):
        #     print("Loaded VGG16 with weights model.")

        # # Load VGG16 without weights
        # modelVGG16 = models.vgg16().to(self.device)
        
        # for param in modelVGG16.parameters():
        #     param.requires_grad = False

        # num_features = modelVGG16.classifier[6].in_features
        # features = list(modelVGG16.classifier.children())[:-1]
        # features.extend([nn.Linear(num_features, 1)])  # Binary classification
        # modelVGG16.classifier = nn.Sequential(*features)
        
        # checkpointVGG16 = torch.load(os.path.join(self.path, "vgg16.pth"), map_location=self.device)
        # modelVGG16.load_state_dict(checkpointVGG16['model_state_dict'])
        # self.models.append(modelVGG16)

        # if(isTroubleshoot):
        #     print("Loaded VGG16 without weights model.")

        # Load DenseNet121 with weights
        modelDensenetWts = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).to(self.device)
        num_ftrs = modelDensenetWts.classifier.in_features

        for param in modelDensenetWts.parameters():
            param.requires_grad = False
        
        modelDensenetWts.classifier = nn.Linear(num_ftrs, 2)  # Binary classification

        checkpointDensenetWts = torch.load(os.path.join(self.path, "densenet121_weights.pth"), map_location=self.device)
        modelDensenetWts.load_state_dict(checkpointDensenetWts['model_state_dict'])
        self.models.append(modelDensenetWts)

        if(isTroubleshoot):
            print("Loaded DenseNet121 with weights model.")

        # Load DenseNet121 without weights
        modelDensenet = models.densenet121().to(self.device)
        num_ftrs = modelDensenet.classifier.in_features

        for param in modelDensenet.parameters():
            param.requires_grad = False

        modelDensenet.classifier = nn.Linear(num_ftrs, 2)  # Binary classification

        checkpointDensenet = torch.load(os.path.join(self.path, "densenet121.pth"), map_location=self.device)
        modelDensenet.load_state_dict(checkpointDensenet['model_state_dict'])
        self.models.append(modelDensenet)

        if(isTroubleshoot):
            print("Loaded DenseNet121 without weights model.")

        # Load EfficientNet_B0 with weights
        modelEfficientNetWts = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(self.device)
        num_ftrs = modelEfficientNetWts.classifier[1].in_features

        for param in modelEfficientNetWts.parameters():
            param.requires_grad = False

        modelEfficientNetWts.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification

        checkpointEfficientNetWts = torch.load(os.path.join(self.path, "efficientnet_b0_weights.pth"), map_location=self.device)
        modelEfficientNetWts.load_state_dict(checkpointEfficientNetWts['model_state_dict'])
        self.models.append(modelEfficientNetWts)

        if(isTroubleshoot):
            print("Loaded EfficientNet_B0 with weights model.")

        # Load EfficientNet_B0 without weights
        modelEfficientNet = models.efficientnet_b0().to(self.device)
        num_ftrs = modelEfficientNet.classifier[1].in_features

        for param in modelEfficientNet.parameters():
            param.requires_grad = False
        
        modelEfficientNet.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification

        checkpointEfficientNet = torch.load(os.path.join(self.path, "efficientnet_b0.pth"), map_location=self.device)
        modelEfficientNet.load_state_dict(checkpointEfficientNet['model_state_dict'])
        self.models.append(modelEfficientNet)

        if(isTroubleshoot):
            print("Loaded EfficientNet_B0 without weights model.")

    def get_models(self):
        return self.models
    
    def get_models_names(self):
        return self.modelNames

# %% [markdown]
# # Running the model load

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

# %%

loadAllModels = loadModels(os.path.join("saved_models"), torch.device(device))

# %%
loaded_model_names = []
loaded_models = []

loaded_models = loadAllModels.load_models()
loaded_model_names = loadAllModels.modelNames


