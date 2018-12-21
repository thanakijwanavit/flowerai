# Imports here
import numpy as np
from torchvision import datasets,transforms
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
from PIL import Image
from sklearn import preprocessing
from PIL import Image, ImageOps
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import json
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import time
import operator
import sys
from get_input_args import get_input_args
import torchvision.models as models
from choosemodel import choosemodel
from deeplearning import do_deep_learning
from time import time
import pprint
pp = pprint.PrettyPrinter(indent=4)

## create variable to get inputs from the command line
in_arg= get_input_args()
learning_rate=in_arg.learning_rate
#Create data dir
data_dir = in_arg.directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.Resize(255, interpolation=2),
                                       transforms.RandomResizedCrop(244),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

verify_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=verify_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# TODO: Build and train your network
#### make models

print('getting model',in_arg.arch)
model = choosemodel(in_arg.arch)




#criterion and optimizer

criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)







# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

dropout=0.2
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, in_arg.hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1',nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(in_arg.hidden_units, in_arg.hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('drop2',nn.Dropout(p=dropout)),
                          ('fc3', nn.Linear(in_arg.hidden_units, in_arg.hidden_units)),
                          ('relu3', nn.ReLU()),
                          ('drop3',nn.Dropout(p=dropout)),
                          ('fc4', nn.Linear(in_arg.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

### Create cat to name dict
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
cat_to_name_dict={int(i):j for i,j in cat_to_name.items()}









# Learning function


### select device
device='cpu'if in_arg.gpu==False else 'gpu'


#### Training
print('training with ',device)
t0=time()
#model = do_deep_learning(model, trainloader, in_arg.epochs, 40, criterion, optimizer,validloader, device)

## test deep learning without external function
# method for validation
epochs=in_arg.epochs
def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

## Train the network
steps=0
model.to('cuda')
for e in range(epochs):
    running_loss = 0
    model.train()
    for images, labels in trainloader:
        steps += 1

        images, labels = images.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            # change nn to eval mode for inference
            model.eval()

            # turn off gradients for validation
            with torch.no_grad():
                valid_loss, valid_accuracy = validation(model, validloader, criterion)

            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Train Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss: {:.4f}".format(valid_loss/len(validloader)),
                  "Validation Accuracy: {:.4f}".format(valid_accuracy/len(validloader)))

            running_loss = 0











print('the model took',time()-t0,'seconds to train')
pp.pprint(model.__dict__)


save_name=in_arg.save_dir
##### save model
print('save model as', save_name)
# TODO: Save the checkpoint
#torch.save(model.state_dict(), save_name)
model.class_to_idx = train_data.class_to_idx
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'cat_to_name_dict':cat_to_name_dict,
              'class_to_idx':model.class_to_idx,
              'arch':in_arg.arch}

torch.save(checkpoint, in_arg.save_dir)
