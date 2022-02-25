#!/usr/bin/env python

# -*- coding: utf-8 -*-

__author__ = "Bryant Rodriguez"
__credits__ = ["Chris Fotache, Sean Psulkowski"]
#A large portion of this code was adapted from Chris Fotache's blog:
#https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Bryant Rodriguez"
__email__ = "bryant1.rodriguez@famu.edu"
__status__ = "Testing"

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import os
import time
import pickle

os.chdir('/home/bryant.rodriguez/Projects/Python/cnnadhere/cnnadhere/')

# Location where train/validation and test sets will be located
data_dir = '/home/bryant.rodriguez/Projects/Python/cnnadhere/cnnadhere/data/SideCam/train'
test_dir = '/home/bryant.rodriguez/Projects/Python/cnnadhere/cnnadhere/data/SideCam/test'


# Split dataset randomly and assign images from dataset to testing

def load_split_train_test(datadir, valid_size=.2):
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor(),
                                           ])
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor(),
                                           ])
    train_data = datasets.ImageFolder(datadir,
                                      transform=train_transforms)
    valid_data = datasets.ImageFolder(datadir,
                                      transform=valid_transforms)
    num_train = len(train_data)
    num_test = int(np.floor(0.1 * num_train))
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx, test_idx = indices[split + num_test:], indices[num_test:split + num_test], indices[:num_test]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=16)
    validloader = torch.utils.data.DataLoader(valid_data,
                                              sampler=valid_sampler, batch_size=16)

    # All of the images will be classified for testing in 'test_classifier.py'
    testloader = torch.utils.data.DataLoader(valid_data,
                                             sampler=test_sampler, batch_size=num_test)
    return trainloader, validloader, testloader


trainloader, validloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)

# Save testloader to be used in "test_classifier.py"

# torch.save(testloader, 'sidecam_testloader5.pt')

device = torch.device("cuda")

# Use pre-trained resnet50
model = models.resnet50(pretrained=True).cuda()

print(model)

# Initialize training parameters

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 10),
                         nn.LogSoftmax(dim=1))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)
print(str(device))
epochs = 20
steps = 0
running_loss = 0
print_every = 10
train_losses, valid_losses = [], []

# Initialize variables for stat tracking
epochStats = epochTime = epochValidationLoss = epochValidationAccuracy = epochTrainingLoss = []

# Start keeping track of time
t0 = time.time()

# file = open("epochstats", 'wt')

# Begin model training
for epoch in range(epochs):

    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss / len(trainloader))
            valid_losses.append(valid_loss / len(validloader))
            print(f" {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Validation loss: {valid_loss / len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy / len(validloader):.3f}")
            running_loss = 0
            model.train()

            t1 = int(time.time()) - int(t0)
            print("Time Elapsed: " + str(t1))
            epochTime.append(t1)

            # file.write(str(epochStats))
# torch.save(model, 'sidecam_model5.pth')
# file.close()


plt.plot(train_losses, linestyle='--', marker='o', label='Training loss')
plt.plot(valid_losses, linestyle='--', marker='o', label='Validation loss')
plt.xlabel('Number of Steps')
plt.ylabel('Loss Percentage')
plt.title('Training and Validation Results (Sidecam)')
plt.legend(frameon=False)
plt.show()
# with open('sidecam5.pickle', 'wb') as f:
#    pickle.dump([valid_losses, train_losses, epochTime], f)
print("Done.")
