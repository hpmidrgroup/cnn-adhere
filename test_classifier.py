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
from torchvision import datasets, transforms
import os
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import pickle

# Location where this program is being run from
os.chdir('/home/bryant.rodriguez/Projects/Python/cnnadhere/cnnadhere/')

# Location of original dataset

data_dir = '/home/bryant.rodriguez/Projects/Python/cnnadhere/cnnadhere/data/SideCam/train/'
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('sidecam_model5.pth')
model.eval()


# Load image and return prediction output

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    input = image_tensor
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


# Parameter is an int corresponding to number of images for training

def get_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    loader = torch.load('sidecam_testloader5.pt')
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels, classes


to_pil = transforms.ToPILImage()

# Retrieve images that were randomly chosen when the classifier was trained

images, labels, classes = get_images(94)

fig = plt.figure(figsize=(100, 100))
classOneIncorrect = classTwoIncorrect = classThreeIncorrect = classOneCorrect = classTwoCorrect = classThreeCorrect = trueCounter = falseCounter = 0
y_true = []
y_pred = []
y_str = []
infTime = []
t0 = time.time()

# Perform class prediction on all images, then compare predictions to actual
# classes.
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)

    res = int(labels[ii]) == index

    if index == 0:
        guess = 'FAIL'
    elif index == 1:
        guess = 'PASS'
    else:
        guess = 'SLIDE'
    y_pred.append(guess)

    if str(labels[ii]) == "tensor(0)":
        truth = 'FAIL'
        if res == True:
            classOneCorrect += 1

        else:
            classOneIncorrect += 1
        y_true.append(truth)

    elif str(labels[ii]) == "tensor(1)":
        truth = 'PASS'
        if res == True:
            classTwoCorrect += 1

        else:
            classTwoIncorrect += 1
        y_true.append(truth)

    else:
        truth = 'SLIDE'
        if res == True:
            classThreeCorrect += 1

        else:
            classThreeIncorrect += 1
        y_true.append(truth)

    t1 = int(time.time()) - int(t0)
    infTime.append(int(t1))
    print(str(classes[index]))
    print("Prediction: " + guess)
    print("True Label: " + truth)
    print("Result: " + str(res))
    print("Time Elapsed: " + str(t1))

    if res == True:
        trueCounter += 1
    else:
        falseCounter += 1

print(str(y_pred))
print(str(y_true))

# An inefficient way of doing this, but it was fast.
print("Total class 1 (FAIL)true predictions: " + str(classOneCorrect))
print("Total class 1 (FAIL) false predictions: " + str(classOneIncorrect))
print("Total class 1 (FAIL) accuracy on testing data was " + str(
    (classOneCorrect / (classOneCorrect + classOneIncorrect))))
print("Total class 2 (PASS)true predictions: " + str(classTwoCorrect))
print("Total class 2 (PASS)false predictions: " + str(classTwoIncorrect))
print("Total class 2 (PASS) accuracy on testing data was " + str(
    (classTwoCorrect / (classTwoCorrect + classTwoIncorrect))))
print("Total class 3 (SLIDE) true predictions: " + str(classThreeCorrect))
print("Total class 3 (SLIDE) false predictions: " + str(classThreeIncorrect))
print("Total class 3 (SLIDE) accuracy on testing data was " + str(
    (classThreeCorrect / (classThreeCorrect + classThreeIncorrect))))
print("Total true predictions: " + str(trueCounter))
print("Total false predictions: " + str(falseCounter))
print("Total accuracy on testing data was " + str((trueCounter / (trueCounter + falseCounter))))

# Create and display confusion matrix with resulting of classification

cm = confusion_matrix(y_true, y_pred, labels=['PASS', 'SLIDE', 'FAIL'], normalize='true')

disp = ConfusionMatrixDisplay(cm, display_labels=['PASS', 'SLIDE', 'FAIL']).plot(cmap='Blues', include_values=True)
plt.title('Sidecam Model 5')
plt.show()

# Save important variables to pickle file
with open('sidecam5_inference.pickle', 'wb') as f:
    pickle.dump([y_pred, y_true, infTime], f)

print("Done.")

