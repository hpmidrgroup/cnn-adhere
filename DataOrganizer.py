from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

device = ("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.DataFrame(columns=["img_name","label"])
train_df["img_name"] = os.listdir("train/")
for idx, i in enumerate(os.listdir("train/")):
    if "PASS" in i:
        train_df["label"][idx] = 0
    if "FAIL" in i:
        train_df["label"][idx] = 1
    if "SLIDE" in i:
        train_df["label"][idx] = 2
    

train_df.to_csv (r'train_csv.csv', index = False, header=True)