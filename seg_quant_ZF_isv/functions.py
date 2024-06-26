import numpy as np
import cv2
import os
import csv
from PIL import Image
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from skimage import filters
from scipy.ndimage import convolve
from patchify import patchify

def multi_predict(resized_img, model, device):
    prediction = []
    # Patch image and make predictions on each patch
    patches_img = patchify(resized_img, (512, 512), step=512).squeeze()
    for i in range(patches_img.shape[0]):
        patch_image = patches_img[i, :, :]
        image = patch_image / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add channel dimension: (1, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            # Prediction
            pred_y = model(image)
            pred_y = torch.softmax(pred_y, dim=1)  # Apply softmax to get probabilities
            pred_y = torch.argmax(pred_y, dim=1)  # Get the predicted class labels
            pred_y = pred_y[0].cpu().numpy()  # Convert to numpy array

        if len(prediction) == 0:
            prediction = pred_y.astype(np.uint8)
        else:
            prediction = np.hstack((prediction, pred_y.astype(np.uint8)))

    return prediction

def prep_multi(img_path, mask_path):
    """ Reading image """
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add channel dimension: (1, 512, 512)
    image = image.astype(np.float32)
    image = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension: (1, 1, 512, 512)

    """ Reading mask """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure mask values are 0, 1, or 2
    mask = np.where(mask == 0, 0, mask)
    mask = np.where(mask == 128, 1, mask)
    mask = np.where(mask == 255, 2, mask)

    mask = mask.astype(np.int64)  # Ensure masks are integers for CrossEntropyLoss

    return image, mask


def resize_file(input_file):
    height, width = input_file.shape[0], input_file.shape[1]
    desired_height = 512

    aspect_ratio = width / height
    new_width = int(desired_height * aspect_ratio)
    resized_file = cv2.resize(input_file, (new_width, desired_height), interpolation=cv2.INTER_LINEAR)

    width_padding = (math.ceil(new_width / desired_height) * desired_height - new_width)
    input_file = Image.fromarray(resized_file)
    result = Image.new(input_file.mode, (new_width + width_padding, desired_height), 0)
    paste_x = (width_padding // 2) - 1
    result.paste(input_file, (paste_x, 0))
    return np.array(result)