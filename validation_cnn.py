import cv2
import csv
import os
from itertools import islice
from sklearn.model_selection import train_test_split
import numpy as np
from keras.applications.vgg16 import VGG16
from keras import layers, optimizers
from keras.layers import Flatten
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.datasets import load_sample_image
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from PIL import Image
from PIL import ImageOps, ImageFilter
from keras.preprocessing.image import ImageDataGenerator
from skimage import exposure
import random
from tensorflow.keras.callbacks import EarlyStopping
from statistics import *
import BreastCancerClassification_VGG16
"""validation of model

       Parameters:
           images (np.array): Array of pixel array of images.
           labels (np.array): Array of corresponding labels.
           model: model that should get validated
           k (int): number of different trainings and test.

       Returns:
           A numpy array with the mean auc and accuracy of the model in the different runs with random train_test_splits

"""
def validation(images, labels, k):
    
    images = images[0:100]
    labels = labels[0:100]

    # split data in five parts of equal size
    splitted_images = np.array_split(images,k)
    splitted_labels = np.array_split(labels,k)

    results_auc = []
    results_acc = []
    
    for i in range(k):
        print(i)
        model = BreastCancerClassification_VGG16.vgg16()
        
        y_test = splitted_labels[i]
        y_train = []
        for j in range(k):
            if j != i:
                y_train.append(splitted_labels[j])
        y_train = np.concatenate(y_train,0)
        
        X_test = splitted_images[i]
        X_train = []
        for j in range(k):
            if j != i:
                X_train.append(splitted_images[j])
        X_train = np.concatenate(X_train,0)
        
        # !adjust number of epochs to best performance on validation set
        model.fit(
            X_train,
            y_train,
            batch_size=1,
            epochs=10,
            verbose=2
        )
        
        result = model.evaluate(X_test,y_test)
        results_auc.append(result[2])
        results_acc.append(result[1])
        
    
    acc = fmean(results_acc)
    auc = fmean(results_auc)
    
    return acc, auc
    
