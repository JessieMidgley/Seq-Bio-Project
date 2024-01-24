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
import scikitplot as skplt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.datasets import load_sample_image
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn import metrics
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
           mode (string): model that should get validated ["VGG16","ResNet50"]
           k (int): number of different trainings and test.


       Returns:
           A numpy array with the mean auc and accuracy of the model in the different runs with random train_test_splits

"""
def validation(images_trainval, labels_trainval, images_test, labels_test, mode, k):
    
    #images_trainval = images_trainval[0:100]
    #labels_trainval = labels_trainval[0:100]

    results_auc = []
    results_acc = []
    
    for i in range(k):
        
        
        if mode == "VGG16":
            model = BreastCancerClassification_VGG16.vgg16()
        elif mode == "ResNet50":
            model = BreastCancerClassification_ResNet50.resnet50()

        X_train, X_val, Y_train, y_val = train_test_split(images_trainval, labels_trainval, test_size=0.125, random_state=None)
        
        callback = EarlyStopping(monitor='loss', mode='auto', patience=3, verbose=1, start_from_epoch=5,
                                 restore_best_weights=True)
        model.fit(
            X_train,
            Y_train,
            batch_size=1,
            epochs=50,
            callbacks=[callback],
            verbose=2
        )
        
        
        predictions = model.predict(images_test)
        predictions = predictions.reshape(-1)
        fpr, tpr, thresholds = metrics.roc_curve(labels_test, predictions)
        label = "model " + str(i)
        plt.plot(fpr, tpr, label=label)

        # compute mean accuracy and mean auc
        result = model.evaluate(images_test,labels_test)
        results_auc.append(result[2])
        results_acc.append(result[1])
        
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Random classifier')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    title = 'ROC Curve on the test set with ' + str(k) + ' different train/validation splits' 
    plt.title(title)
    plt.legend(loc='lower right')
    
    acc = fmean(results_acc)
    auc = fmean(results_auc)
    plt.savefig("./roc_test.pdf")
    print("The mean accuracy is:")
    print(acc)
    print("The mean auc is:")
    print(auc)
    return acc, auc
    
