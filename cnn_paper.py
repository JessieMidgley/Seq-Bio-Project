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
from PIL import Image
from PIL import ImageOps, ImageFilter
from keras.preprocessing.image import ImageDataGenerator
from skimage import exposure
#import imagej import IJ
'''
     input: csvfile, fileformat
     output: dictionary (key: imagefilepath, value: label)
'''
def extractimglabel(csvfile, fileformat):

    #create lookupdictionary
    lookup = {}
    with open("data/csv/dicom_info.csv", newline='') as dicom:
        dicomreader = csv.reader(dicom, delimiter=',')
        next(dicomreader, None)
        for entry in dicomreader:
            if entry[1] not in lookup:
                lookup[entry[1]] = entry[29]
    imglabel = {}
    num = 0
    with open(csvfile, newline='') as csvf:
        csvreader = csv.reader(csvf, delimiter=',')
        next(csvreader, None)
        for img in csvreader:
            if fileformat == 'ROI':
                dir = img[13].split('/')[2]
            elif fileformat == 'full':
                dir = img[11].split('/')[2]
            elif fileformat == 'cropped':
                dir = img[12].split('/')[2]

            found = False
            
            for key,value in lookup.items():
                if dir in key and fileformat in value:
                    
                    path = key[9:]
                    if img[9] == 'MALIGNANT':
                        imglabel['data'+path] = 0
                    if img[9] == 'BENIGN':
                        imglabel['data'+path] = 1
                    found = True
                    
            if not found:
                num = num + 1
   
    return imglabel
        
datatrainmassfull = extractimglabel("data/csv/mass_case_description_train_set.csv","full")
datatestmassfull = extractimglabel("data/csv/mass_case_description_test_set.csv","full")
datamassfull = {**datatrainmassfull, **datatestmassfull}
#datatraincalccropped = extractimglabel("data/csv/calc_case_description_train_set.csv","cropped")
#datatestcalccropped = extractimglabel("data/csv/calc_case_description_test_set.csv","cropped")
#datacalccropped = {**datatraincalccropped, **datatestcalccropped}

'''
    input:  dictionary (key: imagefilepath, value: label)
    output: array of images with height and width of 224 and list of labels
'''
def vgg_transform_feature(data):         
    images = []
    labels = list(data.values())
    for img in data.keys():
        with Image.open(img) as image:
            width = image.size[0]
            height = image.size[1]
            ratio = width/height
            if width > height:
                width = 224
                height = int(224/ratio)
            else:
                height = 224
                width = int(224*ratio)
            image.thumbnail((width,height))
            blackimage = Image.new("RGB", (224,224))
            blackimage.paste(image)
            images.append(blackimage)  
    images = np.array(images)
    images.reshape(images.shape[0],images.shape[1],images.shape[2],3)
    return images,labels

imgfull,labfull = vgg_transform_feature(datamassfull)

# remove white borders
def prepare_images(images):
    preparedimages = []
    for image in images:
        img = Image.fromarray(image)
        blackimage = Image.new("RGB", (224,224),color="black")
        innerimage = ImageOps.crop(img,border=5)
        blackimage.paste(innerimage,(5,5))
        preparedimages.append(blackimage)
    preparedimages = np.array(preparedimages)
    preparedimages.reshape(preparedimages.shape[0],preparedimages.shape[1],preparedimages.shape[2],3)
    return preparedimages

#remove artefacts
def remove_artefacts(images):
    preparedimages = []
    for image in images:
        img = Image.fromarray(image)
        filterimg = img.filter(ImageFilter.BoxBlur(15))
        #filterimg.show()
        maskimage = filterimg.convert("L")
        maskimage = maskimage.point( lambda p: 255 if p > 80 else 0)
        maskimage = maskimage.filter(ImageFilter.MaxFilter(25))
        blackimage = Image.new("RGB", (224,224),color="black")
        preimage = Image.composite(img,blackimage,maskimage)
        preparedimages.append(preimage)
    preparedimages = np.array(preparedimages)
    preparedimages.reshape(preparedimages.shape[0],preparedimages.shape[1],preparedimages.shape[2],3)
    return preparedimages
        
def gammacorrect(images):
    preparedimages = []
    for image in images:
        #image = image/255
        imagesh = Image.fromarray(image)
        corrected = exposure.adjust_gamma(image,2)
        img = Image.fromarray(corrected)
        preparedimages.append(img)
    preparedimages = np.array(preparedimages)
    preparedimages.reshape(preparedimages.shape[0],preparedimages.shape[1],preparedimages.shape[2],3)
    return preparedimages

def clahe(images):
    preparedimages = []
    for image in images:
        cvimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
        claheimage = clahe.apply(cvimage)
        claheimage = clahe.apply(claheimage)
        pilimage = cv2.cvtColor(claheimage, cv2.COLOR_GRAY2RGB)
        preimage = Image.fromarray(pilimage)
        preparedimages.append(preimage)
        
    preparedimages = np.array(preparedimages)
    preparedimages.reshape(preparedimages.shape[0],preparedimages.shape[1],preparedimages.shape[2],3)
    return preparedimages

#preppipeline (remove_artefacts sometimes removes information that is possible necessary)
def preppipeline(images):
    preparedimages = prepare_images(images)
    #preparedimages = remove_artefacts(preparedimages)
    preparedimages = gammacorrect(preparedimages)
    preparedimages = clahe(preparedimages)
    return preparedimages
    
img = preppipeline(imgfull)
lab = labfull
X_train, X_test, Y_train, Y_test = train_test_split(img , lab, train_size=0.80, random_state=42) #42 data is shuffled befor splitting


def dataaugmentation_paper(img,lab):
    augmentedimages = []
    augmentedlabels = []
    n = 0
    while n < len(img):

        image = Image.fromarray(img[n])

        verticalflip = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        horizontalflip = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        verticalhorizontalflip = verticalflip.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        rotateplus = image.rotate(35, fillcolor="black")
        rotateminus = image.rotate(-35, fillcolor="black")
        rotateverticalplus = verticalflip.rotate(35,fillcolor="black")
        rotateverticalminus = verticalflip.rotate(-35,fillcolor="black")
        rotatehorizontalplus = horizontalflip.rotate(35, fillcolor="black")
        rotatehorizontalminus = horizontalflip.rotate(-35, fillcolor="black")
        
        augmentedimages.append(image)
        augmentedimages.append(verticalflip)
        augmentedimages.append(horizontalflip)
        augmentedimages.append(verticalhorizontalflip)
        augmentedimages.append(rotateplus)
        augmentedimages.append(rotateminus)
        augmentedimages.append(rotatehorizontalplus)
        augmentedimages.append(rotatehorizontalminus)
        augmentedimages.append(rotateverticalplus)
        augmentedimages.append(rotateverticalminus)

        i = 0
        while i <= 11:
            augmentedlabels.append(lab[n])
            i = i + 1
        n = n + 1
    augmentedimages = np.array(augmentedimages)
    augmentedimages.reshape(augmentedimages.shape[0],augmentedimages.shape[1],augmentedimages.shape[2],3)
    return augmentedimages, augmentedlabels

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

#print parameters in outputfile
print("training rate 0,0001, mass case, Flatten, Dense(1), no augmentation, no image data generator, preprocessing")


vgg16 = VGG16(input_shape=(224,224,3),weights="imagenet",include_top=False)

# set trainable and untrainalble layers
for layer in vgg16.layers:      
  layer.trainable = False


    
x = tf.keras.layers.Flatten()(vgg16.output)
output = layers.Dense(1, activation='softmax')(x)

model = Model(inputs=vgg16.input, outputs=output)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
  loss=tf.keras.losses.BinaryCrossentropy(),
  optimizer=optimizer,
  metrics=['accuracy',tf.keras.metrics.AUC()]
)

model.summary()
print(len(X_train))
print(len(Y_train))
result = model.fit(
    np.array(X_train),
    np.array(Y_train),
    epochs=50,
    validation_data=(X_test, np.array(Y_test)),
    verbose=2
)

def report(model, aug = False):
    if aug:
        xtest = x_test
        y_true = y_test
    else:
        xtest = X_test
        y_true = Y_test
    y_pred = []
    for i in model.predict(xtest,batch_size=10,verbose=0):
        y_pred.append(np.argmax(i))
    print(classification_report(y_true, y_pred, target_names= ['benign','malignant']))

report(model)


