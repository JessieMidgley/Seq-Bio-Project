import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
import os
import PIL
import cv2
import tensorflow as tf
from keras import optimizers
from keras import backend as K
from keras import layers, models
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.layers import Conv2D, GlobalAveragePooling2D
from image_preprocessing import preprocess_image


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import EarlyStopping


def load_image_paths(image_dir='data/csv/dicom_info.csv'):
    """Load the image file paths from dicom_info.csv"""
    data = pd.read_csv(image_dir)
    cropped_images = data[data.SeriesDescription == 'cropped images'].image_path
    cropped_images = cropped_images.apply(lambda x: './data/' + x)
    full_mammogram_images = data[data.SeriesDescription == 'full mammogram images'].image_path
    full_mammogram_images = full_mammogram_images.apply(lambda x: './data/' + x)
    roi_mask_images = data[data.SeriesDescription == 'ROI mask images'].image_path
    roi_mask_images = roi_mask_images.apply(lambda x: './data/' + x)
    return full_mammogram_images, cropped_images, roi_mask_images



def fix_image_path(data_frame, only_full_images=True):
    """Change the path to the .dcm file to the correct image path"""
    full_mammogram_images, cropped_images, roi_mask_images = load_image_paths()
    to_drop = []
    for index, columns in enumerate(data_frame.values):
        # columns[11] is the image file path
        image_name = columns[11].split('/')[2]
        correct_path = full_mammogram_images[full_mammogram_images.str.contains(image_name, case=True)].values
        if len(correct_path) == 0:
            to_drop.append(index)
            print('Found no image path for file {}'.format(image_name))
        else:
            assert len(correct_path) == 1
            data_frame.iloc[index, 11] = correct_path[0]
        if not only_full_images:
            # columns[12] is the cropped image file path
            image_name = columns[12].split('/')[2]
            correct_path = cropped_images[cropped_images.str.contains(image_name, case=True)].values
            if len(correct_path) == 0:
                print('Found no cropped image path for file {}'.format(image_name))
            else:
                assert len(correct_path) == 1
                data_frame.iloc[index, 12] = correct_path[0]
            # columns[13] is the ROI mask file path
            image_name = columns[13].split('/')[2]
            correct_path = roi_mask_images[roi_mask_images.str.contains(image_name, case=True)].values
            if len(correct_path) == 0:
                print('Found no ROI mask path for file {}'.format(image_name))
            else:
                assert len(correct_path) == 1
                data_frame.iloc[index, 13] = correct_path[0]

    # drop columns with invalid image paths
    data_frame.drop(index=to_drop, inplace=True)



def train_test_validation_split(only_mass_cases=True, only_full_images=True):
    """Split data into train, test, and validation sets (70, 20, 10)"""

    mass_cases_train = pd.read_csv('data/csv/mass_case_description_train_set.csv')
    mass_cases_test = pd.read_csv('data/csv/mass_case_description_test_set.csv')
    fix_image_path(mass_cases_train, only_full_images=only_full_images)
    fix_image_path(mass_cases_test, only_full_images=only_full_images)
    full_cases = pd.concat([mass_cases_train, mass_cases_test], axis=0)

    if not only_mass_cases:
        calc_cases_train = pd.read_csv('data/csv/calc_case_description_train_set.csv')
        calc_cases_test = pd.read_csv('data/csv/calc_case_description_train_set.csv')
        fix_image_path(calc_cases_train, only_full_images=only_full_images)
        # Problem with calc_cases_test: 282 out of 326 image paths are invalid!
        fix_image_path(calc_cases_test, only_full_images=only_full_images)
        full_cases = pd.concat([full_cases, calc_cases_train, calc_cases_test], axis=0)

    # Apply preprocessor to data
    full_cases['processed images'] = full_cases['image file path'].apply(lambda x: image_processor(x))
    # Apply class mapper to pathology column
    class_mapper = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}
    full_cases['labels'] = full_cases['pathology'].replace(class_mapper)

    X_resized = np.array(full_cases['processed images'].tolist())
    X_train, X_temp, y_train, y_temp = train_test_split(X_resized, full_cases['labels'].values, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # Convert integer labels to one-hot encoded labels
    num_classes = len(full_cases['labels'].unique())
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes



def image_processor(image_path, target_size=(224, 224)):
    return preprocess_image(image_path, 'self-written', 'VGG16', maintain_aspect_ratio=False, dicom=False)




def vgg16():
    # Load the training, validation and test data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = train_test_validation_split(only_mass_cases=False)

    # Create the ResNet-50 base model
    base_model = VGG16(input_shape=(224, 224, 3), weights="imagenet", include_top=False)

    x = base_model.output

    x = tf.keras.layers.Flatten()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    # Add custom classification layers on top of the base model

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'AUC'])

    # Augment data
    train_datagen = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                      shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    # Apply augmentation to training data
    #train_data_augmented = dataaugmentation_paper(X_train,Y_train)

    # Stop training when the validation loss has stopped decreasing.
    callback = EarlyStopping(monitor='val_loss', mode='auto', patience=5, verbose=1,
                             restore_best_weights=True)

    # Train the model
    history = model.fit(X_train,y_train,
                        epochs=20,
                        validation_data=(X_val, y_val),
                        callbacks=[callback],
                        verbose=2)

    results = model.evaluate(X_test, y_test)

    return history, results


def main():
    history, results = vgg16()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    AUC = history.history['auc']
    val_AUC = history.history['val_auc']
    epochs = range(1, len(acc) +1)

    plt.rcParams.update(bundles.icml2022(column='full', nrows=1, ncols=2, usetex=False))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(acc, label='Training')
    ax1.plot(val_acc, label='Validation')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('No. epoch')
    ax1.legend(loc='lower right')
    ax1.set_title('Training and Validation Accuracy')

    ax2.plot(AUC, label='Training')
    ax2.plot(val_AUC, label='Validation')
    ax2.set_ylabel('AUC')
    ax2.set_xlabel('No. epoch')
    ax2.legend(loc='upper right')
    ax2.set_title('Training and Validation AUC')
    plt.savefig('./VGG16_with_preprocessing_maxpooling.pdf')


if __name__ == "__main__":
    main()
