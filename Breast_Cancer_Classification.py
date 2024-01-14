import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import PIL
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay


def load_image_paths(image_dir='./venv/CBIS-DDSM/csv/dicom_info.csv'):
    """Load the image file paths from dicom_info.csv"""
    data = pd.read_csv(image_dir)
    cropped_images = data[data.SeriesDescription == 'cropped images'].image_path
    cropped_images = cropped_images.apply(lambda x: './venv/' + x)
    full_mammogram_images = data[data.SeriesDescription == 'full mammogram images'].image_path
    full_mammogram_images = full_mammogram_images.apply(lambda x: './venv/' + x)
    roi_mask_images = data[data.SeriesDescription == 'ROI mask images'].image_path
    roi_mask_images = roi_mask_images.apply(lambda x: './venv/' + x)
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

    mass_cases_train = pd.read_csv('./venv/CBIS-DDSM/csv/mass_case_description_train_set.csv')
    mass_cases_test = pd.read_csv('./venv/CBIS-DDSM/csv/mass_case_description_test_set.csv')
    fix_image_path(mass_cases_train, only_full_images=only_full_images)
    fix_image_path(mass_cases_test, only_full_images=only_full_images)
    full_cases = pd.concat([mass_cases_train, mass_cases_test], axis=0)

    if not only_mass_cases:
        calc_cases_train = pd.read_csv('./venv/CBIS-DDSM/csv/calc_case_description_train_set.csv')
        calc_cases_test = pd.read_csv('./venv/CBIS-DDSM/csv/calc_case_description_test_set.csv')
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
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    y_val = to_categorical(y_val, num_classes)

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes


def image_processor(image_path, target_size=(224, 224)):
    absolute_image_path = os.path.abspath(image_path)
    image = tf.keras.preprocessing.image.load_img(absolute_image_path, target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    return tf.keras.applications.resnet50.preprocess_input(image)


def resnet50():
    # Load the training, validation and test data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = train_test_validation_split(only_mass_cases=False)

    # Create the ResNet-50 base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom classification layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Learning rate decay
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['AUC', 'accuracy'])

    # Augment data
    train_datagen = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    # Apply augmentation to training data
    train_data_augmented = train_datagen.flow(X_train, y_train, batch_size=16)

    # Stop training when the validation loss has stopped decreasing.
    callback = EarlyStopping(monitor='val_loss', mode='auto', patience=5, verbose=1, start_from_epoch=5,
                             restore_best_weights=True)

    # Train the model
    history = model.fit(train_data_augmented,
                        epochs=40,
                        validation_data=(X_val, y_val),
                        callbacks=[callback],
                        verbose=2)

    results = model.evaluate(X_test, y_test)

    return history, results


def main():
    history, results = resnet50()
    print('Evaluating the model on the test data:')
    print("test loss, test AUC, test accuracy:", results)
    plt.plot(history.history['loss'], label='Categorical Cross-entropy (training data)')
    plt.plot(history.history['val_loss'], label='Categorical Cross-entropy (validation data)')
    plt.title('Categorical Cross-entropy')
    plt.ylabel('Categorical Cross-entropy value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
