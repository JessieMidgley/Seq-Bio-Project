import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2

import image_preprocessing
import data_augmentation
import validation


def load_image_paths(image_dir='./venv/CBIS-DDSM/csv/dicom_info.csv'):
    """Loads the image file paths from dicom_info.csv.

            Parameters:
                image_dir (str): The path to the dicom_info.csv file.

            Returns:
                full_mammogram_images (list): A list of the paths to all the full mammogram images.
                cropped_images (list): A list of the paths to all the cropped images.
                roi_mask_images (list) : A list of the paths to all the ROI mask images.
    """

    data = pd.read_csv(image_dir)
    cropped_images = data[data.SeriesDescription == 'cropped images'].image_path
    cropped_images = cropped_images.apply(lambda x: './venv/' + x)
    full_mammogram_images = data[data.SeriesDescription == 'full mammogram images'].image_path
    full_mammogram_images = full_mammogram_images.apply(lambda x: './venv/' + x)
    roi_mask_images = data[data.SeriesDescription == 'ROI mask images'].image_path
    roi_mask_images = roi_mask_images.apply(lambda x: './venv/' + x)
    return full_mammogram_images, cropped_images, roi_mask_images


def fix_image_path(data_frame, only_full_images=True):
    """Changes the path to the DICOM files to the correct image path.

            Parameters:
                data_frame (pd.DataFrame): A dataframe containing the image paths.
                only_full_images (bool): Whether to only consider the full mammogram images.
    """

    full_mammogram_images, cropped_images, roi_mask_images = load_image_paths()
    to_drop = []
    for index, columns in enumerate(data_frame.values):
        # columns[11] is the image file path
        image_name = columns[11].split('/')[2]
        correct_path = full_mammogram_images[full_mammogram_images.str.contains(image_name, case=True)].values
        if len(correct_path) == 0:
            to_drop.append(index)
            # print('Found no image path for file {}'.format(image_name))
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


def train_test_validation_split(model, only_mass_cases=True, only_full_images=True, mode='tf',
                                maintain_aspect_ratio=False, dicom=False, augmentation=False):
    """Splits data into training, validation and test sets (70, 20, 10).

            Parameters:
                model (str): One of 'ResNet50', 'VGG16'.
                    - ResNet50: Preprocessing is adapted to the ResNet50 model.
                    - VGG16: Preprocessing is adapted to the VGG16 model.
                only_mass_cases (bool): If True, only mass cases will be considered.
                only_full_images (bool): If True, only full mammogram images will be considered.
                mode (str): One of 'self-written', 'tf'.
                    - self-written: will apply our own self-written pre-processing steps.
                    - tf: will apply the preprocessing function, that the tf.keras.applications package provides.
                maintain_aspect_ratio (bool): Whether to maintain aspect ratio when resizing the image.
                dicom (bool): Whether the images are in DICOM format (default: False).
                augmentation (bool): Whether to apply data augmentation to enlarge the training set.

            Returns:
                X_train, y_train (np.array): The processed training images and the training labels (one-hot encoded).
                X_val, y_val (np.array): The processed validation images and the validation labels (one-hot encoded).
                X_test, y_test (np.array): The processed test images and the test labels (one-hot encoded).
    """

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
    print('----------Preprocessing the images----------')
    full_cases['processed images'] = full_cases['image file path'].apply(lambda x: image_preprocessing.preprocess_image(
        x, mode=mode, model=model, maintain_aspect_ratio=maintain_aspect_ratio, dicom=dicom, augmentation=augmentation))
    # Apply class mapper to pathology column
    class_mapper = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}
    full_cases['labels'] = full_cases['pathology'].replace(class_mapper)

    X_resized = np.array(full_cases['processed images'].tolist())

    X_train, X_temp, y_train, y_temp = train_test_split(X_resized, full_cases['labels'].values, test_size=0.3,
                                                        random_state=108)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=108)

    if augmentation:
        # Apply data augmentation to enlarge the training set
        print('----------Applying data augmentation to the training set----------')
        X_train, y_train = data_augmentation.dataaugmentation(X_train, y_train,
                                                              maintain_aspect_ratio=maintain_aspect_ratio)
        if model == "VGG16":
            X_test = tf.keras.applications.vgg16.preprocess_input(X_test)
            X_val = tf.keras.applications.vgg16.preprocess_input(X_val)
        elif model == "ResNet50":
            X_test = tf.keras.applications.resnet50.preprocess_input(X_test)
            X_val = tf.keras.applications.resnet50.preprocess_input(X_val)

        n = 0
        for img in X_train:
            if model == "VGG16":
                X_train[n] = tf.keras.applications.vgg16.preprocess_input(img)
            elif model == "ResNet50":
                X_train[n] = tf.keras.applications.resnet50.preprocess_input(img)
            n = n + 1

        X_train = np.array(X_train)
        y_train = np.array(y_train)

    # Convert integer labels to one-hot encoded labels
    num_classes = 2
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    y_val = to_categorical(y_val, num_classes)

    return X_train, y_train, X_val, y_val, X_test, y_test


def resnet50():
    """Instantiates the ResNet50 architecture.

            Returns:
                A ResNet50 model with pretrained weights and custom layers.
    """

    # Create the ResNet-50 base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom classification layers on top of the base model
    x = base_model.output
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    predictions = Dense(2, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the base model except for the last layer
    # for layer in base_model.layers[:-1]:
    for layer in base_model.layers:
        layer.trainable = False

    # Learning rate decay
    # lr_schedule = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)

    # Compile the model
    model.compile(optimizer=Nadam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['AUC', 'accuracy'])

    return model


def train_model(model, model_type, only_mass_cases=True, only_full_images=True, mode='tf', maintain_aspect_ratio=False,
                dicom=False, augmentation=False):
    """Train the model on the CBIS-DDSM dataset and evaluate it on the test dataset.

            Parameters:
                model (tf.keras.Model): The model to be trained.
                model_type (str): One of 'ResNet50', 'VGG16'.
                    - ResNet50: Preprocessing is adapted to the ResNet50 model.
                    - VGG16: Preprocessing is adapted to the VGG16 model.
                only_mass_cases (bool): If True, only mass cases will be considered.
                only_full_images (bool): If True, only full mammogram images will be considered.
                mode (str): One of 'self-written', 'tf'.
                    - self-written: will apply our own self-written pre-processing steps.
                    - tf: will apply the preprocessing function, that the tf.keras.applications package provides.
                maintain_aspect_ratio (bool): Whether to maintain aspect ratio when resizing the image.
                dicom (bool): Whether the images are in DICOM format.
                augmentation (bool): Whether to apply data augmentation to enlarge the training set.

            Returns:
                The trained ResNet50 model.
    """

    # Load the training, validation and test data
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_validation_split(model=model_type,
                                                                                 only_mass_cases=only_mass_cases,
                                                                                 only_full_images=only_full_images,
                                                                                 mode=mode,
                                                                                 maintain_aspect_ratio=maintain_aspect_ratio,
                                                                                 dicom=dicom,
                                                                                 augmentation=augmentation)

    # Stop training when the validation loss has stopped decreasing
    callback = EarlyStopping(monitor='val_loss', mode='auto', patience=3, verbose=1, start_from_epoch=5,
                             restore_best_weights=True)

    train_datagen = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    # Apply augmentation to training data
    # train_data_augmented = train_datagen.flow(X_train, y_train, batch_size=16)

    # Train the model
    print('----------Training the model----------')
    history = model.fit(X_train,
                        y_train,
                        epochs=50,
                        validation_data=(X_val, y_val),
                        callbacks=[callback],
                        verbose=2)

    # Visualize training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.rcParams.update(bundles.icml2022(column='full', nrows=1, ncols=2, usetex=False))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(acc, label='Training')
    ax1.plot(val_acc, label='Validation')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('No. epoch')
    ax1.legend(loc='lower right')
    ax1.set_title('Training and Validation Accuracy')

    ax2.plot(loss, label='Training')
    ax2.plot(val_loss, label='Validation')
    ax2.set_ylabel('Categorical Cross-entropy')
    ax2.set_xlabel('No. epoch')
    ax2.legend(loc='upper right')
    ax2.set_title('Training and Validation Loss')
    plt.savefig('./training_results.pdf')

    print('----------Evaluating the model on the test dataset----------')
    results = model.evaluate(X_test, y_test)
    print("test loss, test AUC, test accuracy:", results)

    return model


def main():
    """
    model = resnet50()
    trained_model = train_model(model, model_type='ResNet50', only_mass_cases=False, only_full_images=True, mode='tf',
                                maintain_aspect_ratio=False, dicom=False, augmentation=False)

    """

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_validation_split(model="ResNet50",
                                                                                 only_mass_cases=False,
                                                                                 only_full_images=True,
                                                                                 mode="tf",
                                                                                 maintain_aspect_ratio=False,
                                                                                 dicom=False,
                                                                                 augmentation=False)
    images_train_val = np.concatenate((X_train, X_val), axis=0)
    labels_train_val = np.concatenate((y_train, y_val), axis=0)

    validation.validation(images_train_val, labels_train_val, X_test, y_test, model_type="ResNet50", k=5)


if __name__ == "__main__":
    main()
