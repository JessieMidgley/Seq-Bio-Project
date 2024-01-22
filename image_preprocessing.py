import numpy as np
import cv2
import pydicom
from PIL import Image
import skimage
from skimage.transform import resize
from cmap import Colormap
import tensorflow as tf


def remove_artefacts(img, shape=(224, 224)):
    """Removes all artefacts from the image.

            Parameters:
                img (np.array): Pixel array of the image (grayscale, values in range [0, 255]).
                shape (tuple): Shape of the image (width, height).

            Returns:
                A numpy array representing the processed image (grayscale, values in range [0, 255]).
    """

    # Remove borders (5px)
    rectangle = np.ones(shape=shape)
    rectangular_mask = cv2.rectangle(rectangle, pt1=(0, 0), pt2=(img.shape[1], img.shape[0]),
                                     color=(0, 0, 0), thickness=5)
    rectangular_mask = skimage.img_as_ubyte(rectangular_mask)
    remove_borders = cv2.bitwise_and(rectangular_mask, img)
    # Morphological Opening
    # TODO: Experiment with the threshold to obtain the best outcomes
    binary_mask = cv2.threshold(remove_borders, thresh=45, maxval=255, type=cv2.THRESH_BINARY)[1]
    kernel = np.ones((20, 20), np.uint8)
    opening = cv2.morphologyEx(binary_mask, op=cv2.MORPH_OPEN, kernel=kernel)
    # artefacts_removal = cv2.bitwise_and(opening, remove_borders)
    # Largest Contour
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = cv2.drawContours(opening, sorted_contours, 0, (255, 255, 255), thickness=1)
    final_image = cv2.bitwise_and(largest_contour, remove_borders)

    return final_image


def image_enhancement(img):
    """Applies Gamma Correction and CLAHE to enhance the quality of the image.

            Parameters:
                img (np.array): Pixel array of the image (grayscale, values in range [0, 255]).

            Returns:
                A numpy array representing the processed image (grayscale, values in range [0, 255]).
    """

    # Scale values to [0,1]
    img = img / 255
    # Apply Gamma Correction
    gamma = 0.5
    img = img**(1/gamma)
    # Scale back to [0, 255]
    img = img * 255
    img = img.astype(np.uint8)
    # Apply CLAHE twice
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = clahe.apply(img)
    # Green Fire Blue
    # cmap = Colormap('imagej:GreenFireBlue').to_mpl()
    # sm = plt.cm.ScalarMappable(cmap=cmap)
    # color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    # color_range = color_range.reshape(256, 1, 3)
    # img = cv2.applyColorMap(img, color_range)

    return img


def resize_image(img, maintain_aspect_ratio=False):
    """Resizes the image to 224px x 224px.

        Parameters:
            img (np.array): Pixel array of the image (grayscale, values in [0, 225]).
            maintain_aspect_ratio (bool): If true, the image is shrunk until it fits to 224px.
            The remaining areas are filled with black.

        Returns:
                A numpy array representing the resized image (grayscale, values in range [0, 255]).
    """

    if not maintain_aspect_ratio:
        return resize(img, output_shape=(224, 224), anti_aliasing=False, order=0)
    else:
        # We have to remove the artefacts now, because we cannot access the edges later
        img = remove_artefacts(img, shape=img.shape)
        img = Image.fromarray(img)
        width = img.size[0]
        height = img.size[1]
        ratio = width / height
        if width > height:
            width = 224
            height = int(224 / ratio)
        else:
            height = 224
            width = int(224 * ratio)
        img.thumbnail((width, height))
        new_img = Image.new("RGB", (224, 224))
        new_img.paste(img)
        return np.array(new_img)


def dicom_to_pixel_array(path):
    """Converts a DICOM file to numpy array.

            Parameters:
                path (str): The path to the DICOM file.

            Returns:
                A numpy array representing the processed image (grayscale, values in range [0, 255]).
    """

    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    # Convert the scale to [0,255]
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def preprocess_image(path, mode, model, maintain_aspect_ratio=False, dicom=False):
    """Applies all pre-processing steps to the image.

            Parameters:
                path (str): The path to the image file.
                mode (str): One of 'self-written', 'tf'.
                    - self-written: will apply our own self-written pre-processing steps.
                    - tf: will apply the preprocessing function, that the tf.keras.applications package provides.
                model (str): One of 'ResNet50', 'VGG16'.
                    - ResNet50: If mode is 'tf', will apply tf.keras.applications.resnet50.preprocess_input().
                    - VGG16: If mode is 'tf', will apply tf.keras.applications.vgg16.preprocess_input().
                maintain_aspect_ratio (bool): Whether to maintain aspect ratio when resizing the image.
                dicom (bool): Whether the images are in DICOM format (default: False).

            Returns:
                A numpy array representing the processed image.
    """

    if dicom:
        img_data = dicom_to_pixel_array(path)
    else:
        img_data = Image.open(path)
        img_data = np.array(img_data)

    # Resize the image to 224px x 224px
    img_data = resize_image(img_data, maintain_aspect_ratio=maintain_aspect_ratio)

    if mode == 'self-written':
        if not maintain_aspect_ratio:
            img_data = remove_artefacts(img_data)
        img_data = image_enhancement(img_data)
        # Since the ImageNet weights are trained on RGB images, convert the grayscale image to RGB
        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        # TODO: maybe there is a problem here because the values are not zero-centered with respect to the ImageNet
        #  dataset!
        # Each color channel is zero-centered (range[-1,1])
        # img_data = (img_data/127.5) - 1
        return img_data
    else:
        # The tensorflow preprocessing functions expect a 3D or 4D np.array with 3 color channels!
        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        if model == 'ResNet50':
            return tf.keras.applications.resnet50.preprocess_input(img_data)
        else:
            # model == 'VGG16'
            return tf.keras.applications.vgg16.preprocess_input(img_data)
