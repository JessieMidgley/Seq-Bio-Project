import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
from skimage import img_as_ubyte
from skimage.transform import resize
from PIL import Image
from cmap import Colormap


def remove_artefacts(img):
    """Remove artefacts from the image.

        img: numpy array, grayscale image with values between 0 and 255.
        output: numpy array, grayscale image with values between 0 and 255.
    """
    # Resize the image to 224px x 224px
    resized_img = resize(img, output_shape=(224, 224), anti_aliasing=False, order=0)
    # Remove borders (5px)
    rectangle = np.ones(shape=(224, 224))
    rectangular_mask = cv2.rectangle(rectangle, pt1=(0, 0), pt2=(resized_img.shape[1], resized_img.shape[0]),
                                     color=(0, 0, 0), thickness=5)
    rectangular_mask = img_as_ubyte(rectangular_mask)
    remove_borders = cv2.bitwise_and(rectangular_mask, resized_img)
    # Morphological Opening
    binary_mask = cv2.threshold(remove_borders, thresh=25, maxval=255, type=cv2.THRESH_BINARY)[1]
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
    cmap = Colormap('imagej:GreenFireBlue').to_mpl()
    sm = plt.cm.ScalarMappable(cmap=cmap)
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    color_range = color_range.reshape(256, 1, 3)
    img = cv2.applyColorMap(img, color_range)

    return img


def dicom_to_pixel_array(path):
    """Convert DICOM file to numpy array.

        path: path to DICOM file.
        output: numpy array, grayscale pixels with values between 0 and 255.
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    # Convert the scale to [0,255]
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def preprocess_image(path, dicom=True):
    """Apply all preprocessing steps to the image.

        path: path to the DICOM file
        output: numpy array, grayscale pixels with values between 0 and 255
    """
    if dicom:
        data = dicom_to_pixel_array(path)
    else:
        data = Image.open(path)
        data = np.array(data)

    data = remove_artefacts(data)
    # Maybe implement the remove_line step here!
    data = image_enhancement(data)
    # Each color channel is zero-centered (range[-1,1])
    # TODO: Maybe try to copy _preprocess_numpy_input() from the keras library github repo or without any scaling!
    # data = (data/127.5) - 1

    return data
