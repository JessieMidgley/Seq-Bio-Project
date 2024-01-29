from PIL import Image
import numpy as np
import image_preprocessing


def dataaugmentation(images, labels, maintain_aspect_ratio=False):
    """Augment images.

            Parameters:
                images (np.array): Array of pixel arrays of the images.
                labels (np.array): Array of the labels that match the images.
                maintain_aspect_ratio (bool): Whether to maintain aspect ratio when resizing the image.
            Returns:
                Two numpy arrays containing the augmented images and the corresponding labels.
    """

    augmented_images = []
    augmented_labels = []
    n = 0

    while n < len(images):

        image = Image.fromarray(images[n])
        vertical_flip = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        horizontal_flip = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        vertical_horizontal_flip = vertical_flip.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        rotate_plus = image.rotate(30, fillcolor="black", expand=True)
        rotate_plus = image_preprocessing.resize_image(np.array(rotate_plus),
                                                       maintain_aspect_ratio=maintain_aspect_ratio)

        rotate_minus = image.rotate(-30, fillcolor="black", expand=True)
        rotate_minus = image_preprocessing.resize_image(np.array(rotate_minus),
                                                        maintain_aspect_ratio=maintain_aspect_ratio)

        rotate_vertical_plus = vertical_flip.rotate(30, fillcolor="black", expand=True)
        rotate_vertical_plus = image_preprocessing.resize_image(np.array(rotate_vertical_plus),
                                                                maintain_aspect_ratio=maintain_aspect_ratio)

        rotate_vertical_minus = vertical_flip.rotate(-30, fillcolor="black", expand=True)
        rotate_vertical_minus = image_preprocessing.resize_image(np.array(rotate_vertical_minus),
                                                                 maintain_aspect_ratio=maintain_aspect_ratio)

        image = image_preprocessing.resize_image(np.array(image),
                                                 maintain_aspect_ratio=maintain_aspect_ratio)
        vertical_flip = image_preprocessing.resize_image(np.array(vertical_flip),
                                                         maintain_aspect_ratio=maintain_aspect_ratio)
        horizontal_flip = image_preprocessing.resize_image(np.array(horizontal_flip),
                                                           maintain_aspect_ratio=maintain_aspect_ratio)
        vertical_horizontal_flip = image_preprocessing.resize_image(np.array(vertical_horizontal_flip),
                                                                    maintain_aspect_ratio=maintain_aspect_ratio)

        augmented_images.append(image)
        augmented_images.append(vertical_flip)
        augmented_images.append(horizontal_flip)
        augmented_images.append(vertical_horizontal_flip)
        augmented_images.append(rotate_plus)
        augmented_images.append(rotate_minus)
        augmented_images.append(rotate_vertical_plus)
        augmented_images.append(rotate_vertical_minus)

        i = 0
        while i <= 7:
            augmented_labels.append(labels[n])
            i = i + 1
        n = n + 1

    print(f'----------Data augmentation: {n} images were augmented----------')

    augmented_images = np.array(augmented_images)
    augmented_images.reshape(augmented_images.shape[0], augmented_images.shape[1], augmented_images.shape[2], 3)
    return augmented_images, augmented_labels
