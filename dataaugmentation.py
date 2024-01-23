from PIL import Image
import numpy as np
import image_preprocessing

""" augment images

        Parameters:
            imgs (np.array): Array of pixel arrays of the images.
            labs (np.array): Array of the labels that match the images.
        Returns:
            Two numpy arrays containing the augmented images as well as the labels.
    
"""
def dataaugmentation(imgs,labs,maintain_aspect_ratio=False):
    augmentedimages = []
    augmentedlabels = []
    n = 0
    while n < len(imgs):

        image = Image.fromarray(imgs[n])
        verticalflip = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        horizontalflip = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        verticalhorizontalflip = verticalflip.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        rotateplus = image.rotate(30, fillcolor="black",expand=True)
        rotateplus = image_preprocessing.resize_image(np.array(rotateplus),maintain_aspect_ratio=maintain_aspect_ratio)
        
        rotateminus = image.rotate(-30, fillcolor="black",expand=True)
        rotateminus = image_preprocessing.resize_image(np.array(rotateminus),maintain_aspect_ratio=maintain_aspect_ratio)
        
        rotateverticalplus = verticalflip.rotate(30, fillcolor="black",expand=True)
        rotateverticalplus = image_preprocessing.resize_image(np.array(rotateverticalplus),maintain_aspect_ratio=maintain_aspect_ratio)
        
        rotateverticalminus = verticalflip.rotate(-30, fillcolor="black",expand=True)
        rotateverticalminus = image_preprocessing.resize_image(np.array(rotateverticalminus),maintain_aspect_ratio=maintain_aspect_ratio)
        
        image = image_preprocessing.resize_image(np.array(image),maintain_aspect_ratio=maintain_aspect_ratio)
        verticalflip = image_preprocessing.resize_image(np.array(verticalflip),maintain_aspect_ratio=maintain_aspect_ratio)
        horizontalflip = image_preprocessing.resize_image(np.array(horizontalflip),maintain_aspect_ratio=maintain_aspect_ratio)
        verticalhorizontalflip = image_preprocessing.resize_image(np.array(verticalhorizontalflip),maintain_aspect_ratio=maintain_aspect_ratio)
        
        augmentedimages.append(image)
        augmentedimages.append(verticalflip)
        augmentedimages.append(horizontalflip)
        augmentedimages.append(verticalhorizontalflip)
        augmentedimages.append(rotateplus)
        augmentedimages.append(rotateminus)
        augmentedimages.append(rotateverticalplus)
        augmentedimages.append(rotateverticalminus)

        i = 0
        while i <= 7:
            augmentedlabels.append(labs[n])
            i = i + 1
        n = n + 1
        
    augmentedimages = np.array(augmentedimages)
    augmentedimages.reshape(augmentedimages.shape[0],augmentedimages.shape[1],augmentedimages.shape[2],3)
    return augmentedimages, augmentedlabels



