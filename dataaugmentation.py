from PIL import Image
import numpy as np

""" augment images

        Parameters:
            imgs (np.array): Array of pixel arrays of the images.
            labs (np.array): Array of the labels that match the images.
        Returns:
            Two numpy arrays containing the augmented images as well as the labels.
    
"""
def dataaugmentation(imgs,labs):
    augmentedimages = []
    augmentedlabels = []
    n = 0
    while n < len(imgs):

        image = Image.fromarray(imgs[n])

        verticalflip = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        horizontalflip = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        verticalhorizontalflip = verticalflip.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        rotateplus = image.rotate(45, fillcolor="black")
        rotateminus = image.rotate(-45, fillcolor="black")
        rotateverticalplus = verticalflip.rotate(45,fillcolor="black")
        rotateverticalminus = verticalflip.rotate(-45,fillcolor="black")
        rotatehorizontalplus = horizontalflip.rotate(45, fillcolor="black")
        rotatehorizontalminus = horizontalflip.rotate(-45, fillcolor="black")
        
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
        while i <= 9:
            augmentedlabels.append(labs[n])
            i = i + 1
        n = n + 1
        
    augmentedimages = np.array(augmentedimages)
    augmentedimages.reshape(augmentedimages.shape[0],augmentedimages.shape[1],augmentedimages.shape[2],3)
    return augmentedimages, augmentedlabels

