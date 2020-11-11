'''
Only run once please.
This python file is to do data augmentation,
because therer are far less images in SmallKnots and LargeKnots. 
This module aims to solve the problems caused by
small and imbalanced datasets

'''

from PIL import ImageEnhance
import os
import numpy as np
from PIL import Image

# increase the bright of the image
def brightnessEnhancement(root_path, img_name):
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


# rotate the image
def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    random_angle = np.random.randint(-2, 2) * 90
    if random_angle == 0:
        rotation_img = img.rotate(-90)
    else:
        rotation_img = img.rotate(random_angle)
    return rotation_img


# filp the image
def flip(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return filp_img


def createImage(imageDir, saveDir):
    i = 0
    for name in os.listdir(imageDir):
        i = i + 1
        if name[-4:] == ".bmp":
            saveName1 = "flip" + str(i) + ".bmp"
            saveImage1 = flip(imageDir, name)
            saveImage1.save(os.path.join(saveDir, saveName1))
            saveName2 = "brightnessE" + str(i) + ".bmp"
            saveImage2 = brightnessEnhancement(imageDir, name)
            saveImage2.save(os.path.join(saveDir, saveName2))
            saveName3 = "rotate" + str(i) + ".bmp"
            saveImage = rotation(imageDir, name)
            saveImage.save(os.path.join(saveDir, saveName3))


imageDir = "./data/LargeKnots"
saveDir = "./data/LargeKnots"
createImage(imageDir, saveDir)
createImage(imageDir, saveDir)

imageDir = "./data/SmallKnots"
saveDir = "./data/SmallKnots"
createImage(imageDir, saveDir)
createImage(imageDir, saveDir)