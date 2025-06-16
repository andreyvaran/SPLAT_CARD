from PIL import Image, ExifTags

import numpy as np


def orientation(image):

    try:

        for tagg in ExifTags.TAGS.keys():

            if ExifTags.TAGS[tagg] == "Orientation":

                exif = image._getexif()

                if exif[tagg] == 3:

                    image = image.rotate(180, expand=True)

                elif exif[tagg] == 6:

                    image = image.rotate(270, expand=True)

                elif exif[tagg] == 8:

                    image = image.rotate(90, expand=True)

                break

        return image

    except (AttributeError, KeyError, IndexError, TypeError):

        return image


def resize_std(image):

    return image.resize((1024, 1024))


def grayscale(img):

    return Image.fromarray(np.dot(np.array(img)[:, :, :3], [0.2125, 0.7154, 0.0721]))


def preprocess(img):

    img = orientation(img)

    img = resize_std(img)

    img = grayscale(img)

    grayscl = np.array(img)

    grayscl = np.stack((grayscl,) * 3, axis=-1)

    return grayscl


def restore_bbox(bbox, scale_new, scale_old):

    temp_bbox = []

    return temp_bbox
