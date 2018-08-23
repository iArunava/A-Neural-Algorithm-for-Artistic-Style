import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing import image as kp_image
from PIL import Image

def load_image(path, max_dim):
    img = Image.open(path)

    # Max of height and width
    longer_side = max(img.size)

    # Scaling factor for the height and width
    scale = max_dim / longer_side

    # Resizing image with the scaling factor
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)

    # Getting matrix from image
    img = kp_image.img_to_array(img)

    # Broadcast the image such that it has a batch dimension
    img = np.expand_dims(img, axis=0)

    return img

def show_images(images):
    num_images = len(images)
    for i, img_name in enumerate(images.keys()):
        img = images[img_name]
        img = np.squeeze(img, axis=0)
        plt.subplot(1, num_images, i+1)
        plt.title(img_name)
        plt.imshow(img)
    plt.show()
    return
