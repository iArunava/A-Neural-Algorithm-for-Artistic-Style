import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing import image as kp_image
from PIL import Image

def gram_matrix(style_tensor):
    # Getting the number of channels
    channels = style_tensor.shape[-1]

    # Reshaping style tensor to have image channels first
    a = tf.reshape(style_tensor, [-1, channels])

    # Getting the number of neurons in one row
    # (or channel as one row of 'a' now contains
    # all the neurons previously present in one channel)
    n = tf.shape(a)[0]

    # Getting the gram_matrix
    gram_matrix = tf.divide(tf.matmul(a, a, transpose_a=True), tf.cast(n, tf.float32))
    return gram_matrix

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

def preprocess_img(img):
    return tf.keras.applications.vgg19.preprocess_input(img)

def deprocess_img(img):
    img_copy = img.copy()

    if len(x.shape) == 4:
        img_copy = np.squeeze(x, 0)
    if len(x.shape) == 3:
        raise ValueError('Invalid input! \n Input image must be of dimensions \
                          [1, height, width, depth] or [height, width, depth]')

    # Undoing the mean preprocessing
    img_copy[:, :, 0] += 103.939
    img_copy[:, :, 1] += 116.779
    img_copy[:, :, 2] += 123.68

    # Flipping BGR to RGB
    img_copy = img_copy[:, :, ::-1]

    # Clipping pixel values that are either below 0 or above 255
    img_copy = np.clip(img_copy, 0, 255).astype('uint8')

    return img_copy
