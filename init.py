import numpy as np
import tensorflow as tf
import argparse
import warnings

from tensorflow.python.keras import models
from tensorflow.python.keras import layers

from utils import *

FLAGS = None
WEIGHTS = './weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

def get_feature_representations(model, c_img, s_img, num_style_layers):
    # Converting np.ndarray to tensor
    c_img = tf.convert_to_tensor(c_img, name='content_image_to_tensor')
    s_img = tf.convert_to_tensor(s_img, name='style_image_to_tensor')

    # Running session to get the representation of the content and style image
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        content_outputs = sess.run(model(c_img))
        style_outputs = sess.run(model(s_img))

    # Getting the respective representations while ignoring the particular Layers
    # that we don't want in each of content and style outputs
    style_features = [s_out[0] for s_out in style_outputs[:num_style_layers]]
    content_features = [c_out[0] for c_out in content_outputs[num_style_layers:]]

    return style_features, content_features


def get_model(style_layers, content_layers):
    # Load the model without the fully connected layers and pretrained on the imagenet dataset
    vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights=WEIGHTS)

    # Set trainable to false as we don't need to train the network
    vgg19.trainable = False

    # Get output layers corresponding to style and content Layers
    style_outputs = [vgg19.get_layer(layer).output for layer in style_layers]
    content_outputs = [vgg19.get_layer(layer).output for layer in content_layers]

    # Combining the output layers of Interest
    model_outputs = style_outputs + content_outputs

    # Build and return the model
    return models.Model(vgg19.input, model_outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--style-image',
        type=str,
        default='',
        help='Path to the style image')

    parser.add_argument('--content-image',
        type=str,
        default='',
        help='Path to the content image')

    parser.add_argument('--max-dim',
        type=int,
        default=512,
        help='The Max dimensions to scale the input content and style images to. \
              Default: 512 \
              If using something other than default value, prefer powers of 2 \
              helps in computation, e.g, 128, 256 etc.')

    parser.add_argument('--learning-rate',
        type=float,
        default=5.0,
        help='The Learning Rate to use.')

    parser.add_argument('--style-weight',
        type=float,
        default=1e-2,
        help='The style weight to use.')

    parser.add_argument('--content-weight',
        type=float,
        default=1e-3,
        help='The content weight to use.')

    parser.add_argument('--show-images-at-start',
        type=bool,
        default=False,
        help='If true, the content and style images are shown at beginning.\
              Default: False')

    parser.add_argument('--show-warnings',
        type=bool,
        default=False,
        help='If true, all warnings are shown \
              Default: False')

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.style_image == '' or FLAGS.content_image == '':
        raise Exception('Path to style or content image not provided')

    if not FLAGS.show_warnings:
        warnings.filterwarnings('ignore')

    # Loading Style and Content Image
    style_img_arr = load_image(FLAGS.style_image, FLAGS.max_dim)#.astype('uint8')
    content_img_arr = load_image(FLAGS.content_image, FLAGS.max_dim)#.astype('uint8')

    if FLAGS.show_images_at_start:
        show_images({'Content':content_img_arr, 'Style':style_img_arr})

    # Preprocess Style and Content Image
    style_prepr_img = preprocess_img(style_img_arr)
    content_prepr_img = preprocess_img(content_img_arr)

    # Getting the intermediate Style and Content Layers of Interest
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    content_layers = ['block5_conv2']

    # Getting the number of content and style Layers
    num_style_layers = len(style_layers)
    num_content_layers = len(content_layers)

    # Build and get the model
    vgg19 = get_model(style_layers, content_layers)

    # Since we do not need to train the model, let's make its layers not trainable
    for layer in vgg19.layers:
        layer.trainable = False

    # Getting the style and content feature representations
    style_features, content_features = get_feature_representations(vgg19,
                                                                style_prepr_img,
                                                                content_prepr_img,
                                                                num_style_layers)

    # Getting the gram_matrices from the style representations
    gram_style_matrices = [gram_matrix(s_feature) for s_feature in style_features]

    # For getting summaries at regular intervals
    iter_count = 1

    # Variables to store the best loss and best image
    b_loss, b_img = float('inf'), None

    # Set the initial image
    init_img = content_prepr_img
    init_img = tf.Variable(init_img, dtype=tf.float32)

    # Initializing Adam Optimizer
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta=0.99, epsilon=1e-1)

    # Create a config to pass and compute the losses
    loss_weights = (FLAGS.style_weight, FLAGS_content_weight)
    cfg = {
        'model' : vgg19,
        'loss_weights' : loss_weights,
        'init_image' : init_img,
        'gram_style_matrices' : gram_style_matrices,
        'content_features' : content_features
    }
    
    print (style_img_arr.shape)
    print (content_img_arr.shape)
