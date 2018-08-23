import numpy as np
import tensorflow as tf
import argparse

from utils import *

FLAGS = None

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

    parser.add_argument('--show-images-at-start',
        type=bool,
        default=False,
        help='If true, the content and style images are shown at beginning.\
              Default: False')

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.style_image == '' or FLAGS.content_image == '':
        raise Exception('Path to style or content image not provided')

    style_img_arr = load_image(FLAGS.style_image, FLAGS.max_dim).astype('uint8')
    content_img_arr = load_image(FLAGS.content_image, FLAGS.max_dim).astype('uint8')

    if FLAGS.show_images_at_start:
        show_images({'Content':content_img_arr, 'Style':style_img_arr})

    print (style_img_arr.shape)
    print (content_img_arr.shape)
