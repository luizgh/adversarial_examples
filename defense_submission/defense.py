"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread
from scipy.ndimage import zoom, rotate, gaussian_filter

import tensorflow as tf
from run_in_background import generate_in_background

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim

from models import inception_model
from models import densenet_model



tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_v4', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images

def random_crop(img, input_shape):
    img_shape = img.shape
    max_y = img_shape[0] - input_shape[0]
    max_x = img_shape[1] - input_shape[1]

    start_y = np.random.choice(max_y, 1)[0]
    start_x = np.random.choice(max_x, 1)[0]

    cropped = img[start_y: start_y + input_shape[0], start_x:start_x + input_shape[1]]
    return cropped

MIN_ZOOM = 1.02
MAX_ZOOM = 1.06
MAX_ROTATE = 4
MIN_BLUR = 0.1
MAN_BLUR = 0.3

def process_image(img):
    img = zoom(img, np.random.uniform(MIN_ZOOM, MAX_ZOOM), order=0)
    img = rotate(img, np.random.uniform(-MAX_ROTATE, MAX_ROTATE), order=0)
    img = gaussian_filter(img, np.random.uniform(-MIN_BLUR, MAN_BLUR))
    img = random_crop(img, (299,299))
    return img

def preprocess_images(images):
    new_imgs = [process_image(img) for img in images]
    return np.stack(new_imgs)

def load_and_process_images(input_dir, batch_shape):
    for f, i in load_images(input_dir, batch_shape):
        yield f, preprocess_images(i)

def main(_):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)
  image_iterator = generate_in_background(load_and_process_images(FLAGS.input_dir, batch_shape))

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model_v3 = inception_model('inception_v3.ckpt')
    model_v3_adv = inception_model('adv_inception_v3.ckpt', new_scope='inception_v3_adv')
    model_incept_resnet_v2 = inception_model('inception_resnet_v2.ckpt', 'inception_resnet_v2')
    model_incept_resnet_adv_ens = inception_model('ens_adv_inception_resnet_v2.ckpt', 'inception_resnet_v2', new_scope='inception_resnet_v2_adv')
    model_v4 = inception_model('inception_v4.ckpt', 'inception_v4')
    model_dense = densenet_model('tf-densenet169.ckpt', 'densenet169')

    probs = model_dense.predict(x_input) + model_v3.predict(x_input) + model_v3_adv.predict(x_input) + model_v4.predict(x_input) + model_incept_resnet_v2.predict(x_input) + model_incept_resnet_adv_ens.predict(x_input) 

    preds = tf.argmax(probs, axis=1)

    with tf.Session(config=config) as sess:
      model_dense.initialize(sess)
      model_incept_resnet_v2.initialize(sess)
      model_incept_resnet_adv_ens.initialize(sess)
      model_v3.initialize(sess)
      model_v3_adv.initialize(sess)
      model_v4.initialize(sess)

      with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
        for filenames, images in image_iterator:
          images = preprocess_images(images)
          labels = sess.run(preds, feed_dict={x_input: images})
          for filename, label in zip(filenames, labels):
            out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
  tf.app.run()
