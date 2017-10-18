"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
from PIL import Image

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
from tensorflow.contrib.slim.nets import inception
import time

from run_in_background import generate_in_background
from models import inception_model
from models import densenet_model

from step_pgd_attack import step_pgd_attack

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_v4', '', 'Path to checkpoint for inception v4 network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 8, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'max_fun_eval', 5, 'Maximum iterations per image')

tf.flags.DEFINE_integer(
    'max_ls', 5, 'Maximum number of line searches')

FLAGS = tf.flags.FLAGS


def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}

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
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].

  eps = 2.0 * FLAGS.max_epsilon / 255.0

  tf.logging.set_verbosity(tf.logging.INFO)
  all_images_target_class = load_target_class(FLAGS.input_dir)

  with tf.Graph().as_default():
    # Prepare graph

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    image_loader = generate_in_background(load_images(FLAGS.input_dir, batch_shape))

    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    model_dense = densenet_model('tf-densenet169.ckpt', 'densenet169')
    model_v3 = inception_model('inception_v3.ckpt')
    model_v3_adv = inception_model('adv_inception_v3.ckpt', new_scope='inception_v3_adv')
    model_incept_resnet_v2 = inception_model('inception_resnet_v2.ckpt', 'inception_resnet_v2')
    model_incept_resnet_adv_ens = inception_model('ens_adv_inception_resnet_v2.ckpt', 'inception_resnet_v2', new_scope='inception_resnet_v2_adv')
    model_v4 = inception_model('inception_v4.ckpt', 'inception_v4')

    probs = model_incept_resnet_v2.predict(x_input)
    pred = tf.argmax(probs, axis=1)

    model_list = [model_dense, model_v3, model_v3_adv, model_v4, model_incept_resnet_v2, model_incept_resnet_adv_ens]
    class combined_model:
      def predict(self, img):
        pred_list = [m.predict(img) for m in model_list]
        #pred_normalized = [logits - tf.reduce_logsumexp(logits, axis=1, keep_dims=True) for logits in pred_list]
        return tf.reduce_mean(pred_list, axis=0)

    all_models = combined_model()

    adversarial_generator = step_pgd_attack(all_models.predict, batch_shape, eps,
                                                               max_iter=FLAGS.max_fun_eval,
                                                               step_iter=4,
                                                               targeted=True,
                                                               initial_lr=48, #best: 48
                                                               lr_decay=0.9,
                                                               alpha=eps / 2)

    with tf.Session(config=config) as sess:
      model_dense.initialize(sess)
      model_incept_resnet_v2.initialize(sess)
      model_incept_resnet_adv_ens.initialize(sess)
      model_v3.initialize(sess)
      model_v3_adv.initialize(sess)
      model_v4.initialize(sess)

      batch_num = 0
      for filenames, images in image_loader:
        target_class_for_batch = (
            [all_images_target_class[n] for n in filenames]
            + [0] * (FLAGS.batch_size - len(filenames)))

        adv_images = adversarial_generator.generate(sess, images, target_class_for_batch)
        save_images(adv_images, filenames, FLAGS.output_dir)



if __name__ == '__main__':
  tf.app.run()
