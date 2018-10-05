from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.platform import gfile

from helper import maybe_download_and_extract, \
  create_inception_graph, \
  parseArguments, \
  get_image_lists, \
  images_to_sprite, \
  get_image_list_embeddings

FLAGS = None

def main(_): #pylint: disable-msg=too-many-statements
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)

  # Set up the pre-trained graph.
  maybe_download_and_extract(FLAGS.model_dir)
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph(FLAGS.model_dir))
  images = get_image_lists(FLAGS.image_dir)

  # create image sprite
  image_data = []
  for image in images:
    data = cv2.imread(image)
    data = cv2.resize(data, (224, 224))
    image_data.append(data)
  image_data = np.array(image_data)
  sprite = images_to_sprite(image_data)
  cv2.imwrite(os.path.join(FLAGS.summaries_dir, 'images_sprite.png'), sprite)

  # print(images)
  with tf.Session() as sess:
    # get image embeddings
    features = get_image_list_embeddings(images, sess, jpeg_data_tensor, bottleneck_tensor)
    
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, os.path.join(FLAGS.summaries_dir, 'embeddings.ckpt'))
    
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    # Comment out if you don't want sprites
    embedding.sprite.image_path = os.path.join(FLAGS.summaries_dir, 'images_sprite.png')
    embedding.sprite.single_image_dim.extend([image_data.shape[1], image_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(FLAGS.summaries_dir), config)

if __name__ == '__main__':
  FLAGS, unparsed = parseArguments()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
