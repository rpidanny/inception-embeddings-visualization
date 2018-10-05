"""
Helper functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import imghdr

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.platform import gfile

from config import DATA_URL,\
  BOTTLENECK_TENSOR_NAME,\
  BOTTLENECK_TENSOR_SIZE,\
  JPEG_DATA_TENSOR_NAME,\
  RESIZED_INPUT_TENSOR_NAME

def create_inception_graph(model_dir):
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = os.path.join(
        model_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def maybe_download_and_extract(dest_directory):
  """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.
  """
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                             filepath,
                                             _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values

def get_image_lists(image_dir):
  extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
  file_list = []
  dir_name = os.path.basename(image_dir)
  print("Looking for images in '" + dir_name + "'")
  for extension in extensions:
    file_glob = os.path.join(image_dir, '*.' + extension)
    file_list.extend(gfile.Glob(file_glob))
  if not file_list:
    print('No files found')
    return []
  images = []
  for file_name in file_list:
    typ = imghdr.what(file_name)
    if typ is 'jpeg': # pylint: disable=R0123
      base_name = os.path.basename(file_name)
      images.append(file_name)
      # print(base_name)
    else:
      print("Invalid Jpeg : Deleting image from trait..")
      tf.gfile.Remove(file_name)
  print('Fount {} images'.format(len(images)))
  return images

def get_image_list_embeddings(image_lists, sess, jpeg_data_tensor, bottleneck_tensor):
  print('Getting embeddings')
  embeddings = []
  for idx, image_path in enumerate(image_lists):
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
      continue
    image_data = gfile.FastGFile(image_path, 'rb').read()
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    embeddings.append(bottleneck_values)
    # bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    # with open(bottleneck_path, 'w') as bottleneck_file:
    #   bottleneck_file.write(bottleneck_string)
    sys.stdout.write('\r{:.2f}% complete'.format((idx/len(image_lists)) * 100))
    sys.stdout.flush()
  embeddings = np.array(embeddings)
  sys.stdout.write('\r{:.2f}% complete'.format(100.00))
  sys.stdout.flush()
  print ("\nFeature vectors shape:", embeddings.shape) 
  print ("Num of images:", embeddings.shape[0])
  print ("Size of individual feature vector:", embeddings.shape[1])
  return tf.Variable(embeddings, name='features')

def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

def parseArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/tensorboard_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  return parser.parse_known_args()
