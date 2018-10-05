# Inception Embeddings Visualization

A script to generate tensorboard embeddings visualization of a group of images.

## Requirements

* tensorflow
* tensorboard

## Usage

* `pip install -r requierments.txt`
* `python main.py --image_dir=path_to_image_dir`
* `tensorboard --logdir=/tmp/tensorboard_logs --port=9001`
* Go to `http://localhost:9001`