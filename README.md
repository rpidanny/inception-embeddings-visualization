# Inception Embeddings Visualization

![alt text](https://github.com/rpidanny/assets/raw/master/inception-embeddings-visualization/screen.gif)

A script to generate tensorboard embeddings visualization of a group of images.

## Requirements

* python
* tensorflow
* tensorboard

## Installation

### Local

`pip install -r requierments.txt`

### Docker

#### Pull image

`docker pull rpidanny/tf1.8-gpu`

#### Run Container

Change directory to your project directory, than run in the interactive mode:

`nvidia-docker run -it -p 9001:9001 -v=$(pwd):$(pwd) --workdir=$(pwd) --rm rpidanny/tf1.8-gpu`

## Usage

### Generate Embeddings

`python main.py --image_dir=path_to_image_dir`

### Load Embeddings on Tensorboard

`tensorboard --logdir=/tmp/tensorboard_logs --port=9001`

Go to [http://localhost:9001](http://localhost:9001)
