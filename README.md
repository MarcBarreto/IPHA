# Genetic IPHA (Iterative Post-hoc Attribution)

## Overview

IPHA is a novel approach to interpreting image classification models by treating interpretability as an optimization problem. The method focuses on maximizing the model's score through an innovative genetic algorithm approach that determines pixel importance in image classification.

## How It Works
The core functionality is based on optimizing the following function:
javascript

```bash
Importance(mask) = f(mask ⊙ x + (1-mask) ⊙ C)
```

where:
mask is a binary vector indicating which pixels are kept vs. replaced
- x is the input image
- C is a reference value
- ⊙ represents element-wise multiplication

The importance of a pixel is determined by comparing the classification score with and without that pixel. A higher score with the pixel included indicates its relevance to the classification.
## Key Features

- Uses genetic algorithms for optimization, avoiding gradient-dependent methods
- Works with complex neural network architectures
- Efficient exploration of solution space
- No explicit gradient calculations required

## Prerequisites
- Conda environment
- Python 3.x
- Dependencies listed in requirements.txt

## Installation

1. Create a new conda environment:
```bash
conda create --name ipha-env
conda activate ipha-env
```
2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage
Run the algorithm using:
```bash
python3 main.py [type] [model_path] [noise_type] [image_id] [save_path]
```

Parameters:
- Type: 0 for processing one image or 1 for processing multiple images
- model_path: Path to the classification model
- noise_type: Choose from [black, white, gaussian, norm_mean]
- image_id: image ID (0, 1, ... 9999) or the number of images to process
- save_path: Required only if you wish to save the image

Default Configuration

The `main.py` implementation was developed for use with:

- Model: [ResNet18 Model](https://www.kaggle.com/models/markbarreto/resnet_cifar10/PyTorch/default/1)
- Dataset: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

However, the `IPHA.py` algorithm can be adapted to work with other models and datasets.

## Contributing
Feel free to submit issues and enhancement requests.

## License
This project is licensed under the [MIT License](LICENSE). See the LICENSE file for more details.
