# MLPug
MLPug is a machine learning library agnostic framework for model training.

So much of the functionality you need to train your machine learning model is 
independent of the machine learning library your, e.g. PyTorch and Tensorflow.
For instance, 

 * checkpoint management,
 * evaluation of validation set loss and other metrics, 
 * progress logging, 
 * progress visualization using Tensorboard, 
 * the use of gradient accumulation to train with large batch sizes using limited GPU memory, etc.. 

You need such functionality no matter what machine learning framework you are using.

MLPug provides a single framework with a unified API for all such training functionality
independent of the framework you are using. This also implied that when you switch frameworks
you can reuse your training code with no, or minimal, changes.

## Supported backends
Currently, MLPug supports the following deep learning/machine learning library 'backends':

 * PyTorch
 * Tensorflow (in development, some feature not available yet)
   
Further, support for the following machine learning library backends are planned: 
 * Microsoft DeepSpeed
 * Jax

If you like your favorite machine learning library to be supported, please file an issue!

## Almost at version 0.1!
MLPug is still in development. If you are having trouble using MLPug for your use case or 
when you have found a bug, please file an issue.


## Installing MLPug
Please ensure that you are using Python3.7+.

Install as follows:
```
pip install mlpug
```

### Usage with PyTorch
When you want to use MLPug with PyTorch, you will need to install it:
```
pip install torch torchvision
```

### Usage with Tensorflow
When you want to use MLPug with Tensorflow, you will need to install it:
```
pip install tensorflow
```

# Hello World!
This is the Hello World of training with MLPug.





