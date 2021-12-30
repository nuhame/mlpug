# MLPug
MLPug is a machine learning library agnostic framework for model training.

A lot of the functionality you need to train your machine learning model is 
independent of the machine learning library you're using, e.g. PyTorch and Tensorflow.

MLPug provides a single framework with a unified API for all such training functionality,
independent of the machine learning library you are using. 

Thus, when switching machine learning library, you can reuse your training code with no, or minimal, change!

## Dive right in!

### Run the repository examples

You can find the example code [here](mlpug/examples/documentation/). 
How MLPug is used in the examples is explained further [here](#hello-world-with-pytorch).

Clone the MLPug repo:

```
git clone https://github.com/nuhame/mlpug.git
```

#### MLPug with PyTorch
To run the PyTorch examples, install PyTorch first, further use Python >= 3.7.
```
cd mlpug

# MLPug Hello World example
python mlpug/examples/documentation/pytorch/hello_world.py

# MLPug Fashion MNIST example
# Run `fashion_mnist.py -h` for options
python mlpug/examples/documentation/pytorch/fashion_mnist.py
```

There are similar [examples for using MLPug with PyTorch/XLA](mlpug/examples/documentation/pytorch/xla) (Training with Pytorch on TPUs).

#### MLPug with Tensorflow
To run the Tensorflow examples, install Tensorflow first, further use Python >= 3.7.
```
cd mlpug

# MLPug Hello World example
# Run hello_world.py or hello_world_not_eager.py
python mlpug/examples/documentation/tensorflow/hello_world.py

# MLPug Fashion MNIST example
# Run `fashion_mnist.py -h` for options
python mlpug/examples/documentation/pytorch/fashion_mnist.py
```
### Use MLPug in your own project

```
pip install mlpug
```

```Python
# Using MLPug with PyTorch
import mlpug.pytorch as mlp
```

```Python
# Using MLPug with PyTorch/XLA (Training with Pytorch on TPUs)
import mlpug.pytorch.xla as mlp
```

```Python
# Using MLPug with Tensorflow
import mlpug.tensorflow as mlp
```

# What is MLPug?
MLPug is a machine learning library agnostic framework for model training.

A lot of the functionality you need to train your machine learning model is 
independent of the machine learning library you're using, e.g. PyTorch and Tensorflow.
For instance, 

 * checkpoint management,
 * evaluation of validation set loss and other custom metrics, 
 * progress logging, 
 * progress visualization using Tensorboard, 
 * the use of gradient accumulation to train with large batch sizes using limited GPU memory, etc.. 

You need such functionality no matter what machine learning framework you are using.

MLPug provides a single framework with a unified API for all such training functionality,
independent of the machine learning library you are using. This also implies that when you switch library
you can reuse your training code with no, or minimal, changes.

## Supported deep learning libraries
Currently, MLPug supports the following deep learning/machine learning libraries:

 * PyTorch
 * PyTorch/XLA (Training with Pytorch on TPUs)
 * Tensorflow (in development, some features not available yet)

## MLPug focus
Although MLPug should be able to deal with any training job, its functionality is mostly focussed on dealing with  
training large models on large datasets, using limited hardware (GPU or TPU) resources and memory.

## Almost at version 0.1!
MLPug is still in development. If you are having trouble using MLPug for your use case, or 
when you have found a bug, please file an issue.

## Contents
[Installing MLPug](#installing-mlpug) \
\
[Hello World](#hello-world) ([PT](#hello-world-with-pytorch) | 
[XLA](#hello-world-with-pytorchxla) | 
[TF](#hello-world-with-tensorflow)) 

[Feature parity list](#feature-parity-list)
\
\
\
The following sections are documentation **ToDo's**, but provide insight in to MLPug's features: \
[The `logs` object](#the-logs-object) \
\
[Callbacks and the training life cycle](#callbacks-and-the-training-life-cycle) \
\
[Progress Logging](#progress-logging) \
\
[Model components vs Training model](#model-components-vs-training-model) \
\
[Distributed training](#distributed-training) \
\
[Checkpoint management](#checkpoint-management) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Using the CheckpointManager](#using-the-checkpointmanager) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Using training checkpoints](#using-training-checkpoints) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Using model checkpoints](#using-model-checkpoints) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Checkpointing on error or interrupt](#checkpointing-on-error-or-interrupt) \
\
[MLPug metric evaluators](#mlpug-metric-evaluators) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Auxiliary batch training results](#auxiliary-batch-training-results) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Calculating custom metrics](#calculating-custom-metrics) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Conditional computation of metrics](#conditional-computation-of-metrics) \
\
[Batch chunking, dealing with GPU memory limits](#batch-chunking-dealing-with-gpu-memory-limits) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Gradient Accumulation](#gradient-accumulation) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Chunked Metric Computation](#chunked-metric-computation) \
\
[Using Tensorboard](#using-tensorboard) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Tensorboard made easy with AutoTensorboard](#tensorboard-made-easy-with-auto-tensorboard) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[More fine grained control](#more-fine-grained-control) \
\
[Learning Rate Scheduling](#learning-rate-scheduling) \
\
[Multi GPU training](#multi-gpu-training) \
\
[Mixed Precision Training](#mixed-precision-training) \
\
[CUDA Memory tools](#cuda-memory-tools) \
\
[Using multiple optimizers](#using-multiple-optimizers)

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

## Hello World!
This is the Hello World of training with MLPug. You will see that the usage of MLPug with Pytorch, 
Pytorch/XLA and Tensorflow is very similar.

For details please see :

 * [pytorch/hello_world.py](mlpug/examples/documentation/pytorch/hello_world.py),

 * [pytorch/xla/hello_world.py](mlpug/examples/documentation/pytorch/xla/hello_world.py), 

 * [tensorflow/hello_world.py](mlpug/examples/documentation/tensorflow/hello_world.py) and 
[tensorflow/hello_world_not_eager.py](mlpug/examples/documentation/tensorflow/hello_world_not_eager.py)
   
You can download and run these examples (for XLA you need to use a TPU on Google Cloud, or use Google Colab).

When reading through the explanation below it might be that you still have a lot of questions about the why and how of
training with MLPug, however I will expand the MLPug documentation soon, so you will get better insight.

### 'Hello World' with PyTorch
To use MLPug with Pytorch
```python
import mlpug.pytorch as mlp
```

Before we can start training we need an iterable dataset that can provide our training batches.

```python
training_dataset = torch.utils.data.DataLoader(training_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=3)
```

... and a model we want to train
```python
classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10))
```

MLPug needs a way to evaluate the loss of the model. One way to do that is to define a `TrainModel` that 
outputs the loss 
```python
class TrainModel(torch.nn.Module):
    def __init__(self, classifier):
        super(TrainModel, self).__init__()

        self.classifier = classifier
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, batch_data, evaluate_settings, inference_mode=None):
        images, true_labels = batch_data

        logits = self.classifier(images)
        return self.loss_func(logits, true_labels)

train_model = TrainModel(classifier)
```

To train the model we will also need an optimizer
```python
optimizer = torch.optim.Adam(classifier.parameters(), eps=1e-7)
```

To now use MLPug to start training, we need to create a `Trainer` which will be used by a `TrainingManager`.
```python
trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer, model_components=classifier)
```

MLPug uses a callback system allowing you to customize and extend the training functionality. 
The list of callback instances you provide the `TrainingManager` will be called using hooks at different stages of the 
training process.
```python
# At minimum you want to log the loss in the training progress
# By default the batch loss and the moving average of the loss are calculated and logged
loss_evaluator = mlp.evaluation.MetricEvaluator(trainer=trainer)
callbacks = [
    mlp.callbacks.TrainingMetricsLogger(metric_evaluator=loss_evaluator),
    # Calculate validation loss only once per epoch over the whole dataset
    mlp.callbacks.TestMetricsLogger(validation_dataset,
                                    'validation',
                                    metric_evaluator=loss_evaluator,
                                    batch_level=False),
    mlp.callbacks.LogProgress(log_period=progress_log_period, set_names=['training', 'validation']),
]
```

The `TrainingMetricsLogger` and the `TestMetricsLogger` callback instances log training and validation set loss values 
in a `logs` object that is passed through all callbacks during training. The `LogProgress` callback instance logs the 
metric values stored in the received `logs` object.

We can now instantiate the `TrainingManager` and pass it the `trainer`. 
```python
manager = mlp.trainers.TrainingManager(trainer,
                                       training_dataset,
                                       num_epochs=num_epochs,
                                       callbacks=callbacks)
```

Before we can start training we still have to provide the `train_model` to the trainer.
```python
trainer.set_training_model(train_model)
```

The final step is to actually start training:
```python
manager.start_training()
```

Running `pytorch/hello_world.py` finishes like this:
```text
###############################################################################
Epoch 9/9	READY - Duration 0:00:08
Moving average:
training       : loss          0.238.

Computed over dataset:
validation     : loss          0.346.



INFO    : TrainingManager::_train : Training completed. All good! ❤️

Using the classifier ...
real label = 9, predicted label = 9
```

### 'Hello World' with PyTorch/XLA

The Hello World example with PyTorch/XLA, is largely the same as with [PyTorch](#hello-world-with-pytorch). There are only
two small differences.

To use MLPug with Pytorch/XLA, load the correct backend
```python
import mlpug.pytorch.xla as mlp
```

Load your model on a TPU core:
```python
import torch_xla.core.xla_model as xm

...

device = xm.xla_device()

train_model = TrainModel(classifier, device)
classifier.to(device)
```

### 'Hello World' with Tensorflow
Below we will focus only on the minor differences between using MLPug with [PyTorch](#hello-world-with-pytorch) and Tensorflow.

To use MLPug with Tensorflow
```python
import mlpug.tensorflow as mlp
```

The only real difference is that, for Tensorflow, you can specify if the trainer needs to run in eager mode or not.
If not, you need to specify the input `batch_data_signature`.
```python
trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer,
                                      model_components=classifier,
                                      eager_mode=True)
```

```python
trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer,
                                      model_components=classifier,
                                      batch_data_signature=(tf.TensorSpec(shape=(None, 28, 28), dtype=tf.float64),
                                                            tf.TensorSpec(shape=(None,), dtype=tf.uint8),))
```
When you run [tensorflow/hello_world.py](mlpug/examples/documentation/tensorflow/hello_world.py) and 
[tensorflow/hello_world_not_eager.py](mlpug/examples/documentation/tensorflow/hello_world_not_eager.py) you will see
that when not running in eager mode, training is much faster.

Running `tensorflow/hello_world.py` finishes like this:
```text
###############################################################################
Epoch 9/9	READY - Duration 0:00:15
Moving average:
training       : loss          0.229.

Computed over dataset:
validation     : loss          0.370.



INFO    : TrainingManager::_train : Training completed. All good! ❤️

Using the classifier ...
real label = 9, predicted label = 9
```

Running `tensorflow/hello_world_not_eager.py` finishes like this:
```text
###############################################################################
Epoch 9/9	READY - Duration 0:00:06
Moving average:
training       : loss          0.229.

Computed over dataset:
validation     : loss          0.370.



INFO    : TrainingManager::_train : Training completed. All good! ❤️

Using the classifier ...
real label = 9, predicted label = 9
```

Note the difference in epoch duration!


## Feature parity list


|              Feature                          |   PyTorch   | PyTorch/XLA | Tensorflow  |     JAX     |           Comments               |
|-----------------------------------------------|-------------|-------------|-------------|-------------|----------------------------------|
| Callbacks and training life cycle             |      ✓      |      ✓      |      ✓      |             | |
| Progress Logging                              |      ✓      |      ✓      |      ✓      |             | |
| Distributed training                          |      ✓      |      ✓      |      ✓      |             | Both multi-GPU and multi-TPU support for PyTorch and TF.  TPU training with TF is untested |
| Model and training checkpoint management      |      ✓      |      ✓      |      ✓      |             | |
| Custom  metric evaluation                     |      ✓      |      ✓      |      ✓      |             | |
| Conditional evaluation of metrics             |      ✓      |      ✓      |      ✓      |             | |
| Batch Chunking: gradient accumulation         |      ✓      |      ✓      |      ❌     |             | TF ToDo |
| Batch Chunking: chunked evaluation of metrics |      ✓      |      ✓      |      ✓      |             | |
| Tensorboard support                           |      ✓      |      ✓      |      ✓      |             | Might be refactored |
| Learning Rate scheduling                      |      ✓      |      ✓      |      ✓      |             | Might be refactored |
| Mixed Precision Training                      |      ✓      |      ❌     |      ❌     |             | Should work with TF, but no specific support |
| Using multiple optimizers                     |      ✓      |      ✓      |      ✓      |             | |
| Multi-task training                           |      ❌     |     ❌      |     ❌      |             | ToDo |

