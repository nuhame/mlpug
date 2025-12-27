# MLPug
MLPug is a library for training and evaluating Machine Learning (ML) models, able to use different 
ML libraries as backends. 

A lot of the functionality you need to train and evaluate your model is 
independent of the ML library you're using, such as PyTorch, Jax, Apple MLX or TinyGrad. 
MLPug aims to provide a single framework with a unified API for all such training and evaluation functionality,
independent of the ML library you are using. 

**Thus, when switching ML library, you don't have to learn a new training API and you can reuse your own training code 
with no, or minimal, change! 🤩🎉**

## MLPug is at version 0.1!
MLPug is still in development. If you are having trouble using MLPug for your use case, or 
when you have found a bug, please file an issue.

## Dive right in! Installing MLPug as a Python package for your project.
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


## Basic template of usage

```
# ################# SETUP ##################
# A TrainModel uses your model to calculate and return the training loss and other basic data
train_model = TrainModel(classifier)

# The trainier knows how to train the model on a batch
trainer = mlp.trainers.DefaultTrainer(
    optimizers=optimizer,
    model_components=classifier
)

# At minimum, you want to log the loss to track training progress
# By default the batch loss and the moving average of the loss are calculated and logged
loss_evaluator = mlp.evaluation.MetricEvaluator(trainer=trainer)
callbacks = [
    mlp.callbacks.TrainingMetricsLogger(metric_evaluator=loss_evaluator),
    # Calculate validation loss only once per epoch over the whole dataset
    mlp.callbacks.DatasetMetricsLogger(validation_dataset,
                                       'validation',
                                       metric_evaluator=loss_evaluator,
                                       batch_level=False),
    # Log the progress during training                                       
    mlp.callbacks.LogProgress(log_period=progress_log_period, set_names=['training', 'validation']),
]

manager = mlp.trainers.TrainingManager(trainer,
                                       training_dataset,
                                       num_epochs=num_epochs,
                                       callbacks=callbacks)

trainer.set_training_model(train_model)

# ################# START! #################
manager.start_training()
```

Also see section [Explaining the MLPug Hello World example](#explaining-the-mlpug-hello-world-example).

## Running the repository examples
You can find the example code [here](examples/).

In the MLPug examples package there are three examples:
 * A simple Hello World example
 * A Fashion MNIST example with multiple commandline options for common MLPug features
 * [PyTorch only] A complex example where we train a GPT2 chatbot based on the Persona dataset

To get started with the examples clone the MLPug repo:
```
git clone https://github.com/nuhame/mlpug.git
```

To run the examples, be sure to be in the root of the MLPug repository.
```
cd mlpug
```

Ensure that the current directory (the MLPug repo root) is added to the python path, such that python can
find the example scripts:
```
export PYTHONPATH=.:$PYTHONPATH
```

It is advised to run the examples in a Python virtual environment:
```
python3 -m venv  ~/.virtualenvs/mlpug
# Activate the virtual environment
source ~/.virtualenvs/mlpug/bin/activate
```

In your `>=Python3.9` virtual environment, install the basic requirements:
```
pip install -r requirements.txt 
```
### Examples of using MLPug with PyTorch

Next to examples for PyTorch, there are equivalent examples for using MLPug with PyTorch/XLA, see the
`pytorch/xla` directories of the examples. For XLA you need to use a TPU on Google Cloud, or
use Google Colab. Here we focus on PyTorch. 

#### Hello world
Next to the default requirements, for the Hello World example, install the example specific requirements in your 
`>=Python3.9` virtual environment:
```
pip install -r requirements.txt
pip install -r examples/hello_world/pytorch/requirements.txt
```

```
# MLPug Hello World example
python examples/hello_world/pytorch/train.py
```

#### Fashion MNIST
For the Fashion MNIST example install the following requirements:
```
pip install -r requirements.txt
pip install -r examples/fashion_mnist/pytorch/requirements.txt
```

```
# MLPug Fashion MNIST example
python examples/fashion_mnist/pytorch/train.py

# If you have multiple GPUs
python examples/fashion_mnist/pytorch/train.py --distributed

# Use a batch size of 32 with micro-batches of 8 for gradient accumulation
python examples/fashion_mnist/pytorch/train.py --batch-size 32 --micro-batch-size 8
```

Some other useful flags are:
 * `--use-mixed-precision`: When flag is set, mixed precision will be applied during training
 * `--eager-mode`: When flag is set, forward and backward computation graphs will NOT be compiled (i.e. eager mode)

Run `train.py -h` for all options

#### Training a Persona Chatbot based on GPT2
Next to the default requirements, install the example specific requirements in your 
`>=Python3.9` virtual environment:
```
pip install -r requirements.txt 
pip install -r examples/persona_chatbot/requirements.txt 
pip install -r examples/persona_chatbot/pytorch/requirements.txt 
```

What micro-batch size to use for gradient accumulation depends on your system.
The following setup assumes multiple GPUs (`--distributed`).
```
python examples/persona_chatbot/pytorch/train.py \
                 --experiment-name persona-bot-experiment \
                 --num-dataloader-workers 2 \
                 --num-choices 8 \
                 --sequence-length-outlier-threshold 0.05 \
                 --batch-size 32 \
                 --micro-batch-size 8 \
                 --learning-rate 1e-4 \
                 --distributed \
                 --num-epochs 6 \
                 --progress-log-period 10
```

See `train.py -h` to see all options.

## What is MLPug?
MLPug is a library for training and evaluating Machine Learning (ML) models, able to use different 
ML libraries as backends.

A lot of the functionality you need to train and evaluate your machine learning model is 
independent of the machine learning library you're using, such as PyTorch, Jax, Apple MLX or TinyGrad.
For instance, 

 * checkpoint management,
 * evaluation of validation set loss and other custom metrics, 
 * progress logging, 
 * progress visualization using Tensorboard, 
 * the use of gradient accumulation to train with large batch sizes using limited GPU memory, etc.. 

You need such functionality no matter what machine learning framework you are using.

MLPug aims to provide a single framework with a unified API for all such training and evaluation functionality,
independent of the ML library you are using. This also implies that when you switch library
you can reuse your training code with no, or minimal, changes.

## Supported machine learning libraries
Currently, MLPug supports the following deep learning/machine learning libraries:

 * PyTorch
 * PyTorch/XLA (Training with Pytorch on TPUs)

The ambition is to also add Jax, Apple MLX and TinyGrad as backends.

## MLPug focus
Although MLPug should be able to deal with any training job, its functionality is mostly focussed on dealing with  
training large models on large datasets, using limited hardware (GPU or TPU) resources and memory.

## Detailed documentation
The following sections are documentation **ToDo's**, but provide insight in to MLPug's features: 

[Feature parity list](#feature-parity-list)


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
[Gradient Accumulation with Micro-batches](#gradient-accumulation-with-micro-batches) \
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

### Explaining the MLPug Hello World example

#### Hello World with PyTorch

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
trainer = mlp.trainers.DefaultTrainer(
    optimizers=optimizer, 
    model_components=classifier
)
```

MLPug uses a callback system allowing you to customize and extend the training functionality. 
The list of callback instances you provide the `TrainingManager` will be called using hooks at different stages of the 
training process.

```python
# At minimum, you want to log the loss to track training progress
# By default the batch loss and the moving average of the loss are calculated and logged
loss_evaluator = mlp.evaluation.MetricEvaluator(trainer=trainer)
callbacks = [
    mlp.callbacks.TrainingMetricsLogger(metric_evaluator=loss_evaluator),
    # Calculate validation loss only once per epoch over the whole dataset
    mlp.callbacks.DatasetMetricsLogger(validation_dataset,
                                       'validation',
                                       metric_evaluator=loss_evaluator,
                                       batch_level=False),
    # Log the progress during training
    mlp.callbacks.LogProgress(log_period=progress_log_period, set_names=['training', 'validation']),
]
```

The `TrainingMetricsLogger` and the `DatasetMetricsLogger` callback instances log training and validation set loss values 
in a `logs` object that is passed through all callbacks during training. The `LogProgress` callback instance logs the 
metric values stored in the received `logs` object to the console (stdout).

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

Running `examples/hello_world/pytorch/train.py` finishes like this:
```text
###############################################################################
Epoch 9/9	READY - Duration 0:00:08
Computed over sliding window:
training       : loss          0.239.

Computed over dataset:
validation     : loss          0.355.



INFO    : TrainingManager::_train : Training completed. All good! ❤️

Using the classifier ...
real label = 9, predicted label = 9
```

#### Hello World with PyTorch/XLA
The Hello World example with PyTorch/XLA, is largely the same as with [PyTorch](#hello-world-with-pytorch). 
There are only two small differences.

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

## Feature parity list


|              Feature                          |   PyTorch   | PyTorch/XLA |     JAX     |           Comments               |
|-----------------------------------------------|-------------|-------------|-------------|----------------------------------|
| Callbacks and training life cycle             |      ✓      |      ✓      |             | |
| Progress Logging                              |      ✓      |      ✓      |             | |
| Distributed training                          |      ✓      |      ✓      |             | Multi-GPU and multi-TPU support |
| Distributed evaluation                        |      ✓      |      ✓      |             | Multi-GPU and multi-TPU support |
| Model and training checkpoint management      |      ✓      |      ✓      |             | |
| Custom metric evaluation                      |      ✓      |      ✓      |             | |
| Conditional evaluation of metrics             |      ✓      |      ✓      |             | |
| Gradient accumulation (micro-batches)         |      ✓      |      ✓      |             | |
| Tensorboard support                           |      ✓      |      ✓      |             | Might be refactored |
| Learning Rate scheduling                      |      ✓      |      ✓      |             | Might be refactored |
| Mixed Precision Training                      |      ✓      |      ❌     |             | |
| Using multiple optimizers                     |      ✓      |      ✓      |             | |



