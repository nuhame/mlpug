# mlpug
Mlpug is a machine learning library agnostic framework for model training.

A lot of the functionality you need to train your machine learning model is 
independent of the machine learning library you're using, e.g. PyTorch and Tensorflow.
For instance, 

 * checkpoint management,
 * evaluation of validation set loss and other custom metrics, 
 * progress logging, 
 * progress visualization using Tensorboard, 
 * the use of gradient accumulation to train with large batch sizes using limited GPU memory, etc.. 

You need such functionality no matter what machine learning framework you are using.

Mlpug provides a single framework with a unified API for all such training functionality,
independent of the machine learning library you are using. This also implied that when you switch library
you can reuse your training code with no, or minimal, changes.

## Supported backends
Currently, mlpug supports the following deep learning/machine learning library 'backends':

 * PyTorch
 * Tensorflow (in development, some features not available yet)
   
Further, support for the following machine learning library backends are planned: 
 * Microsoft DeepSpeed
 * Jax

If you like your favorite machine learning library to be supported, please file an issue!

## Almost at version 0.1!
Mlpug is still in development. If you are having trouble using mlpug for your use case, or 
when you have found a bug, please file an issue.

## Contents
[Installing mlpug](#installing-mlpug) \
[Hello World](#hello-world) ([PT](#hello-world-with-pytorch) | [TF](#hello-world-with-tensorflow))

The following sections are documentation ToDo's: \
[The `logs` object](#the-logs-object) \
[Callbacks and the training life cycle](#callbacks-and-the-training-life-cycle) \
[Progress Logging](#progress-logging) \
[CheckpointManager](#checkpoint-manager) \
[Calculating custom metrics](#calculating-custom-metrics) \
[Tensorboard](#tensorboard) \
[Multi GPU training](#multi-gpu-training) \
[Mixed Precision Training](#mixed-precision-training) \
[Gradient Accumulation](#gradient-accumulation) \
[Metric computation having a large batch size](#metric-computation-having-a-large-batch-size) \
[CUDA Memory tools](#cuda-memory-tools)

## Installing mlpug
Please ensure that you are using Python3.7+.

Install as follows:
```
pip install mlpug
```

### Usage with PyTorch
When you want to use mlpug with PyTorch, you will need to install it:
```
pip install torch torchvision
```

### Usage with Tensorflow
When you want to use mlpug with Tensorflow, you will need to install it:
```
pip install tensorflow
```

## Hello World!
This is the Hello World of training with mlpug. You will see that the usage of mlpug with Pytorch and Tensorflow is 
very similar.

For details please see [tensorflow/hello_world.py](mlpug/examples/documentation/tensorflow/hello_world.py), 
[tensorflow/hello_world_not_eager.py](mlpug/examples/documentation/tensorflow/hello_world_not_eager.py) and [pytorch/hello_world.py](mlpug/examples/documentation/pytorch/hello_world.py)

I suggest you download and run these examples.

When reading through the explanation below it might be that you still have a lot of questions about the why and how of
training with mlpug, however I will expand the mlpug documentation soon, so you will get better insight.

### 'Hello World' with PyTorch
To use mlpug with Pytorch
```python
import mlpug.pytorch as mlp
```

Before we can start training we need an iterable dataset that can provide our training batches.

```python
training_dataset = torch.utils.data.DataLoader(training_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=3)

# Using the test set as a validation set, just for demonstration purposes
validation_dataset = torch.utils.data.DataLoader(test_data,
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

Mlpug needs a way to evaluate the loss of the model. One way to do that is to define a `TrainModel` that 
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

To now use mlpug to start training, we need to create a `Trainer` which will be used by a `TrainingManager`.
```python
trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer, model_components=classifier)
```

Mlpug uses a callback system allowing you to customize and extend the training functionality. 
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

### 'Hello World' with Tensorflow
Below we will focus only on the minor differences between using mlpug with PyTorch and Tensorflow.

To use mlpug with Tensorflow
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
