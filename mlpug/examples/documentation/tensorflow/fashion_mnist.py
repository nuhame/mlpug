# Partially based on https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb and
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb

import os
import sys

import pickle

import tensorflow as tf
import h5py
from tensorflow.python.keras.saving import hdf5_format

from basics.logging import get_logger

# Import mlpug for Tensorflow backend
import mlpug.tensorflow as mlp

from mlpug.examples.documentation.shared_args import base_argument_set, describe_args


def load_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Adding a dimension to the array -> new shape == (28, 28, 1)
    # We are doing this because the first layer in our model is a convolutional
    # layer and it requires a 4D input (batch_size, height, width, channels).
    # batch_size dimension will be added later on.
    train_images = train_images[..., None]
    test_images = test_images[..., None]

    # Scale
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def create_callbacks_for(trainer,
                         experiment_name,
                         model_hyper_parameters,
                         distribution_strategy,
                         validation_dataset,
                         progress_log_period):
    # At minimum you want to log the loss in the training progress
    # By default the batch loss and the moving average of the loss are calculated and logged
    loss_evaluator = mlp.evaluation.MetricEvaluator(trainer=trainer,
                                                    distribution_strategy=distribution_strategy)
    callbacks = [
        mlp.callbacks.TrainingMetricsLogger(metric_evaluator=loss_evaluator),
        # Calculate validation loss only once per epoch over the whole dataset
        mlp.callbacks.TestMetricsLogger(validation_dataset,
                                        'validation',
                                        metric_evaluator=loss_evaluator,
                                        batch_level=False),
        mlp.callbacks.CheckpointManager(base_checkpoint_filename=experiment_name,
                                        batch_level=False,  # monitor per epoch
                                        metric_to_monitor="validation.dataset.loss",
                                        metric_monitor_period=1,  # every epoch
                                        create_checkpoint_every=0,  # We are only interested in the best model,
                                        # not the latest model
                                        archive_last_model_checkpoint_every=0,  # no archiving
                                        backup_before_override=False,
                                        model_hyper_parameters=model_hyper_parameters),
        mlp.callbacks.LogProgress(log_period=progress_log_period, set_names=['training', 'validation']),
    ]

    return callbacks


def build_model(hidden_size=128):
    return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(10)
        ])


# MLPug needs a TrainModel that outputs the loss
class TrainModel(tf.keras.Model):
    def __init__(self, classifier, global_batch_size):
        super(TrainModel, self).__init__()

        self.classifier = classifier
        self.global_batch_size = global_batch_size

        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM)

    def call(self, batch_data, evaluate_settings, inference_mode=None):
        images, true_labels = batch_data

        logits = self.classifier(images, training=not inference_mode)
        loss = self.loss_func(true_labels, logits)/self.global_batch_size

        return loss


def train_model(args, logger):

    distributed = args.distributed

    # ########### EXPERIMENT SETUP ############
    tf.random.set_seed(args.seed)  # For reproducibility

    num_gpus_available = len(tf.config.list_physical_devices('GPU'))

    if distributed:
        if num_gpus_available < 1:
            logger.error(f"No GPUs available for data distributed training over multiple GPUs")
            return

        num_devices = args.num_devices if args.num_devices > 0 else num_gpus_available
        if num_devices > num_gpus_available:
            logger.warn(f"Number of requested GPUs is lower than available GPUs, "
                        f"limiting training to {num_gpus_available} GPUS")
            num_devices = num_gpus_available

        devices = [f"/gpu:{i}" for i in range(num_devices)]
        strategy = tf.distribute.MirroredStrategy(devices=devices)

        global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    else:
        device = "/device:GPU:0" if num_gpus_available > 0 else "/CPU:0"

        strategy = tf.distribute.OneDeviceStrategy(device=device)
        global_batch_size = args.batch_size

    # ########################################

    # ########## SETUP BATCH DATASETS ##########
    logger.info(f"Global batch size: {global_batch_size}")

    train_images, train_labels, test_images, test_labels = load_data()

    training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(global_batch_size)

    # Using the test set as a validation set, just for demonstration purposes
    validation_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(global_batch_size)

    num_batches_per_epoch = int(training_dataset.cardinality())
    if distributed:
        # Distribute training and validation set
        training_dataset = strategy.experimental_distribute_dataset(training_dataset)
        validation_dataset = strategy.experimental_distribute_dataset(validation_dataset)

    # ##########################################

    # Build within the strategy scope
    with strategy.scope():
        # ############ BUILD THE MODEL #############

        classifier = build_model(args.hidden_size)

        train_model = TrainModel(classifier, global_batch_size)
        # ##########################################

        # ############ SETUP OPTIMIZER #############
        optimizer = tf.keras.optimizers.Adam()
        # ##########################################

    # ############# SETUP TRAINING ##############
    print(f"Element spec : \n{training_dataset.element_spec}")
    trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer,
                                          model_components=classifier,
                                          distribution_strategy=strategy,
                                          # See issue https://github.com/tensorflow/tensorflow/issues/29911
                                          batch_data_signature=training_dataset.element_spec)

    model_hyper_parameters = {
        "hidden_size": args.hidden_size
    }

    callbacks = create_callbacks_for(trainer,
                                     args.experiment_name,
                                     model_hyper_parameters,
                                     strategy,
                                     validation_dataset,
                                     args.progress_log_period)

    manager = mlp.trainers.TrainingManager(trainer,
                                           training_dataset,
                                           num_epochs=args.num_epochs,
                                           num_batches_per_epoch=num_batches_per_epoch,
                                           callbacks=callbacks)

    trainer.set_training_model(train_model)
    # ##########################################

    # ################# START! #################
    manager.start_training()
    # ##########################################


def test_model(model_checkpoint_filename, logger, device=None):
    _, _, test_images, test_labels = load_data()

    if device is None:
        device = "/CPU:0"

    logger.info(f'Loading model checkpoint ...')
    with open(model_checkpoint_filename, 'rb') as f:
        checkpoint = pickle.load(f)

    with tf.device(device):
        # hyper_parameters contains 'hidden_size'
        classifier = build_model(**checkpoint['hyper_parameters'])
        with h5py.File(checkpoint['model'], 'r') as f:
            hdf5_format.load_weights_from_hdf5_group(f, classifier.layers)

        image = test_images[0]
        real_label = test_labels[0]

        logits = classifier(tf.expand_dims(image, 0))
        probabilities = tf.nn.softmax(logits)

        predicted_label = tf.math.argmax(probabilities, axis=-1)

        logger.info(f"real label = {real_label}, predicted label = {tf.squeeze(predicted_label)}\n")


if __name__ == '__main__':
    # ############# SETUP LOGGING #############
    mlp.logging.use_fancy_colors()

    logger_name = os.path.basename(__file__)
    logger = get_logger(logger_name)
    # ########################################

    # ############## PARSE ARGS ##############
    parser = base_argument_set()

    parser.parse_args()

    args = parser.parse_args()

    describe_args(args, logger)

    # ############## TRAIN MODEL ##############
    train_model(args, logger)

    # ######### USE THE TRAINED MODEL ##########
    sys.stdout.write("\n\n\n\n")
    sys.stdout.flush()

    logger.info("Using the classifier ...")

    num_gpus_available = len(tf.config.list_physical_devices('GPU'))
    device = "/device:GPU:0" if num_gpus_available > 0 else "/CPU:0"

    model_checkpoint_filename = f'../trained-models/{args.experiment_name}-best-model-checkpoint.m-ckp'
    test_model(model_checkpoint_filename, device=device, logger=logger)
