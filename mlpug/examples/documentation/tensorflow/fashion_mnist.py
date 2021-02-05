# Partially based on https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb and
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb

import os
import sys

import tensorflow as tf

from basics.logging import get_logger

# Import mlpug for Tensorflow backend
import mlpug.tensorflow as mlp

from mlpug.examples.documentation.shared_args import base_argument_set


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


def create_callbacks_for(trainer, distribution_strategy, validation_dist_dataset):
    # At minimum you want to log the loss in the training progress
    # By default the batch loss and the moving average of the loss are calculated and logged
    loss_evaluator = mlp.evaluation.MetricEvaluator(trainer=trainer,
                                                    distribution_strategy=distribution_strategy)
    callbacks = [
        mlp.callbacks.TrainingMetricsLogger(metric_evaluator=loss_evaluator),
        # Calculate validation loss only once per epoch over the whole dataset
        mlp.callbacks.TestMetricsLogger(validation_dist_dataset,
                                        'validation',
                                        metric_evaluator=loss_evaluator,
                                        batch_level=False),
        mlp.callbacks.LogProgress(log_period=progress_log_period, set_names=['training', 'validation']),
    ]

    return callbacks


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


if __name__ == '__main__':
    # ############# SETUP LOGGING #############
    mlp.logging.use_fancy_colors()

    logger_name = os.path.basename(__file__)
    logger = get_logger(logger_name)
    # ########################################

    # ############## PARSE ARGS ##############
    parser = base_argument_set()

    parser.add_argument(
        '--num-gpus',
        type=int, required=False, default=None,
        help='The number of GPUs to use for training, if None, all available will be used')

    parser.parse_args()

    args = parser.parse_args()

    batch_size_per_replica = args.batch_size
    learning_rate = args.learning_rate

    progress_log_period = args.progress_log_period

    num_epochs = args.num_epochs

    seed = args.seed

    num_gpus = args.num_gpus

    logger.info(f"Batch size per replica: {batch_size_per_replica}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Progress log period: {progress_log_period}")
    logger.info(f"Num. training epochs: {num_epochs}")
    logger.info(f"Random seed: {seed}")

    num_gpus_str = "all available" if num_gpus is None else num_gpus
    logger.info(f"Num. GPUs to use for training: {num_gpus_str}")

    tf.random.set_seed(seed)  # For reproducibility

    # ###### SETUP DISTRIBUTED TRAINING ######
    devices = None
    if num_gpus is not None:
        devices = [f"/gpu:{i}" for i in range(num_gpus)]

    strategy = tf.distribute.MirroredStrategy(devices=devices)
    # ########################################

    # ########## SETUP BATCH DATASETS ##########
    global_batch_size = batch_size_per_replica*strategy.num_replicas_in_sync
    logger.info(f"Global batch size: {global_batch_size}")

    train_images, train_labels, test_images, test_labels = load_data()

    training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(global_batch_size)

    # Using the test set as a validation set, just for demonstration purposes
    validation_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(global_batch_size)

    # Distribute training and validation set
    training_dist_dataset = strategy.experimental_distribute_dataset(training_dataset)
    validation_dist_dataset = strategy.experimental_distribute_dataset(validation_dataset)
    # ##########################################

    # Build within the strategy scope to distribute the training of the model
    with strategy.scope():
        # ############ BUILD THE MODEL #############

        classifier = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        # classifier = tf.keras.Sequential([
        #     tf.keras.layers.Flatten(input_shape=(28, 28)),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dense(10)
        # ])

        train_model = TrainModel(classifier, global_batch_size)
        # ##########################################

        # ############ SETUP OPTIMIZER #############
        optimizer = tf.keras.optimizers.Adam()
        # ##########################################

    # ############# SETUP TRAINING ##############
    print(f"Element spec : \n{training_dist_dataset.element_spec}")
    trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer,
                                          model_components=classifier,
                                          distribution_strategy=strategy,
                                          # See issue https://github.com/tensorflow/tensorflow/issues/29911
                                          batch_data_signature=training_dist_dataset.element_spec)

    callbacks = create_callbacks_for(trainer, strategy, validation_dist_dataset)

    manager = mlp.trainers.TrainingManager(trainer,
                                           training_dist_dataset,
                                           num_epochs=num_epochs,
                                           num_batches_per_epoch=int(training_dataset.cardinality()),
                                           callbacks=callbacks)

    trainer.set_training_model(train_model)
    # ##########################################

    # ################# START! #################
    manager.start_training()
    # ##########################################

    # ######### USE THE TRAINED MODEL ##########
    sys.stdout.write("\n\n\n\n")
    sys.stdout.flush()

    logger.info("Using the classifier ...")
    image = test_images[0]
    real_label = test_labels[0]

    logits = classifier(tf.expand_dims(image, 0))
    probabilities = tf.nn.softmax(logits)

    predicted_label = tf.math.argmax(probabilities, axis=-1)

    logger.info(f"real label = {real_label}, predicted label = {tf.squeeze(predicted_label)}\n")
