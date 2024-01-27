import tensorflow as tf

import numpy as np


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


def build_model(hidden_size=128):
    return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(10)
        ])


class TrainModel(tf.keras.Model):
    def __init__(self, classifier):
        super(TrainModel, self).__init__()

        self.classifier = classifier

        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM)

    def call(self, batch_data, inference_mode=None):
        images, labels = batch_data

        batch_size = tf.shape(labels)[0]

        logits = self.classifier(images, training=not inference_mode)
        loss = self.loss_func(labels, logits)/tf.cast(batch_size, tf.float32)

        return {
            "loss": loss,
            "num_samples": batch_size,
            "auxiliary_results": {
                # To compute classification quality
                "logits": logits
            }
        }


def gather_loss(loss, num_samples):
    # Convert back from average to loss sum
    return tf.cast(num_samples, loss.dtype) * loss, num_samples


def gather_loss_distributed(strategy, loss, num_samples):
    loss_sum = strategy.reduce(
        tf.distribute.ReduceOp.SUM,
        loss,
        axis=None)

    tot_num_samples = strategy.reduce(
        tf.distribute.ReduceOp.SUM,
        num_samples,
        axis=None)

    return loss_sum.numpy(), tot_num_samples.numpy()


def average_loss(loss_sum, tot_num_samples):
    return loss_sum/tot_num_samples


def gather_classification_data(batch, auxiliary_results):
    labels = batch[1]

    logits = auxiliary_results["logits"]

    prediction_probability = tf.nn.softmax(logits, axis=1)
    predictions = tf.math.argmax(prediction_probability, axis=1)

    # Need to cast to int32 because can't gather int8 in next step: gather_classification_data_distributed
    return tf.cast(labels, tf.int32), predictions


def gather_classification_data_distributed(strategy, labels, predictions):
    tf.print("Gather labels from all devices ...")
    labels = strategy.gather(labels, axis=0)
    tf.print("Gather predictions from all devices ...")
    predictions = strategy.gather(predictions, axis=0)

    return labels.numpy(), predictions.numpy()


def calc_classification_quality(labels, predictions ):
    num_samples = len(labels)
    accuracy = np.sum(labels == predictions)/num_samples

    return accuracy


if __name__ == '__main__':
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('192.168.178.15', port=54491, stdoutToServer=True, stderrToServer=True)

    # tf.config.run_functions_eagerly(True)

    batch_size = 32

    num_devices = len(tf.config.list_physical_devices('GPU'))
    devices = [f"GPU:{i}" for i in range(num_devices)]
    strategy = tf.distribute.MirroredStrategy(
        devices=devices,
        # !!! CODE ONLY WORKS (NO DEADLOCK) WITH ReductionToOneDevice OPS !!!
        cross_device_ops=None  # tf.distribute.ReductionToOneDevice()
    )

    global_batch_size = batch_size * strategy.num_replicas_in_sync

    train_images, train_labels, _, _ = load_data()

    training_dataset = tf.data.Dataset\
        .from_tensor_slices((train_images, train_labels))\
        .batch(global_batch_size)

    # Distribute training
    training_dataset = strategy.experimental_distribute_dataset(training_dataset)

    with strategy.scope():
        classifier = build_model()
        train_model = TrainModel(classifier)

    @tf.function
    def run_train_model(batch):
        return strategy.run(
            train_model,
            args=(batch,),
            kwargs={"inference_mode": False}
        )

    @tf.function
    def run_gather_loss(loss, num_samples):
        return strategy.run(
            gather_loss,
            args=(loss, num_samples),
        )

    @tf.function
    def run_gather_classification_data(batch, auxiliary_results):
        return strategy.run(
            gather_classification_data,
            args=(batch, auxiliary_results),
        )

    # Get a batch
    batch = next(iter(training_dataset))

    results = run_train_model(batch)

    # Gather metric data per device
    loss_sum, tot_num_samples = run_gather_loss(results['loss'], results["num_samples"])
    labels, predictions = run_gather_classification_data(batch, results["auxiliary_results"])

    print(f"loss_sum.values[-1].device           : {loss_sum.values[-1].device}")
    print(f"loss_sum.values[-1].backing_device   : {loss_sum.values[-1].backing_device}")

    print(f"predictions.values[-1].device        : {predictions.values[-1].device}")
    print(f"predictions.values[-1].backing_device: {predictions.values[-1].backing_device}")

    # Combine metric data from all devices
    loss_sum, tot_num_samples = gather_loss_distributed(strategy, loss_sum, tot_num_samples)
    # !!! HANGS HERE WHEN DEFAULT cross_device_ops ARE USED !!!
    labels, predictions = gather_classification_data_distributed(strategy, labels, predictions)

    # Calculate final metrics
    loss = average_loss(loss_sum, tot_num_samples)
    accuracy = calc_classification_quality(labels, predictions)

    print(f"loss    : {loss}")
    print(f"accuracy: {accuracy}")
