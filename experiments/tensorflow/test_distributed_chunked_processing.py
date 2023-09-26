from functools import partial

import tensorflow as tf

from mlpug.tensorflow.batch_chunking import ChunkableTupleBatchDim0, ChunkableBatchDataset
from mlpug.debugging import enable_pycharm_remote_debugging

from examples.fashion_mnist.tensorflow.train import load_data, build_model


class Processor:

    def __init__(self, model):
        self._model = model

    @tf.function()
    def process_chunk(self, chunk):
        tf.print(tf.shape(chunk[0]))
        tf.print(tf.shape(chunk[1]))

        return self._model(chunk[0])


def test_processing(batch, tv,  processor):
    print(tv)
    tf.print(f"[{batch[0].device}] GPU Input batch size = {tf.shape(batch[0])}")

    batch = ChunkableTupleBatchDim0.wrapper(batch)

    batch = ChunkableBatchDataset(
        batch,
        batch_chunk_size=16)

    results = []
    for chunk in batch:
        results += [processor.process_chunk(chunk)]

    return tf.concat(results, axis=0)


if __name__ == '__main__':
    # enable_pycharm_remote_debugging("192.168.178.15:54491")
    batch_size = 128

    num_gpus_available = len(tf.config.list_physical_devices('GPU'))

    train_images, train_labels, test_images, test_labels = load_data()

    devices = [f"/gpu:{i}" for i in range(num_gpus_available)]
    strategy = tf.distribute.MirroredStrategy(devices=devices)

    global_batch_size = batch_size * strategy.num_replicas_in_sync

    training_dataset = tf.data.Dataset \
        .from_tensor_slices((train_images, train_labels)) \
        .batch(global_batch_size)

    training_dataset = strategy.experimental_distribute_dataset(training_dataset)

    processor = Processor(build_model())
    process_batch = partial(test_processing, processor=processor)

    # For some reason there is an uncaught StopIteration exception at the end of the sequence
    # The final partial batch is handled perfectly, not sure what the issue is
    # Seems like a TF bug to me.
    test_val = 342
    for batch in training_dataset:
        results = strategy.run(
            process_batch,
            args=(batch, test_val)
        )

        results = strategy.unwrap(results)

        results = tf.concat(results, axis=0)

        tf.print(f"Final batch result = {tf.shape(results)}")