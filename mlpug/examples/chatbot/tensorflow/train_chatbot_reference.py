import os

import pickle

import tensorflow as tf
import tensorflow_datasets as tfds

from mlpug.examples.chatbot.shared import create_argument_parser
from mlpug.examples.chatbot.conversation_dataset import load_sentence_pair_data

from mlpug.examples.chatbot.tensorflow.original_transformer_tutorial.model_data_generation import \
    create_chatbot_tf_encode_func, \
    create_length_filter_func

from mlpug.examples.chatbot.tensorflow.original_transformer_tutorial.transformer import Transformer

from mlpug.examples.chatbot.tensorflow.original_transformer_tutorial.training import TrainModel, CustomSchedule

import mlpug.tensorflow as mlp


from basics.logging import get_logger

BUFFER_SIZE = 1000


def dataset_path_for(subset):
    return os.path.join(dataset_path, f"{subset}-{base_dataset_filename}")


def create_dataset_generator(pairs):

    def generator():
        for pair in pairs:
            yield tuple(pair)

    return generator


def prepare_dataset(raw_dataset, tf_encode, filter_sequence_length):
    dataset = tf.data.Dataset.from_generator(
        create_dataset_generator(raw_dataset),
        (tf.string, tf.string),
        (tf.TensorShape([]), tf.TensorShape([])))

    dataset = dataset.map(tf_encode)
    dataset = dataset.filter(filter_sequence_length)
    # cache the dataset to memory to get a speedup while reading from it.
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE).padded_batch(global_batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == "__main__":

    parser = create_argument_parser()

    parser.add_argument(
        '--max-sequence-length',
        type=int, required=False, default=60,
        help='Max. sequence length')

    parser.add_argument(
        '--num-attention-heads',
        type=int, required=False, default=12,
        help='Num. attention heads')

    parser.add_argument(
        '--feed-forward-layer-size',
        type=int, required=False, default=3072,
        help='Element-wise feed forward layer size')

    args = parser.parse_args()

    if args.remote_debug:
        import pydevd_pycharm
        pydevd_pycharm.settrace('192.168.178.85', port=57491, stdoutToServer=True, stderrToServer=True)

    mlp.logging.use_fancy_colors()

    logger_name = os.path.basename(__file__)
    logger = get_logger(logger_name)

    # TODO : seed
    # seed = args.seed
    # logger.info(f"Seed : {seed}")
    # np.random.seed(args.seed)

    use_mixed_precision = args.float16

    ##################################################
    #
    # [START] Setup
    #
    ##################################################

    # ############ Conversations dataset #############
    dataset_path = args.dataset_path
    base_dataset_filename = args.base_dataset_filename
    logger.info(f"dataset_path : {dataset_path}")
    logger.info(f"base_dataset_filename : {base_dataset_filename}")

    max_sequence_length = args.max_sequence_length
    ##################################################

    # ############ Model configuration ###############
    embedding_size = args.embedding_size
    logger.info(f"embedding_size : {embedding_size}")

    state_size = args.state_size
    logger.info(f"state_size : {state_size}")

    num_layers = args.num_layers
    logger.info(f"num_layers : {num_layers}")

    num_attention_heads = args.num_attention_heads
    logger.info(f"num_attention_heads : {num_attention_heads}")

    feed_forward_layer_size = args.feed_forward_layer_size

    dropout = args.dropout
    logger.info(f"dropout rate : {dropout}")
    ##################################################

    # ########### Training/optimization ##############
    experiment_name = args.experiment_name
    logger.info(f"experiment_name: {experiment_name}")

    num_gpus = args.num_gpus
    num_gpus_str = "all available" if num_gpus is None else num_gpus
    logger.info(f"Num. GPUs to use for training: {num_gpus_str}")

    batch_size_per_replica = args.batch_size
    logger.info(f"Batch size per replica: {batch_size_per_replica}")

    clip = args.gradient_clipping
    willClip = clip > 0.0
    logger.info(f"Gradient clipping : {clip} (Will clip? {willClip})")

    learning_rate = args.learning_rate
    logger.info(f"Base learning rate : {learning_rate}")

    decoder_learning_ratio = args.decoder_learning_rate_ratio
    logger.info(f"Decoder LR ratio : {learning_rate}")

    num_epochs = args.num_epochs
    logger.info(f"Number of training epochs : {num_epochs}")

    progress_logging_period = args.progress_logging_period
    logger.info(f"Progress logging period : {progress_logging_period}")

    # For Tensorboard
    metric_names = {
        'batch.loss': 'cross_entropy',
        'batch.duration': 'training_time',
        'window_average.duration': 'training_time',
        'dataset.duration': 'training_time',
        'batch_size': 'size'
    }

    # ########### Load datasets ##############
    logger.info('Setup data sets ...')

    logger.info('Loading training set ...')
    training_dataset, voc = load_sentence_pair_data(dataset_path_for('training'), logger)
    logger.info('Loading validation set ...')
    validation_dataset, _unused_ = load_sentence_pair_data(dataset_path_for('validation'), logger)

    # training_dataset = training_dataset[:40000]
    # validation_dataset = validation_dataset[:12000]

    logger.debug(f"Number of sentence pairs in training set: {len(training_dataset)}")
    logger.debug(f"Number of sentence pairs in validation set: {len(validation_dataset)}")

    all_training_sentences = []
    for pair in training_dataset:
        all_training_sentences += pair

    logger.debug(f"Building tokenizer vocabulary ...")
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        all_training_sentences, target_vocab_size=2**13)

    vocab_size = tokenizer.vocab_size + 2

    del all_training_sentences
    logger.debug(f"Building tokenizer vocabulary ... READY")

    tf_encode = create_chatbot_tf_encode_func(tokenizer)
    filter_sequence_length = create_length_filter_func(max_sequence_length)
    ############################################

    # ########### Build model and optimizers ##############
    devices = None
    if num_gpus is not None:
        devices = [f"/gpu:{i}" for i in range(num_gpus)]

    strategy = tf.distribute.MirroredStrategy(devices=devices)
    with strategy.scope():
        logger.info('Building model ...')

        transformer = Transformer(num_layers, state_size,
                                  num_attention_heads, feed_forward_layer_size,
                                  vocab_size, vocab_size,
                                  pe_input=vocab_size,
                                  pe_target=vocab_size,
                                  rate=dropout)

        train_model = TrainModel(transformer)

        # Initialize optimizers
        # TODO : learning rate provided by arguments not used
        logger.info('Building optimizers ...')
        learning_rate = CustomSchedule(state_size)

        # learning_rate = 5e-3
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # ####################################################

    # ############### Prepare datasets ###################
    global_batch_size = batch_size_per_replica*strategy.num_replicas_in_sync
    logger.info(f"Global batch size : {global_batch_size}")

    training_dataset = prepare_dataset(training_dataset, tf_encode, filter_sequence_length)
    validation_dataset = prepare_dataset(validation_dataset, tf_encode, filter_sequence_length)

    len_training_dataset = training_dataset.reduce(0, lambda x, _: x + 1).numpy()
    len_validation_dataset = validation_dataset.reduce(0, lambda x, _: x + 1).numpy()

    logger.info(f"Number of training batches : {len_training_dataset}")
    logger.info(f"Number of validation batches : {len_validation_dataset}")

    # Distribute training and validation set
    training_dist_dataset = strategy.experimental_distribute_dataset(training_dataset)
    validation_dist_dataset = strategy.experimental_distribute_dataset(validation_dataset)
    # ####################################################

    # ############## SETUP TRAINING ##################
    logger.info('Prepare training ...')

    trainer = mlp.trainers.DefaultTrainer(optimizer,
                                          transformer,
                                          # use_mixed_precision=use_mixed_precision,
                                          distribution_strategy=strategy,
                                          batch_data_signature=training_dist_dataset.element_spec)

    average_loss_evaluator = mlp.evaluation.MetricEvaluator(trainer=trainer,
                                                            distribution_strategy=strategy,
                                                            batch_metric_funcs={
                                                                "loss": mlp.evaluation.GatherMaskedLoss(strategy)
                                                            },
                                                            name="AverageLossEvaluator")

    callbacks = [mlp.callbacks.TrainingMetricsLogger(metric_evaluator=average_loss_evaluator),
                 mlp.callbacks.TestMetricsLogger(validation_dist_dataset,
                                                 'validation',
                                                 metric_evaluator=average_loss_evaluator,
                                                 batch_averaging_window=len_validation_dataset),
                 # mlp.callbacks.BatchSizeLogger(),
                 mlp.callbacks.CheckpointManager(metric_to_monitor='validation.window_average.loss',
                                                 base_checkpoint_filename=args.experiment_name,
                                                 archive_last_model_checkpoint_every=20000),
                 mlp.callbacks.LogProgress(log_period=progress_logging_period, set_names=["training", "validation"]),
                 mlp.callbacks.AutoTensorboard(experiment_name=experiment_name, dataset_name='training',
                                               metric_names=metric_names),
                 mlp.callbacks.AutoTensorboard(experiment_name=experiment_name, dataset_name='validation',
                                               metric_names=metric_names),
                 # Batch-level batch duration and batch size
                 mlp.callbacks.Tensorboard(['batch.duration', 'batch_size'],
                                           experiment_name=experiment_name,
                                           dataset_name='training_params',
                                           metric_names=metric_names,
                                           ignore_missing_metrics=True),
                 # Batch-level average batch duration
                 mlp.callbacks.Tensorboard(['window_average.duration'],
                                           experiment_name=experiment_name,
                                           dataset_name='training_params',
                                           metrics_are_averages=True,
                                           metric_names=metric_names,
                                           ignore_missing_metrics=True),
                 # Epoch-level epoch duration
                 mlp.callbacks.Tensorboard(['dataset.duration'],
                                           experiment_name=experiment_name,
                                           dataset_name='training_params',
                                           batch_level=False,
                                           metric_names=metric_names,
                                           ignore_missing_metrics=True)]

    manager = mlp.trainers.TrainingManager(trainer,
                                           training_dist_dataset,
                                           num_batches_per_epoch=len_training_dataset,
                                           num_epochs=num_epochs,
                                           callbacks=callbacks,
                                           experiment_data=args)

    tc_file = args.training_checkpoint
    if tc_file:
        logger.info(f"Loading training checkpoint : {tc_file}")

        with open(tc_file, 'rb') as f:
            checkpoint = pickle.load(f)

        manager.set_state(checkpoint)

        logger.info(f"Ready loading training checkpoint.")

        del checkpoint

    trainer.set_training_model(train_model)

    ##################################################

    logger.info('Start training ...')
    manager.start_training()







