import os
import random

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as PT_DDP

import evaluation
from examples.legacy.chatbot.shared import create_argument_parser

from examples.legacy.chatbot.pytorch.original_chatbot_tutorial.model_data_generation import \
    create_sentence_pairs_collate_fn
from examples.legacy.chatbot.conversation_dataset import IndexedSentencePairsDataset, load_sentence_pair_data

from examples.legacy.chatbot.pytorch.original_chatbot_tutorial.seq2seq import EncoderRNN, LuongAttnDecoderRNN
from examples.legacy.chatbot.pytorch.original_chatbot_tutorial.training import Seq2SeqTrainModel
from examples.legacy.chatbot.pytorch.original_chatbot_tutorial.loss import masked_loss

import mlpug.pytorch as mlp

from mlpug.utils import get_value_at

from basics.logging import get_logger


def transfer_to_gpu(opt):
    # If you have cuda, configure cuda to call
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


def dataset_path_for(subset):
    return os.path.join(dataset_path, f"{subset}-{base_dataset_filename}")


class TeacherForcingController(mlp.callbacks.Callback):

    def on_epoch_start(self, logs):
        if 'training_settings' not in logs:
            logs['training_settings'] = {}
            logs['evaluate_settings'] = {}

        return True

    def on_batch_training_start(self, training_batch, logs):
        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        logs['training_settings']['use_teacher_forcing'] = use_teacher_forcing
        logs['evaluate_settings']['use_teacher_forcing'] = use_teacher_forcing

        return True


class ChatbotTrainer(mlp.trainers.DefaultTrainer):

    def _evaluate_loss(self, batch_data, evaluate_settings=None, inference_mode=None):

        use_teacher_forcing = get_value_at('use_teacher_forcing', evaluate_settings)
        if use_teacher_forcing is None:
            use_teacher_forcing = True

        padded_input_batch, input_lengths, output_batch, output_mask, max_output_len = batch_data

        padded_input_batch = padded_input_batch.to(device)
        input_lengths = input_lengths.to(device)
        output_batch = output_batch.to(device)
        output_mask = output_mask.to(device)

        batch_size = padded_input_batch.size(1)

        init_decoder_input = torch.tensor([[SOS_token for _ in range(batch_size)]], dtype=torch.long, device=device)

        results = self.training_model(padded_input_batch, input_lengths, init_decoder_input, max_output_len,
                                      output_batch, use_teacher_forcing)

        per_sample_loss = results["loss"]

        # loss, loss_sum, num_samples
        return masked_loss(per_sample_loss, output_mask)

    def _prepare_update_model_parameters(self):
        if not willClip:
            return

        # Clip gradients: gradients are modified in place
        if not use_mixed_precision:
            _ = nn.utils.clip_grad_norm_(self.training_model.parameters(), clip)
        else:
            for optimizer in self.optimizers.values():
                # Unscales the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                nn.utils.clip_grad_norm_(self.training_model.parameters(), clip)


if __name__ == "__main__":

    parser = create_argument_parser()

    parser.add_argument(
        '--attention-model',
        type=str, required=False, default='general',
        help='Attention model type: "dot", "general" or "concat"')

    parser.add_argument(
        '--local_rank',
        type=int, required=False, default=None,
        help='Set automatically when running this script in distributed parallel mode')

    parser.add_argument(
        '--teacher-forcing-ratio',
        type=float, required=False, default=1.0,
        help='How often teacher forcing should be applied, compared to not applying it')

    args = parser.parse_args()

    num_gpus = args.num_gpus
    use_mixed_precision = args.float16
    distributed = args.local_rank is not None
    single_process_data_parallel = (num_gpus is not None) and (num_gpus > 1)
    use_cuda = torch.cuda.is_available()

    is_first_worker = (distributed and args.local_rank == 0) or (not distributed)

    logger_name = os.path.basename(__file__)
    if distributed:
        logger_name += f" (WORKER {args.local_rank})"

    logger = get_logger(logger_name)
    mlp.logging.use_fancy_colors()

    if distributed and single_process_data_parallel:
        logger.warn("The num_gpus argument will be ignored in Distributed Data Parallel mode. "
                    "Number of GPUs will be defined through --nproc_per_node argument of the "
                    "torch.distributed.launch script")
        num_gpus = None
        single_process_data_parallel = False

    if distributed and not use_cuda:
        logger.error("Distributed Data Parallel mode can only be used when NVidia GPUs are available")
        exit(-1)

    if use_mixed_precision and single_process_data_parallel:
        logger.error("Mixed precision training can't be combined with single process data parallel, "
                     "use distributed data parallel mode instead")
        exit(-1)

    if args.remote_debug and is_first_worker:
        import pydevd_pycharm
        pydevd_pycharm.settrace('192.168.178.85', port=57491, stdoutToServer=True, stderrToServer=True)

    seed = args.seed
    logger.info(f"Seed : {seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info(f"Using CUDA? {use_cuda}")
    logger.info(f"In Distributed Data Parallel mode ? {distributed}")
    logger.info(f"Using single process Data Parallel mode "
                f"(multiple GPUs in one process)? {single_process_data_parallel}\n\n")
    logger.info(f"Use mixed precision training? {use_mixed_precision}")

    if distributed:
        logger.info(f"Distributed Data Parallel mode : Using GPU {args.local_rank} ")
        torch.cuda.set_device(args.local_rank)

        dist.init_process_group(backend='nccl', init_method='env://')

    device = torch.device("cuda" if use_cuda else "cpu")

    ##################################################
    #
    # [START] Setup
    #
    ##################################################

    # ############ Conversations dataset #############
    # Default word tokens

    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    dataset_path = args.dataset_path
    base_dataset_filename = args.base_dataset_filename
    logger.info(f"dataset_path : {dataset_path}")
    logger.info(f"base_dataset_filename : {base_dataset_filename}")

    ##################################################

    # ############ Model configuration ###############
    embedding_size = args.embedding_size
    logger.info(f"embedding_size : {embedding_size}")

    state_size = args.state_size
    logger.info(f"state_size : {state_size}")

    num_layers = args.num_layers
    logger.info(f"num_layers : {num_layers}")

    attn_model = args.attention_model
    logger.info(f"Attention model : {attn_model}")

    dropout = args.dropout
    logger.info(f"dropout rate : {dropout}")
    ##################################################

    # ########### Training/optimization ##############
    experiment_name = args.experiment_name
    logger.info(f"experiment_name: {experiment_name}")

    batch_size = args.batch_size
    if single_process_data_parallel:
        batch_size = num_gpus * batch_size
        logger.info(f"batch_size (all GPUs combined): {batch_size}")
    else:
        logger.info(f"batch_size (per process): {batch_size}")

    clip = args.gradient_clipping
    willClip = clip > 0.0
    logger.info(f"Gradient clipping : {clip} (Will clip? {willClip})")

    teacher_forcing_ratio = args.teacher_forcing_ratio
    logger.info(f"Teacher forcing ratio : {teacher_forcing_ratio}")

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
        'batch.loss': 'cross entropy',
        'batch.duration': 'training time',
        'sliding_window.duration': 'training time',
        'dataset.duration': 'training_time',
        'batch_size': 'size'
    }

    ##################################################

    ##################################################
    #
    # [END] Setup
    #
    ##################################################

    # ############## SETUP DATASETS ###################
    logger.info('Setup data sets ...')

    logger.info('Loading training set ...')
    training_dataset, voc = load_sentence_pair_data(dataset_path_for('training'), logger)
    logger.info('Loading validation set ...')
    validation_dataset, _unused_ = load_sentence_pair_data(dataset_path_for('validation'), logger)

    logger.debug(f"Number of sentence pairs in training set: {len(training_dataset)}")
    logger.debug(f"Number of sentence pairs in validation set: {len(validation_dataset)}")

    training_dataset = IndexedSentencePairsDataset(training_dataset, voc, EOS_token)
    validation_dataset = IndexedSentencePairsDataset(validation_dataset, voc, EOS_token)

    # collate_sentence_pairs = create_sentence_pairs_collate_fn(PAD_token, fixed_sequence_length=40)
    collate_sentence_pairs = create_sentence_pairs_collate_fn(PAD_token)

    training_sampler = None
    validation_sampler = None
    if distributed:
        training_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)

    train_dataset_loader = torch.utils.data.DataLoader(training_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=(training_sampler is None),
                                                       num_workers=3,
                                                       collate_fn=collate_sentence_pairs,
                                                       pin_memory=True,
                                                       sampler=training_sampler)

    validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=(validation_sampler is None),
                                                            num_workers=3,
                                                            collate_fn=collate_sentence_pairs,
                                                            pin_memory=True,
                                                            sampler=validation_sampler)

    ##################################################

    # ############### BUILD MODEL ####################
    logger.info('Building model ...')

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, embedding_size)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(state_size, embedding, num_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, state_size, voc.num_words, num_layers, dropout)

    training_model = Seq2SeqTrainModel(encoder, decoder)

    logger.info(f"Transfer training model to device {device} ...")
    # Use appropriate device
    training_model = training_model.to(device)

    ##################################################

    # ############## SETUP TRAINING ##################

    # Initialize optimizers
    logger.info('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # TODO : Is this required for DDP and mixed procession training?
    if use_cuda:
        logger.info(f"Transfer optimizers model to {device} ...")
        transfer_to_gpu(encoder_optimizer)
        transfer_to_gpu(decoder_optimizer)

    if single_process_data_parallel:
        logger.info(f"We are using {num_gpus} GPUs in a single process, making training model data parallel ...")
        # TODO : actually use the right number of GPUs
        training_model = nn.DataParallel(training_model, dim=1)
    elif distributed:
        training_model = PT_DDP(training_model,
                                device_ids=[args.local_rank],
                                output_device=args.local_rank,
                                dim=1)

    logger.info('Prepare training ...')

    trainer = ChatbotTrainer([encoder_optimizer, decoder_optimizer],
                             {
                                 'embedding': embedding,
                                 'encoder': encoder,
                                 'decoder': decoder
                             },
                             use_mixed_precision=use_mixed_precision)

    average_loss_evaluator = mlp.evaluation.MetricEvaluator(trainer=trainer,
                                                            batch_metric_funcs={
                                                                "loss": evaluation.GatherMaskedLoss()
                                                            },
                                                            name="AverageLossEvaluator")

    callbacks = []
    if distributed:
        callbacks += [mlp.callbacks.DistributedSamplerManager(training_sampler, name="SamplerManager[training]"),
                      mlp.callbacks.DistributedSamplerManager(validation_sampler, name="SamplerManager[validation]")]

    callbacks += [TeacherForcingController(name="TeacherForcingController"),
                  mlp.callbacks.TrainingMetricsLogger(metric_evaluator=average_loss_evaluator),
                  mlp.callbacks.DatasetMetricsLogger(validation_dataset_loader,
                                                     dataset_name='validation',
                                                     metric_evaluator=average_loss_evaluator)]

    if is_first_worker:
        callbacks += [
            mlp.callbacks.BatchSizeLogger(),
            mlp.callbacks.CheckpointManager(metric_to_monitor='validation.sliding_window.loss',
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
            mlp.callbacks.Tensorboard(['sliding_window.duration'],
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
                                           train_dataset_loader,
                                           num_epochs=num_epochs,
                                           callbacks=callbacks,
                                           experiment_data=args)

    tc_file = args.training_checkpoint
    if tc_file:
        # Can we make this whole procedure a part of MLPug?
        logger.info(f"Loading training checkpoint : {tc_file}")
        map_location = None
        if distributed:
            map_location = f'cuda:{args.local_rank}'

        checkpoint = torch.load(tc_file, map_location=map_location)

        manager.set_state(checkpoint)

        if distributed:
            dist.barrier()
        logger.info(f"Ready loading training checkpoint.")

        # Required, else loading a checkpoint can lead to OOMs
        del checkpoint
        if use_cuda:
            torch.cuda.empty_cache()

    trainer.set_training_model(training_model)

    ##################################################

    logger.info('Start training ...')
    manager.start_training()
