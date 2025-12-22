import logging

import os
import sys

from basics.logging import get_logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist

from torch_xla import runtime as xr

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

# Import mlpug for Pytorch/XLA backend
import mlpug.pytorch.xla as mlp

from mlpug.debugging import enable_pycharm_remote_debugging

from examples.fashion_mnist.shared_args import create_arg_parser, describe_args
from examples.fashion_mnist.pytorch.train import (
    load_data,
    build_model,
    TrainModel,
    test_model
)

def create_callbacks_for(trainer,
                         experiment_name,
                         model_hyper_parameters,
                         is_primary,
                         validation_dataset,
                         eager_mode,
                         progress_log_period):
    # TODO: add DistributedSamplerManager callbacks to reshuffle the data per epoch

    # At minimum, you want to log the loss in the training progress
    # By default the batch loss and the moving average of the loss are calculated and logged
    loss_evaluator = mlp.evaluation.MetricEvaluator(
        # The trainer knows how to evaluate the model
        trainer=trainer,
        eager_mode=eager_mode
    )

    callbacks = [
        mlp.callbacks.TrainingMetricsLogger(metric_evaluator=loss_evaluator),
        # Calculate validation loss only once per epoch over the whole dataset
        mlp.callbacks.DatasetMetricsLogger(
            validation_dataset,
            'validation',
            metric_evaluator=loss_evaluator,
            batch_level=False),
        mlp.callbacks.CheckpointManager(
            base_checkpoint_filename=experiment_name,
            batch_level=False,  # monitor per epoch
            metric_to_monitor="validation.dataset.loss",
            metric_monitor_period=1,  # every epoch
            create_checkpoint_every=0,  # We are only interested in the best model, not the latest model
            archive_last_model_checkpoint_every=0,  # no archiving
            backup_before_override=False,
            model_hyper_parameters=model_hyper_parameters)
    ]

    # Only primary worker needs to log progress
    if is_primary:
        callbacks += [
            mlp.callbacks.LogProgress(log_period=progress_log_period, set_names=['training', 'validation']),
        ]

    return callbacks


def worker_fn(rank, args):
    world_size = xm.xrt_world_size()

    distributed = args.distributed
    is_primary = xm.is_master_ordinal()

    is_using_pjrt = xr.using_pjrt()

    mlp.logging.use_fancy_colors(log_level=logging.WARNING)

    if is_primary and args.remote_debug_ip:
        enable_pycharm_remote_debugging(args.remote_debug_ip)

    # ########## EXPERIMENT SETUP  ###########
    torch.random.manual_seed(args.seed)  # For reproducibility

    if distributed:
        logger_name = f"[Device {rank}] {os.path.basename(__file__)}"
    else:
        logger_name = os.path.basename(__file__)

    logger = get_logger(logger_name)

    if is_primary:
        logger.info(f"Distributed Data Parallel mode : Using {world_size} XLA devices (Using PJRT={is_using_pjrt})")
        logger.info(f"Global batch size: {args.batch_size * world_size}")
    # ########################################

    # ############## DEVICE SETUP ##############
    xla_available = len(xm.get_xla_supported_devices()) > 0
    if not xla_available:
        raise Exception("No XLA devices available, unable to train")

    if args.force_on_cpu:
        raise ValueError("Force on CPU selected: Run the examples/fashion_mnist/pytorch/train.py example instead")

    if distributed:
        logger.info(f"Training using multiple XLA devices: Using XLA device {rank}/{world_size}")
        if is_using_pjrt:
            dist.init_process_group('xla', init_method='xla://')
        else:
            dist.init_process_group('xla', world_size=world_size, rank=rank)

    else:
        logger.info(f"Single XLA device mode : Using XLA device {rank} ")

    device = xm.xla_device()
    # ########################################

    # ########## SETUP BATCH DATASETS ##########
    if distributed and not is_primary:
        xm.rendezvous("data_loaded")

    training_data, test_data = load_data()

    if distributed and is_primary:
        xm.rendezvous("data_loaded")

    training_sampler = None
    validation_sampler = None
    if distributed:
        training_sampler = torch.utils.data.distributed.DistributedSampler(
            training_data,
            num_replicas=world_size,
            rank=rank)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            test_data,
            num_replicas=world_size,
            rank=rank)

    # DataLoader yields micro-batches (or full batches if no gradient accumulation)
    dataloader_batch_size = args.micro_batch_size if args.micro_batch_size else args.batch_size

    training_dataset = torch.utils.data.DataLoader(
        training_data,
        batch_size=dataloader_batch_size,
        shuffle=(training_sampler is None),
        sampler=training_sampler,
        num_workers=0)
    training_dataset = pl.MpDeviceLoader(training_dataset, device)

    # Using the test set as a validation set, just for demonstration purposes
    validation_dataset = torch.utils.data.DataLoader(
        test_data,
        batch_size=dataloader_batch_size,
        shuffle=(validation_sampler is None),
        sampler=validation_sampler,
        num_workers=0)
    validation_dataset = pl.MpDeviceLoader(validation_dataset, device)
    # ##########################################

    # ############ BUILD THE MODEL #############
    classifier = build_model(args.hidden_size)

    train_model = TrainModel(classifier, device)
    train_model.to(device)

    # Initialization is nondeterministic with multiple threads in PjRt.
    # Synchronize model parameters across replicas manually.
    if xr.using_pjrt():
        logger.info("Broadcast master parameters ...")
        xm.broadcast_master_param(train_model)

    if distributed:
        logger.info("Wrap train_model in DDP ...")
        train_model = DDP(train_model, gradient_as_bucket_view=True)
    # ############################################

    # ############ SETUP OPTIMIZER #############
    # Scale learning rate to num devices
    # lr = args.learning_rate * world_size
    lr = args.learning_rate
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    # ##########################################

    # ############# SETUP TRAINING ##############
    trainer = mlp.trainers.DefaultTrainer(
        optimizers=optimizer,
        model_components=classifier,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
    )

    model_hyper_parameters = {
        "hidden_size": args.hidden_size
    }

    callbacks = create_callbacks_for(
        trainer,
        args.experiment_name,
        model_hyper_parameters,
        is_primary,
        validation_dataset,
        args.progress_log_period)

    manager = mlp.trainers.TrainingManager(
        trainer,
        training_dataset,
        num_epochs=args.num_epochs,
        callbacks=callbacks,
        experiment_data={
            "args": args
        })

    trainer.set_training_model(train_model)
    # ##########################################

    # ################# START! #################
    manager.start_training()
    # ##########################################

    logger.info("DONE.")


if __name__ == '__main__':
    os.environ['PJRT_DEVICE'] = 'TPU'

    # ############# SETUP LOGGING #############
    mlp.logging.use_fancy_colors()
    logger = get_logger(os.path.basename(__file__))
    # ########################################

    # ############## PARSE ARGS ##############
    parser = create_arg_parser()
    parser.parse_args()

    args = parser.parse_args()

    describe_args(args, logger)

    # ############## TRAIN MODEL ##############
    if args.distributed:
        # num_devices automatically derived when None is given
        num_devices = args.num_devices if args.num_devices is not None and args.num_devices > 0 else None

        xmp.spawn(worker_fn, args=(args,), nprocs=num_devices)
    else:
        logger.info(f"Single device mode.")
        logger.info(f"Global batch size: {args.batch_size}")

        worker_fn(0, args)

    # ######### USE THE TRAINED MODEL ##########
    sys.stdout.write("\n\n\n")
    sys.stdout.flush()

    logger.info("Using the trained classifier ...")

    model_checkpoint_filename = f'../trained-models/{args.experiment_name}-best-model-checkpoint.pt'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_model(model_checkpoint_filename, logger, device=device)
