import os
import sys

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from basics.logging import get_logger

# Import mlpug for Pytorch/XLA backend
import mlpug.pytorch.xla as mlp

from mlpug.examples.documentation.shared_args import base_argument_set, describe_args
from mlpug.examples.documentation.pytorch.fashion_mnist import \
    load_data, \
    build_model, \
    TrainModel, \
    test_model


def create_callbacks_for(trainer,
                         experiment_name,
                         model_hyper_parameters,
                         is_primary,
                         validation_dataset,
                         progress_log_period):
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
        mlp.callbacks.CheckpointManager(base_checkpoint_filename=experiment_name,
                                        batch_level=False,  # monitor per epoch
                                        metric_to_monitor="validation.dataset.loss",
                                        metric_monitor_period=1,  # every epoch
                                        create_checkpoint_every=0,  # We are only interested in the best model,
                                                                    # not the latest model
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


def worker_fn(rank, flags):
    args = flags['args']
    world_size = flags['world_size']

    distributed = args.distributed
    is_primary = rank == 0

    mlp.logging.use_fancy_colors()

    # ########## EXPERIMENT SETUP  ###########
    torch.random.manual_seed(args.seed)  # For reproducibility

    if distributed:
        logger_name = f"[Device {rank}] {os.path.basename(__file__)}"
    else:
        logger_name = os.path.basename(__file__)

    logger = get_logger(logger_name)
    # ########################################

    # ############## DEVICE SETUP ##############
    xla_available = len(xm.get_xla_supported_devices()) > 0
    if not xla_available:
        logger.error("No XLA devices available, unable to train")
        return

    if distributed:
        logger.info(f"Training using multiple XLA devices: Using XLA device {rank}/{world_size}")
    else:
        logger.info(f"Single XLA device mode : Using XLA device {rank} ")

    device = xm.xla_device()
    # ########################################

    # ########## SETUP BATCH DATASETS ##########
    if distributed and not is_primary:
        xm.rendezvous("loading_data")

    training_data, test_data = load_data()

    if distributed and is_primary:
        xm.rendezvous("loading_data")

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

    training_dataset = torch.utils.data.DataLoader(training_data,
                                                   batch_size=args.batch_size,
                                                   shuffle=(training_sampler is None),
                                                   sampler=training_sampler,
                                                   num_workers=3)

    # Using the test set as a validation set, just for demonstration purposes
    validation_dataset = torch.utils.data.DataLoader(test_data,
                                                     batch_size=args.batch_size,
                                                     shuffle=(validation_sampler is None),
                                                     sampler=validation_sampler,
                                                     num_workers=3)
    # ##########################################

    # ############ BUILD THE MODEL #############
    classifier = build_model(args.hidden_size)

    train_model = TrainModel(classifier, device)

    # Move model to assigned GPU (see torch.cuda.set_device(args.local_rank))
    classifier.to(device)
    # ############################################

    # ############ SETUP OPTIMIZER #############
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    # ##########################################

    # ############# SETUP TRAINING ##############
    trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer, model_components=classifier)

    model_hyper_parameters = {
        "hidden_size": args.hidden_size
    }

    callbacks = create_callbacks_for(trainer,
                                     args.experiment_name,
                                     model_hyper_parameters,
                                     is_primary,
                                     validation_dataset,
                                     args.progress_log_period)

    manager = mlp.trainers.TrainingManager(trainer,
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
    # ############# SETUP LOGGING #############
    mlp.logging.use_fancy_colors()
    logger = get_logger(os.path.basename(__file__))
    # ########################################

    # ############## PARSE ARGS ##############
    parser = base_argument_set()

    parser.parse_args()

    args = parser.parse_args()

    describe_args(args, logger)

    # ############## TRAIN MODEL ##############
    flags = {
        'args': args
    }
    if args.distributed:
        num_xla_devices_available = 8
        flags['world_size'] = args.num_devices if args.num_devices > 0 else num_xla_devices_available
        if flags['world_size'] > num_xla_devices_available:
            logger.warn(f"Number of requested XLA devices is lower than available XLA devices, "
                        f"limiting training to {num_xla_devices_available} XLA devices")
            world_size = num_xla_devices_available

        logger.info(f"Distributed Data Parallel mode : Using {flags['world_size']} XLA devices")
        logger.info(f"Global batch size: {args.batch_size * flags['world_size']}")

        xmp.spawn(worker_fn,
                  args=(flags,),
                  nprocs=flags['world_size'],
                  start_method='fork')
    else:
        flags['world_size'] = 1
        logger.info(f"Single device mode.")
        logger.info(f"Global batch size: {args.batch_size}")

        worker_fn(0, flags)

    # ######### USE THE TRAINED MODEL ##########
    sys.stdout.write("\n\n\n")
    sys.stdout.flush()

    logger.info("Using the trained classifier ...")

    model_checkpoint_filename = f'../trained-models/{args.experiment_name}-best-model-checkpoint.pt'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_model(model_checkpoint_filename, logger, device=device)
