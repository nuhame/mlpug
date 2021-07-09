import os
import sys

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from basics.logging import get_logger

# Import mlpug for Pytorch/XLA backend
import mlpug.pytorch.xla as mlp

from mlpug.examples.documentation.shared_args import base_argument_set
from mlpug.examples.documentation.pytorch.fashion_mnist import \
    load_data, \
    create_callbacks_for, \
    TrainModel


def worker_fn(worker_index, flags):
    args = flags['args']

    # ########## TRAINING SETUP  ###########
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    progress_log_period = args.progress_log_period

    num_epochs = args.num_epochs

    seed = args.seed

    torch.random.manual_seed(seed)

    distributed = args.distributed

    if distributed:
        logger_name = f"[Worker {worker_index}] {os.path.basename(__file__)}"
    else:
        logger_name = os.path.basename(__file__)

    logger = get_logger(logger_name)

    is_first_worker = not distributed or xm.is_master_ordinal()

    if is_first_worker:
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Progress log period: {progress_log_period}")
        logger.info(f"Num. training epochs: {num_epochs}")
        logger.info(f"Random seed: {seed}")
        logger.info(f"Distributed: {distributed}")

    # ########################################

    # ############## DEVICE SETUP ##############
    xla_available = len(xm.get_xla_supported_devices()) > 0
    if not xla_available:
        logger.error("No XLA devices available, unable to train")
        return

    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    if distributed:
        logger.info(f"Training over multiple XLA devices: Using XLA device {rank}/{world_size}")
    else:
        logger.info(f"Single XLA device mode : Using XLA device {rank} ")

    device = xm.xla_device()
    # ########################################

    # ########## SETUP BATCH DATASETS ##########
    if distributed and not is_first_worker:
        xm.rendezvous("loading_data")

    training_data, test_data = load_data()

    if distributed and is_first_worker:
        xm.rendezvous("loading_data")

    training_sampler = None
    validation_sampler = None
    if distributed:
        training_sampler = torch.utils.data.distributed.DistributedSampler(
            training_data,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal())
        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            test_data,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal())

    training_dataset = torch.utils.data.DataLoader(training_data,
                                                   batch_size=batch_size,
                                                   shuffle=(training_sampler is None),
                                                   sampler=training_sampler,
                                                   num_workers=3)

    # Using the test set as a validation set, just for demonstration purposes
    validation_dataset = torch.utils.data.DataLoader(test_data,
                                                     batch_size=batch_size,
                                                     shuffle=(validation_sampler is None),
                                                     sampler=validation_sampler,
                                                     num_workers=3)
    # ##########################################

    # ############ BUILD THE MODEL #############
    classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10))

    train_model = TrainModel(classifier, device)

    # Move model to assigned GPU (see torch.cuda.set_device(args.local_rank))
    classifier.to(device)
    # ############################################

    # ############ SETUP OPTIMIZER #############
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    # ##########################################

    # ############# SETUP TRAINING ##############
    trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer, model_components=classifier)

    callbacks = create_callbacks_for(trainer,
                                     is_first_worker,
                                     validation_dataset,
                                     progress_log_period)

    manager = mlp.trainers.TrainingManager(trainer,
                                           training_dataset,
                                           num_epochs=num_epochs,
                                           callbacks=callbacks)

    trainer.set_training_model(train_model)
    # ##########################################

    # ################# START! #################
    manager.start_training()
    # ##########################################

    # ######### USE THE TRAINED MODEL ##########
    if is_first_worker:
        logger.info("Using the classifier ...")
        first_sample = next(iter(test_data))
        image = first_sample[0]
        real_label = first_sample[1]

        # Transfer the image to the assigned device
        image = image.to(device)

        logits = classifier(image)
        probabilities = torch.softmax(logits, dim=-1)

        predicted_label = torch.argmax(probabilities)

        logger.info(f"real label = {real_label}, predicted label = {predicted_label}\n")

    xm.rendezvous("worker_ready")


if __name__ == '__main__':
    # ############# SETUP LOGGING #############
    mlp.logging.use_fancy_colors()
    logger = get_logger(os.path.basename(__file__))
    # ########################################

    # ############## PARSE ARGS ##############
    parser = base_argument_set()

    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Set to distribute training over multiple GPUs')
    parser.add_argument(
        '--num_xla_devices',
        type=int, required=False, default=8,
        help='Number of XLA devices to use in distributed mode, '
             'usually this is the number of TPU cores.')

    parser.parse_args()

    args = parser.parse_args()

    flags = {
        'args': args
    }
    if args.distributed:
        world_size = args.num_xla_devices
        logger.info(f"Distributed Data Parallel mode : Using {world_size} XLA devices")
        xmp.spawn(worker_fn,
                  args=(flags,),
                  nprocs=world_size,
                  start_method='fork')
    else:
        worker_fn(0, flags)
