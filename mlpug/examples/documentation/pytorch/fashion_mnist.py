import os
import sys

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import torchvision as tv

from mlpug.examples.documentation.shared_args import base_argument_set, describe_args

# Import mlpug for Pytorch backend
import mlpug.pytorch as mlp

from basics.logging import get_logger


def load_data():
    transform = tv.transforms.ToTensor()

    training_data = tv.datasets.FashionMNIST('./mlpug-datasets-temp/',
                                             train=True,
                                             download=True,
                                             transform=transform)
    test_data = tv.datasets.FashionMNIST('./mlpug-datasets-temp/',
                                         train=False,
                                         download=True,
                                         transform=transform)

    return training_data, test_data


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


def build_model(hidden_size=128):
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 10))


# MLPug needs a TrainModel that outputs the loss
class TrainModel(torch.nn.Module):
    def __init__(self, classifier, device):
        super(TrainModel, self).__init__()

        self.classifier = classifier
        self.device = device

        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, batch_data, evaluate_settings, inference_mode=None):
        images, true_labels = batch_data

        images = images.to(self.device)
        true_labels = true_labels.to(self.device)

        logits = self.classifier(images)
        return self.loss_func(logits, true_labels)


def worker_fn(rank, args, world_size):

    distributed = args.distributed
    is_primary = rank == 0

    mlp.logging.use_fancy_colors()
    
    # ########### EXPERIMENT SETUP ############
    torch.random.manual_seed(args.seed)  # For reproducibility

    if distributed:
        logger_name = f"[Device {rank}] {os.path.basename(__file__)}"
    else:
        logger_name = os.path.basename(__file__)

    logger = get_logger(logger_name)
    # ########################################

    # ############## DEVICE SETUP ##############
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rank)

        if distributed:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

            logger.info(f"Training using multiple GPUs: Using GPU {rank}/{world_size}")
        else:
            logger.info(f"Single device mode : Using GPU {rank} ")
    else:
        if distributed:
            logger.error(f"No GPUs available for data distributed training over multiple GPUs")
            return

        logger.info(f"Single device mode : Using CPU")

    device = torch.device("cuda" if use_cuda else "cpu")
    # ########################################

    # ########## SETUP BATCH DATASETS ##########
    if distributed and not is_primary:
        dist.barrier()

    training_data, test_data = load_data()

    if distributed and is_primary:
        dist.barrier()

    training_sampler = None
    validation_sampler = None
    if distributed:
        training_sampler = torch.utils.data.distributed.DistributedSampler(training_data)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

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
    if distributed:
        train_model = DDP(train_model, device_ids=[rank])
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


def test_model(model_checkpoint_filename, logger, device=None):
    if device is None:
        device = torch.device("cpu")

    logger.info(f'Loading model checkpoint ...')
    checkpoint = torch.load(model_checkpoint_filename, map_location=device)

    # Contains 'hidden_size'
    classifier = build_model(**checkpoint['hyper_parameters'])
    classifier.load_state_dict(checkpoint['model'])

    _, test_data = load_data()

    first_sample = next(iter(test_data))
    image = first_sample[0]
    real_label = first_sample[1]

    classifier.eval()
    with torch.no_grad():
        logits = classifier(image)
        probabilities = torch.softmax(logits, dim=-1)

        predicted_label = torch.argmax(probabilities)

        logger.info(f"real label = {real_label}, predicted label = {predicted_label}\n")


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
    if args.distributed:
        num_gpus_available = torch.cuda.device_count()
        world_size = args.num_devices if args.num_devices > 0 else num_gpus_available
        if world_size > num_gpus_available:
            logger.warn(f"Number of requested GPUs is lower than available GPUs, "
                        f"limiting training to {num_gpus_available} GPUS")
            world_size = num_gpus_available

        logger.info(f"Distributed Data Parallel mode : Using {world_size} GPUs")
        logger.info(f"Global batch size: {args.batch_size*world_size}")

        mp.spawn(worker_fn,
                 args=(args, world_size,),
                 nprocs=world_size,
                 join=True)
    else:
        logger.info(f"Single device mode.")
        logger.info(f"Global batch size: {args.batch_size}")

        worker_fn(0, args=args, world_size=1)

    # ######### USE THE TRAINED MODEL ##########
    sys.stdout.write("\n\n\n")
    sys.stdout.flush()

    logger.info("Using the trained classifier ...")

    model_checkpoint_filename = f'../trained-models/{args.experiment_name}-best-model-checkpoint.pt'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_model(model_checkpoint_filename, device=device, logger=logger)
