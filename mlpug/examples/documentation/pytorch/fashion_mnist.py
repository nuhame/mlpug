import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import torchvision as tv

from mlpug.examples.documentation.shared_args import base_argument_set

# Import mlpug for Pytorch backend
import mlpug.pytorch as mlp

from basics.logging import get_logger


def load_data():
    transform = tv.transforms.ToTensor()

    training_data = tv.datasets.FashionMNIST('./mlpug-datasets-temp/', train=True, download=True, transform=transform)
    test_data = tv.datasets.FashionMNIST('./mlpug-datasets-temp/', train=False, download=True, transform=transform)

    return training_data, test_data


def create_callbacks_for(trainer, is_first_worker, validation_dataset, progress_log_period):
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
    ]

    # Only first worker needs to log progress
    if is_first_worker:
        callbacks += [
            mlp.callbacks.LogProgress(log_period=progress_log_period, set_names=['training', 'validation']),
        ]

    return callbacks


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
    # ########## TRAINING SETUP  ###########
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    progress_log_period = args.progress_log_period

    num_epochs = args.num_epochs

    seed = args.seed

    torch.random.manual_seed(seed)

    distributed = args.distributed

    if distributed:
        logger_name = f"[GPU {rank}] {os.path.basename(__file__)}"
    else:
        logger_name = os.path.basename(__file__)

    logger = get_logger(logger_name)

    is_first_worker = (distributed and rank == 0) or (not distributed)

    if is_first_worker:
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Progress log period: {progress_log_period}")
        logger.info(f"Num. training epochs: {num_epochs}")
        logger.info(f"Random seed: {seed}")
        logger.info(f"Distributed: {distributed}")

    # ########################################

    # ############## DEVICE SETUP ##############
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rank)

        if distributed:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
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
    if distributed and rank > 0:
        dist.barrier()

    training_data, test_data = load_data()

    if distributed and rank == 0:
        dist.barrier()

    training_sampler = None
    validation_sampler = None
    if distributed:
        training_sampler = torch.utils.data.distributed.DistributedSampler(training_data)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

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
    if distributed:
        train_model = DDP(train_model, device_ids=[rank])
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
    sys.stdout.write("\n\n\n\n")
    sys.stdout.flush()
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

    parser.parse_args()

    args = parser.parse_args()

    if args.distributed:
        world_size = torch.cuda.device_count()
        logger.info(f"Distributed Data Parallel mode : Using {world_size} GPUs")
        mp.spawn(worker_fn,
                 args=(args, world_size,),
                 nprocs=world_size,
                 join=True)
    else:
        worker_fn(0, args=args, world_size=1)
