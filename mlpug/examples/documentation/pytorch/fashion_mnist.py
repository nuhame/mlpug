import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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


def create_callbacks_for(trainer, is_first_worker):
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
    def __init__(self, classifier):
        super(TrainModel, self).__init__()

        self.classifier = classifier
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, batch_data, evaluate_settings, inference_mode=None):
        images, true_labels = batch_data

        logits = self.classifier(images)
        return self.loss_func(logits, true_labels)


if __name__ == '__main__':
    # ############# SETUP LOGGING #############
    mlp.logging.use_fancy_colors()

    logger_name = os.path.basename(__file__)
    logger = get_logger(logger_name)
    # ########################################

    # ############## PARSE ARGS ##############
    parser = base_argument_set()

    parser.add_argument(
        '--local_rank',
        type=int, required=False, default=None,
        help='Set automatically when running this script in distributed parallel mode')

    parser.parse_args()

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate

    progress_log_period = args.progress_log_period

    num_epochs = args.num_epochs

    seed = args.seed

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Progress log period: {progress_log_period}")
    logger.info(f"Num. training epochs: {num_epochs}")
    logger.info(f"Random seed: {seed}")

    torch.random.manual_seed(seed)

    distributed = args.local_rank is not None
    is_first_worker = (distributed and args.local_rank == 0) or (not distributed)
    # ########################################

    # ###### SETUP DISTRIBUTED TRAINING ######
    if distributed:
        logger.info(f"Distributed Data Parallel mode : Using GPU {args.local_rank} ")
        torch.cuda.set_device(args.local_rank)

        dist.init_process_group(backend='nccl', init_method='env://')
    # ########################################

    # ########## SETUP BATCH DATASETS ##########
    training_data, test_data = load_data()

    training_sampler = None
    validation_sampler = None
    if distributed:
        training_sampler = torch.utils.data.distributed.DistributedSampler(training_data)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    training_dataset = torch.utils.data.DataLoader(training_data,
                                                   batch_size=batch_size,
                                                   shuffle=(training_sampler is None),
                                                   num_workers=3)

    # Using the test set as a validation set, just for demonstration purposes
    validation_dataset = torch.utils.data.DataLoader(test_data,
                                                     batch_size=batch_size,
                                                     shuffle=(validation_sampler is None),
                                                     num_workers=3)
    # ##########################################

    # ############ BUILD THE MODEL #############
    classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10))

    train_model = TrainModel(classifier)

    # Move model to assigned GPU (see torch.cuda.set_device(args.local_rank))
    train_model.cuda()
    if distributed:
        train_model = DDP(train_model, device_ids=[args.local_rank], output_device=args.local_rank)
    # ############################################

    # ############ SETUP OPTIMIZER #############
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    # ##########################################

    # ############# SETUP TRAINING ##############
    trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer, model_components=classifier)

    callbacks = create_callbacks_for(trainer, is_first_worker)

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
        image = image.cuda()

        logits = classifier(image)
        probabilities = torch.softmax(logits, dim=-1)

        predicted_label = torch.argmax(probabilities)

        logger.info(f"real label = {real_label}, predicted label = {predicted_label}\n")
