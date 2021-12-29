import torch
import torch_xla.core.xla_model as xm
import torchvision as tv

# Import mlpug for Pytorch/XLA backend
import mlpug.pytorch.xla as mlp

# ################ SETTINGS ################
batch_size = 32
num_epochs = 10

progress_log_period = 500

torch.random.manual_seed(0)  # For reproducibility
mlp.logging.use_fancy_colors()
# ##########################################

# ############## PROCESS DATA ##############
# Load data
# Note : the data is already scaled

transform = tv.transforms.ToTensor()

training_data = tv.datasets.FashionMNIST('./mlpug-datasets-temp/', train=True, download=True, transform=transform)
test_data = tv.datasets.FashionMNIST('./mlpug-datasets-temp/', train=False, download=True, transform=transform)
# ##########################################

# ########## SETUP BATCH DATASETS ##########
training_dataset = torch.utils.data.DataLoader(training_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=3)

# Using the test set as a validation set, just for demonstration purposes
validation_dataset = torch.utils.data.DataLoader(test_data,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=3)
# ##########################################


# ############ BUILD THE MODEL #############
classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10))


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


device = xm.xla_device()

train_model = TrainModel(classifier, device)
classifier.to(device)

# ############################################


# ############ SETUP OPTIMIZER #############
optimizer = torch.optim.Adam(classifier.parameters(), eps=1e-7)  # Same parameters as for TF
# ##########################################

# ############# SETUP TRAINING ##############
trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer, model_components=classifier)

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
    mlp.callbacks.LogProgress(log_period=progress_log_period, set_names=['training', 'validation']),
]

manager = mlp.trainers.TrainingManager(trainer,
                                       training_dataset,
                                       num_epochs=num_epochs,
                                       callbacks=callbacks)

trainer.set_training_model(train_model)
# ##########################################

if __name__ == '__main__':
    # ################# START! #################
    manager.start_training()
    # ##########################################

    # ######### USE THE TRAINED MODEL ##########
    print("\nUsing the classifier ...")
    first_sample = next(iter(test_data))
    image = first_sample[0]
    real_label = first_sample[1]

    image = image.to(device)

    logits = classifier(image)
    probabilities = torch.softmax(logits, dim=-1)

    predicted_label = torch.argmax(probabilities)

    print(f"real label = {real_label}, predicted label = {predicted_label}\n")
