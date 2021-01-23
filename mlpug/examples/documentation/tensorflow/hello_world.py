# Partially based on https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb

import tensorflow as tf

# Import mlpug for Tensorflow backend
import mlpug.tensorflow as mlp

# ################ SETTINGS ################
batch_size = 32
num_epochs = 10

progress_log_period = 500

mlp.logging.use_fancy_colors()
# ##########################################

# ############## PROCESS DATA ##############
# Load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Scale
train_images = train_images / 255.0
test_images = test_images / 255.0
# ##########################################

# ########## SETUP BATCH DATASETS ##########
# In this example we don't use the test set yet

training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
# ##########################################

# ############ BUILD THE MODEL #############
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


# MLPug needs a TrainModel that outputs the loss
class TrainModel(tf.keras.Model):
    def __init__(self, model):
        super(TrainModel, self).__init__()

        self.model = model
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, batch_data, evaluate_settings, inference_mode=None):
        images, true_labels = batch_data

        predicted_labels = self.model(images)
        return self.loss_func(true_labels, predicted_labels)


train_model = TrainModel(model)
# ##########################################

# ############ SETUP OPTIMIZER #############
optimizer = tf.keras.optimizers.Adam()
# ##########################################

# ############# SETUP TRAINING ##############
trainer = mlp.trainers.DefaultTrainer(optimizer, model, eager_mode=True)

# At minimum you want to evaluate the training loss and log the training progress
average_loss_evaluator = mlp.evaluation.MetricEvaluator(trainer=trainer, name="AverageLossEvaluator")
callbacks = [
    mlp.callbacks.TrainingMetricsLogger(metric_evaluator=average_loss_evaluator),
    mlp.callbacks.LogProgress(log_period=progress_log_period),
]

manager = mlp.trainers.TrainingManager(trainer,
                                       training_dataset,
                                       num_epochs=num_epochs,
                                       num_batches_per_epoch=int(training_dataset.cardinality()),
                                       callbacks=callbacks)

trainer.set_training_model(train_model)
# ##########################################

# ################# START! #################
manager.start_training()
# ##########################################
