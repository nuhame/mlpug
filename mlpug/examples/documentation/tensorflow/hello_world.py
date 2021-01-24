# Partially based on https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb

import tensorflow as tf

# Import mlpug for Tensorflow backend
import mlpug.tensorflow as mlp

# ################ SETTINGS ################
batch_size = 32
num_epochs = 10

progress_log_period = 500

tf.random.set_seed(0)  # For reproducibility
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
training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)

# Using the test set as a validation set, just for demonstration purposes
validation_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
# ##########################################

# ############ BUILD THE MODEL #############
classifier = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


# MLPug needs a TrainModel that outputs the loss
class TrainModel(tf.keras.Model):
    def __init__(self, classifier):
        super(TrainModel, self).__init__()

        self.classifier = classifier
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, batch_data, evaluate_settings, inference_mode=None):
        images, true_labels = batch_data

        logits = self.classifier(images)
        return self.loss_func(true_labels, logits)


train_model = TrainModel(classifier)
# ##########################################

# ############ SETUP OPTIMIZER #############
optimizer = tf.keras.optimizers.Adam()
# ##########################################

# ############# SETUP TRAINING ##############
trainer = mlp.trainers.DefaultTrainer(optimizers=optimizer,
                                      model_components=classifier,
                                      eager_mode=True)

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
                                       num_batches_per_epoch=int(training_dataset.cardinality()),
                                       callbacks=callbacks)

trainer.set_training_model(train_model)
# ##########################################

if __name__ == '__main__':
    # ################# START! #################
    manager.start_training()
    # ##########################################

    # ######### USE THE TRAINED MODEL ##########
    print("\nUsing the classifier ...")
    image = test_images[0]
    real_label = test_labels[0]

    logits = classifier(tf.expand_dims(image, 0))
    probabilities = tf.nn.softmax(logits)

    predicted_label = tf.math.argmax(probabilities, axis=-1)

    print(f"real label = {real_label}, predicted label = {tf.squeeze(predicted_label)}\n")
