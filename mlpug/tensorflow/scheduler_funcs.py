import tensorflow as tf

from mlpug.scheduler_funcs import LRWarmupSchedule


class LambdaLR:
    """
    This is an experimental LR scheduler for Tensorflow
    """

    def __init__(self, optimizer, lr_scheduling_func):

        self.optimizer = optimizer
        self.lr_scheduling_func = lr_scheduling_func

        self._base_lr = tf.identity(self.optimizer.learning_rate)

    def step(self, iter):
        lr_factor = self.lr_scheduling_func(iter)
        self.optimizer.learning_rate = self._base_lr * lr_factor
