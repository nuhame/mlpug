from mlpug.mlpug_exceptions import CallbackInvalidException
from mlpug.trainers.callbacks.callback import Callback


class DistributedSamplerManager(Callback):

    def __init__(self, sampler, name="DistributedSamplerManager"):
        super(DistributedSamplerManager, self).__init__(name=name)

        if sampler is None or not hasattr(sampler, 'set_epoch') or not callable(sampler.set_epoch):
            raise CallbackInvalidException(name, "No valid DistributedSampler provided, missing set_epoch method")

        self.sampler = sampler
        
    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):

        self._set_epoch(start_epoch)

        return True

    def on_epoch_start(self, logs):
        """

        :param logs:

        :return: success (True or False)
        """

        self._set_epoch(logs["epoch"])

        return True

    def _set_epoch(self, epoch):
        self._log.debug(f"Set epoch {epoch} for sampler ...")
        self.sampler.set_epoch(epoch)

