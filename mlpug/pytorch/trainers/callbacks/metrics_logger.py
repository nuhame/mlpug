import torch

from mlpug.trainers.callbacks.metrics_logger import *

from mlpug.utils import is_chunkable


class TestMetricsLogger(TestMetricsLoggerBase):

    def _evaluate_loss(self, batch, evaluate_settings=None):
        if is_chunkable(batch):
            # Get raw batch
            batch = batch[:]

        with torch.no_grad():
            loss, auxiliary_results = self.trainer.evaluate_loss(
                batch,
                inference_mode=True,
                evaluate_settings=evaluate_settings)

        return loss, auxiliary_results
