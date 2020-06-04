import torch

from mlpug.evaluation import MetricEvaluatorBase

from mlpug.utils import is_chunkable


class MetricEvaluator(MetricEvaluatorBase):

    def __init__(self, *args, name="MetricEvaluator", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def _create_default_model_evaluate_func(self):

        def evaluate_loss(batch, evaluate_settings=None):
            if is_chunkable(batch):
                # Get raw batch
                batch = batch[:]

            with torch.no_grad():
                results = self._trainer.evaluate_loss(
                    batch,
                    inference_mode=True,
                    evaluate_settings=evaluate_settings)

            return results

        return evaluate_loss

