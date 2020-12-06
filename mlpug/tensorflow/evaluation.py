from mlpug.evaluation import MetricEvaluatorBase

from mlpug.utils import is_chunkable


def forward_loss(loss, **kwargs):
    return loss.numpy(), 1


class MetricEvaluator(MetricEvaluatorBase):

    def __init__(self, *args, batch_metric_funcs=None, name="MetricEvaluator", **kwargs):
        if batch_metric_funcs is None:
            batch_metric_funcs = {
                "loss": forward_loss
            }

        super().__init__(*args, batch_metric_funcs=batch_metric_funcs, name=name, **kwargs)

    def _create_default_model_evaluate_func(self):

        def evaluate_loss(batch, evaluate_settings=None):
            if is_chunkable(batch):
                # Get raw batch
                batch = batch[:]

            results = self._trainer.evaluate_loss(
                batch,
                inference_mode=True,
                evaluate_settings=evaluate_settings)

            return results

        return evaluate_loss
