from statistics import mean

import torch

from mlpug.trainers.callbacks.lr_scheduler_wrapper import LRSchedulerWrapper as LRSchedulerWrapperBase

from mlpug.pytorch.multi_processing import MultiProcessingMixin


class LRSchedulerWrapperMixin(LRSchedulerWrapperBase):
    """
    NOTE: The following warning can occur when Automatic Mixed Precision and no update was made:
          UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`

          See https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814/2
    """

    def on_training_start(self, *args, **kwargs):
        success = super().on_training_start(*args, **kwargs)

        eager_mode = self.trainer.eager_mode
        if not eager_mode:
            self._log.debug("Enabling compilation of LR scheduling ...")

            compile_kwargs = self.trainer.compile_kwargs
            self._exec_schedulers = torch.compile(self._exec_schedulers, **compile_kwargs)

        return success

    def get_state(self):
        """

        :return: state, success (True or False)
        """
        state = {}
        for name, scheduler in self._schedulers.items():
            state[name] = scheduler.state_dict()

        return state, True

    def set_state(self, state):
        """
        :param state:
        :return: success (True or False)
        """

        success = True
        for name, scheduler_state in state.items():
            if name not in self._schedulers:
                self._log.error(f"Scheduler {name} not found, unable to set state, skipping ...")
                success = False
                continue

            self._schedulers[name].load_state_dict(scheduler_state)

        return success

    def _exec_schedulers(self, training_iter, model_quality=None):
        for name, scheduler in self._schedulers.items():
            if self._metric_to_monitor:
                scheduler.step(model_quality)
            else:
                scheduler.step()

        return True

    def _get_current_lr(self):
        """
        :return: dict with learning rate, per optimizer
        """

        current_lr = {}
        for name, optimizer in self.optimizers.items():
            group_lrs = []
            for group in optimizer.param_groups:
                group_lrs.append(group['lr'].item())

            current_lr[name] = mean(group_lrs)

        return current_lr


class LRSchedulerWrapper(MultiProcessingMixin, LRSchedulerWrapperMixin):
    pass
