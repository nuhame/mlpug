from mlpug.trainers.callbacks.lr_scheduler_wrapper import LRSchedulerWrapperBase

from statistics import mean


class LRSchedulerWrapper(LRSchedulerWrapperBase):

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
            lr = []
            for group in optimizer.param_groups:
                lr.append(group['lr'])

            current_lr[name] = mean(lr)

        return current_lr
