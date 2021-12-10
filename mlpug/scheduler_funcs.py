from mlpug.base import Base


class LRWarmupSchedule(Base):

    def __init__(self, num_warmup_iters, total_iters, name=None, **kwargs):
        super().__init__(pybase_logger_name=name, **kwargs)

        if num_warmup_iters < 0:
            self._log.warning(f"Number of warmup iterations must < 0 {num_warmup_iters}, setting to 0")
            num_warmup_iters = 0

        if total_iters <= num_warmup_iters:
            self._log.warning(f"Total of iterations {total_iters} must be > "
                              f"number of warmup iterations {num_warmup_iters}, setting to {num_warmup_iters+1}")
            total_iters = num_warmup_iters + 1

        self.num_warmup_iters = float(num_warmup_iters)
        self.total_iters = float(total_iters)

    def __call__(self, training_iter):
        training_iter = float(training_iter)

        if training_iter <= self.num_warmup_iters:
            return training_iter/self.num_warmup_iters
        else:
            return max((self.total_iters-training_iter)/(self.total_iters-self.num_warmup_iters), 0.0)


