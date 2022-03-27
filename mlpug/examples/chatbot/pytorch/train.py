import torch
import mlpug.pytorch as mlp


class TrainingProcess(mlp.Base):

    def __init__(self, rank, args, world_size, name="TrainingProcess"):
        # Disable logging below warning level, when not primary process/device
        disable_logging = world_size > 0 and rank > 0

        # Logger name shows process/device index when distributed
        logger_name = None
        if world_size > 0:
            logger_name = f"[Device {rank}] {name}"

        super().__init__(disable_logging=disable_logging, pybase_logger_name=logger_name)

        self._rank = rank
        self._args = args
        self._world_size = world_size

    def setup(self):
        self._set_random_seed()

    def start(self):
        pass

    def _set_random_seed(self):
        torch.random.manual_seed(args.seed)  # For reproducibility
