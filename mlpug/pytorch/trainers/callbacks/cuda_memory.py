import sys

import torch
import torch.distributed as dist

from mlpug.pytorch.trainers.callbacks import Callback

from mlpug.mlpug_exceptions import CallbackInvalidException


class EmptyCudaCache(Callback):

    def __init__(self, on_batch_training_complete=True, on_epoch_complete=True, name="EmptyCudaCache", **kwargs):
        super(EmptyCudaCache, self).__init__(name=name, **kwargs)

        self._on_batch_training_complete = on_batch_training_complete
        self._on_epoch_complete = on_epoch_complete

        self._log.debug(f"Will empty CUDA cache on batch training complete : {self._on_batch_training_complete}")
        self._log.debug(f"Will empty CUDA cache on epoch complete : {self._on_epoch_complete}")

    def on_batch_training_completed(self, training_batch, logs):
        if not self._on_batch_training_complete:
            return True

        torch.cuda.empty_cache()

        return True

    def on_epoch_completed(self, logs):
        if not self._on_epoch_complete:
            return True

        torch.cuda.empty_cache()

        return True


class LogCudaMemory(Callback):

    def __init__(self,
                 devices=None,
                 on_batch_training_start=False,
                 on_batch_training_complete=True,
                 name="LogCudaMemory",
                 **kwargs):
        """

        TODO: add memory statistics to log

        :param devices: list of indices of devices to log memory for
        :param on_batch_training_start:
        :param on_batch_training_complete:
        :param name:
        """

        super(LogCudaMemory, self).__init__(name=name, **kwargs)

        try:
            self._pynvml = __import__('pynvml')
        except Exception as e:
            raise CallbackInvalidException("The pynvml is not installed, please do "
                                           "`pip install pynvml` in your python environment")

        if devices is None:
            num_devices = dist.get_world_size() if dist.is_initialized() else 1
            devices = range(num_devices)
        else:
            num_devices = len(devices)

        self._log.debug(f"Devices to log memory usage for : {list(devices)}")

        self._num_devices = num_devices
        self._devices = devices

        self._on_batch_training_start = on_batch_training_start
        self._on_batch_training_complete = on_batch_training_complete

        self._max_memory_used = []

        self._pynvml.nvmlInit()

    def on_batch_training_start(self, training_batch, logs):
        if not self._on_batch_training_start:
            return True

        self._log_memory()

        return True

    def on_batch_training_completed(self, training_batch, logs):
        if not self._on_batch_training_complete:
            return True

        self._log_memory()

        return True

    def _log_memory(self):
        sys.stdout.write("\n")
        sys.stdout.write("|----------|----------------|----------------|----------------| -------------- |\n")
        sys.stdout.write("| Device # | Free           | Used           | Max. Used      | Total          |\n")
        sys.stdout.write("|----------|----------------|----------------|----------------| -------------- |\n")
        sys.stdout.flush()
        for device_index in self._devices:
            device_handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(device_handle)

            total_memory = self._to_mb(mem_info.total)
            memory_used = self._to_mb(mem_info.used)
            memory_free = self._to_mb(mem_info.free)

            if len(self._max_memory_used) < self._num_devices:
                self._max_memory_used += [memory_used]
            else:
                self._max_memory_used[device_index] = max(memory_used, self._max_memory_used[device_index])

            max_memory_used = self._max_memory_used[device_index]

            memory_values_str_format = "| {:<8} | {:<12}MB | {:<12}MB | {:<12}MB | {:<12}MB |\n"
            memory_values_str = memory_values_str_format.format(device_index,
                                                                memory_free,
                                                                memory_used,
                                                                max_memory_used,
                                                                total_memory)

            sys.stdout.write(memory_values_str)

        sys.stdout.write("|----------|----------------|----------------|----------------| -------------- |\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _to_mb(self, num_bytes):
        return int(round(num_bytes/(1024*1024), 0))
