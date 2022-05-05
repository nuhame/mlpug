import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from basics.logging_utils import log_exception
from basics.logging import get_logger

import mlpug.pytorch as mlp

from examples.chatbot.training_process import TrainingProcess as TrainingProcessBase

module_logger = get_logger(os.path.basename(__file__))

try:
    from transformers import GPT2Config, GPT2Tokenizer, GPT2DoubleHeadsModel, AdamW
    from transformers.trainer_pt_utils import get_parameter_names
except Exception as e:
    log_exception(module_logger, "Please `pip install transformers`", e)


class GatherNextSentencePredictionData:

    def __init__(self, is_primary=True, is_distributed=False, num_devices=1):
        self.is_primary = is_primary
        self.is_distributed = is_distributed
        self.num_devices = num_devices

        self.softmax = torch.nn.Softmax()

    def __call__(self, batch, auxiliary_results, **kwargs):
        # Batch is a tuple with the following items :
        # [0] input_ids_batch,
        # [1] token_type_ids_batch,
        # [2] token_labels_ids_batch,
        # [3] last_token_idx_batch,
        # [4] reply_class_batch
        targets = batch[4]

        nsp_logits = auxiliary_results["nsp_logits"]

        # TODO: check nsp_logits shape
        prediction_probability = self.softmax(nsp_logits)
        predictions = torch.argmax(prediction_probability)

        if self.is_distributed:
            targets = self._gather(targets)
            predictions = self._gather(predictions)

        if targets is not None:
            targets = targets.cpu().numpy()

        if predictions is not None:
            predictions = predictions.cpu().numpy()

        return targets, predictions

    def _gather(self, tensor):
        gathered_tensors = None

        # Gather the targets and predictions on primary device
        if self.is_primary:
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.num_devices)]

        torch.distributed.gather(tensor, gather_list=gathered_tensors)

        if self.is_primary:
            gathered_tensors = torch.concat(gathered_tensors, dim=0)

        return gathered_tensors


# MLPug needs a TrainModel that outputs the loss
class TrainModel(torch.nn.Module):
    def __init__(self, model, lm_loss_weight, device):
        super(TrainModel, self).__init__()

        self.model = model
        self.lm_loss_weight = lm_loss_weight
        self.device = device

    def forward(self, batch_data, evaluate_settings, inference_mode=None):
        input_ids_batch, \
            token_type_ids_batch, \
            token_labels_ids_batch, \
            last_token_idx_batch, \
            reply_class_batch = batch_data

        input_ids_batch = input_ids_batch.to(self.device)
        token_type_ids_batch = token_type_ids_batch.to(self.device)
        token_labels_ids_batch = token_labels_ids_batch.to(self.device)
        last_token_idx_batch = last_token_idx_batch.to(self.device)
        reply_class_batch = reply_class_batch.to(self.device)

        results = self.model(
            input_ids=input_ids_batch,
            token_type_ids=token_type_ids_batch,
            labels=token_labels_ids_batch,
            mc_token_ids=last_token_idx_batch,
            mc_labels=reply_class_batch)

        loss = (results.loss*self.lm_loss_weight+results.mc_loss)/(self.lm_loss_weight+1.0)

        return {
            "loss": loss,
            "auxiliary_results": {
                # required to calculate next sentence prediction (classification) quality
                "nsp_logits": results.mc_logits
            }
        }


# See TrainingProcessBase for more information on the different methods implemented here
# Here we implement the methods that are specific to our problem and specific to our ML library, PyTorch.
class TrainingProcess(TrainingProcessBase):

    MLPUG_MODULE = mlp

    def _setup_compute(self):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(self.rank)

            if self.is_distributed:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'

                dist.init_process_group(backend='nccl',
                                        rank=self.rank,
                                        world_size=self._num_devices)

                self._log.info(f"Training using multiple GPUs: Using GPU {self.rank}/{self.num_devices}")
            else:
                self._log.info(f"Single device mode : Using GPU {self.rank}")
        else:
            if self.is_distributed:
                self._log.error(f"No GPUs available for data distributed training over multiple GPUs")
                return

            self._log.info(f"Single device mode : Using CPU")

        self._device = torch.device("cuda" if use_cuda else "cpu")

    def _load_dataset(self):
        # In distributed training mode, we allow the primary process to download the data first.
        # After that the secondary processes can load the data from a cache.
        if self.is_distributed and not self.is_primary:
            dist.barrier()

        dataset = super()._load_dataset()

        if self.is_distributed and self.is_primary:
            dist.barrier()

        return dataset

    def _setup_batch_datasets(self):
        training_sampler = None
        validation_sampler = None
        if self.is_distributed:
            training_sampler = torch.utils.data.distributed.DistributedSampler(self._sample_training_set)
            validation_sampler = torch.utils.data.distributed.DistributedSampler(self._sample_validation_set)

        self._batch_training_set = torch.utils.data.DataLoader(
            self._sample_training_set,
            batch_size=self._args.batch_size,
            shuffle=False,  # Samples already shuffled
            sampler=training_sampler,
            num_workers=self._args.num_dataloader_workers)

        # Using the test set as a validation set, just for demonstration purposes
        self._batch_validation_set = torch.utils.data.DataLoader(
            self._sample_validation_set,
            batch_size=self._args.batch_size,
            shuffle=False,  # Samples already shuffled
            sampler=validation_sampler,
            num_workers=self._args.num_dataloader_workers)

    def _build_model(self):
        if self.is_distributed and not self.is_primary:
            dist.barrier()

        model_config = self._gather_model_config()

        self._log.info(f"Loading pre-trained GPT-2 model : {self._args.pretrained_model}")
        # Load pre-trained GPT-2 model
        self._model = GPT2DoubleHeadsModel(model_config)

        self._model.resize_token_embeddings(new_num_tokens=self._orig_num_tokens + self._num_special_tokens)

        self._log.info(f"Configuration of loaded model : \n{model_config}")

        if self.is_distributed and self.is_primary:
            dist.barrier()

    def _setup_training_model(self):
        self._training_model = TrainModel(self._model, self._args.lm_loss_weight, self._device)

        self._training_model.to(self._device)
        if self.is_distributed:
            self._training_model = DDP(self._training_model, device_ids=[self.rank])

    def _build_optimizer(self):
        # Adapted from Huggingface Transformers package v4.17, Trainer::create_optimizer, transformers/trainer.py:827
        decay_parameters = get_parameter_names(self._model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self._model.named_parameters() if n in decay_parameters],
                "weight_decay": self._args.weight_decay,
            },
            {
                "params": [p for n, p in self._model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, **{
            "lr": self._args.learning_rate,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        })

    def _create_gather_loss_function(self):
        # Using default implementation.
        # In distributed training mode, this will average the loss of the batches on all devices
        return mlp.evaluation.GatherLossSimple(requester=str(self))

    def _create_gather_classification_data_function(self):
        # In distributed training mode, this will gather the Next Sentence Predictions and Targets
        return GatherNextSentencePredictionData(is_primary=self.is_primary,
                                                is_distributed=self.is_distributed,
                                                num_devices=self.num_devices)

    @staticmethod
    def get_logger_info(rank, num_devices, name):
        """

        :param rank:
        :param num_devices:
        :param name:

        :return: (Logger name: String, disable_logging: Boolean)
        """

        if num_devices > 1:
            return f"[Device {rank}/{num_devices}] {name}", rank != 0
        else:
            return name, False
