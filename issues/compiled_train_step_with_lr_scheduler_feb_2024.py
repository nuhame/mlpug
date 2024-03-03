import os

import torch
import torch.distributed as dist

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import GPT2Tokenizer, GPT2Config, GPT2DoubleHeadsModel
from transformers.trainer_pt_utils import get_parameter_names

from basics.logging import get_logger

from examples.persona_chatbot.datasets.conversations import ConversationSampleFactory
from examples.persona_chatbot.datasets.manager import DatasetManager
from examples.persona_chatbot.datasets.tokenizers import HFTokenizer
from examples.persona_chatbot.special_tokens import SPECIAL_TOKENS_MAPPING

from examples.persona_chatbot.pytorch.collation import BatchCollator
from examples.persona_chatbot.pytorch.train import TrainModel

import mlpug.pytorch as mlp
from examples.persona_chatbot.training_process import find_optimal_max_sequence_length, filter_out_too_long_sequences

mlp.logging.use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


def log_info(message, force=False):
    is_distributed = dist.is_initialized()
    is_primary = dist.get_rank() == 0

    if not force and (is_distributed and not is_primary):
        return

    module_logger.info(f"[Rank {dist.get_rank()}] {message}")


def execute_on_primary_device_first(func, *args, **kwargs):
    is_distributed = dist.is_initialized()
    is_primary = dist.get_rank() == 0

    if is_distributed and not is_primary:
        dist.barrier()

    results = func(*args, **kwargs)

    if is_distributed and is_primary:
        dist.barrier()

    return results


def create_tokenizer(pretrained_model_name_or_path: str):
    log_info(f"Loading Tokenizer for {pretrained_model_name_or_path} model ...")

    hf_tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)

    orig_num_tokens = len(hf_tokenizer.encoder)
    num_special_tokens = hf_tokenizer.add_special_tokens(SPECIAL_TOKENS_MAPPING)

    tot_num_tokens = orig_num_tokens + num_special_tokens

    return hf_tokenizer, tot_num_tokens


def create_dataset(
    hf_tokenizer,
    max_conversations,
    num_choices,
    outlier_threshold=0.05
):
    tokenizer_func = HFTokenizer(hf_tokenizer)

    sample_factory = ConversationSampleFactory(
        tokenizer_func,
        bos=SPECIAL_TOKENS_MAPPING['bos_token'],
        eos=SPECIAL_TOKENS_MAPPING['eos_token'],
        speaker1=SPECIAL_TOKENS_MAPPING['additional_special_tokens'][0],
        speaker2=SPECIAL_TOKENS_MAPPING['additional_special_tokens'][1])

    dataset_manager = DatasetManager(sample_factory)

    training_set = dataset_manager.get_dataset_for(
        "train",
        max_num_samples=max_conversations,
        num_choices_per_sample=num_choices
    )

    opt_max_sequence_length, max_sequence_length = find_optimal_max_sequence_length(
        training_set,
        outlier_threshold=outlier_threshold
    )

    training_set = filter_out_too_long_sequences(training_set, opt_max_sequence_length)

    return training_set, opt_max_sequence_length


def create_batch_dataset(dataset, batch_size, pad_token_id, max_sequence_length, num_dataloader_workers=2):
    is_distributed = dist.is_initialized()

    sampler = None
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    batch_dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Samples already shuffled
        sampler=sampler,
        num_workers=num_dataloader_workers,
        collate_fn=BatchCollator(
            pad_token_idx=pad_token_id,
            max_sequence_length=max_sequence_length),
        pin_memory=True)

    return batch_dataset


def create_model(pretrained_model_name_or_path, tot_num_tokens, dropout_rate=0.1):
    log_info(f"Loading pre-trained GPT-2 model : {pretrained_model_name_or_path}")

    resid_pdrop = embd_pdrop = attn_pdrop = dropout_rate

    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path, **{
        "resid_pdrop": resid_pdrop,
        "embd_pdrop": embd_pdrop,
        "attn_pdrop": attn_pdrop
    })

    # Load pre-trained GPT-2 model
    model = GPT2DoubleHeadsModel.from_pretrained(pretrained_model_name_or_path, config=model_config)

    model.resize_token_embeddings(new_num_tokens=tot_num_tokens) # self._orig_num_tokens + self._num_special_tokens)

    log_info(f"Configuration of loaded model : \n{model_config}")

    return model


def create_optimizer(model, learning_rate=1e-4, weight_decay=0.0):
    # Adapted from Huggingface Transformers package v4.17, Trainer::create_optimizer, transformers/trainer.py:827
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, **{
        "lr": learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    })

    return optimizer


def create_scheduler(optimizer, batch_dataset, lr_warmup_epochs, num_epochs):
    num_iters_per_epoch = len(batch_dataset)
    num_warmup_iters = lr_warmup_epochs * num_iters_per_epoch
    total_iters = num_epochs * num_iters_per_epoch

    log_info(f"Applying LR warmup schedule: \n"
             f"num_warmup_iters = {num_warmup_iters}\n"
             f"total_iters      = {total_iters}")

    lr_scheduling_func = mlp.scheduler_funcs.LRWarmupSchedule(num_warmup_iters, total_iters)

    return LambdaLR(optimizer, lr_scheduling_func)


def train_step(batch, training_model, optimizer, scheduler):
    optimizer.zero_grad()

    results = training_model(batch)
    loss = results["loss"]

    # TODO: add AMP
    loss.backward()

    optimizer.step()

    scheduler.step()

    return results


def worker_fn(rank, config, world_size):
    # This worker assumed that CUDA is available (at least one or more CUDA devices

    is_distributed = config["distributed"]

    torch.set_float32_matmul_precision('high')

    if is_distributed:
        log_info(f"Distributed Data Parallel (DDP) mode.", force=True)

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        torch.cuda.set_device(rank)
        device = torch.device("cuda")

        backend = 'nccl'

        log_info(f"Training using multiple GPUs: Using GPU {rank}/{world_size}", force=True)

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )

        log_info(f"Communication backend used for DDP: {backend}")
    else:
        device = torch.device("cuda")
        log_info(f"Single device mode : Using GPU")

    hf_tokenizer, tot_num_tokens = execute_on_primary_device_first(
        create_tokenizer,
        config["model_name"]
    )

    training_set, opt_max_sequence_length = execute_on_primary_device_first(
        create_dataset,
        hf_tokenizer,
        config["max_conversations"],
        config["num_choices"]
    )

    batch_training_set = create_batch_dataset(
        training_set,
        config["batch_size"],
        hf_tokenizer.pad_token_id,
        opt_max_sequence_length,
        num_dataloader_workers=config["num_dataloader_workers"]
    )

    model = execute_on_primary_device_first(
        create_model,
        config["model_name"],
        tot_num_tokens
    )

    training_model = TrainModel(model, lm_loss_weight=1.0, device=device)
    training_model.to(device)

    if is_distributed:
        device_ids = [rank]
        training_model = DDP(training_model, device_ids=device_ids)

    optimizer = create_optimizer(model, config["learning_rate"])

    num_epochs = config["num_epochs"]
    scheduler = create_scheduler(
        optimizer,
        batch_training_set,
        lr_warmup_epochs=1,
        num_epochs=num_epochs)

    for epoch in range(num_epochs):
        for step_idx, batch in enumerate(batch_training_set):
            results = train_step(batch, training_model, optimizer, scheduler)

            # TODO: Gather evaluation metrics












