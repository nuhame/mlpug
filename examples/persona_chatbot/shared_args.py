import logging

from examples.shared_args import create_arg_parser as create_base_arg_parser
from examples.shared_args import describe_config as describe_base_config


def create_arg_parser(description="Finetune GPT2 as persona aware chatbot"):
    parser = create_base_arg_parser(description=description)

    parser.add_argument(
        '--pretrained-model',
        type=str, required=False, default='gpt2',
        help='Huggingface pre-trained model name of GPT2 model')

    parser.add_argument(
        '--max-conversations',
        type=int, required=False, default=None,
        help='Num. of conversations to use during training. '
             'This is typically used for debugging or demo purposes.')

    parser.add_argument(
        '--num-choices',
        type=int, required=False, default=None,
        help='Num. of reply choices')

    parser.add_argument(
        '--sequence-length-outlier-threshold',
        type=float, required=False, default=0.05,
        help='Fraction of samples to discard, in order to remove samples with very long (outlier) sequence length')

    parser.add_argument(
        '--force-generate-samples',
        action='store_true',
        help='When flag is set, will regenerate multiple choice conversation samples and re-cache the samples')

    parser.add_argument(
        '--lr-warmup-schedule',
        action='store_true',
        help='When given, will apply a LR schedule, starting with a warmup period after which the LR '
             'is cooled down again to 0 at the end of the last epoch.  This LR schedule works on the batch level.')

    parser.add_argument(
        '--lr-warmup-epochs',
        type=int, required=False, default=1,
        help='LR warmup period in epochs (will be converted to batch iterations). '
             'Only relevant when --lr-warmup-schedule flag is given.')

    parser.add_argument(
        '--weight-decay',
        type=float, required=False, default=0.0,
        help='Weight decay')

    parser.add_argument(
        '--dropout-rate',
        type=float, required=False, default=0.1,
        help='Dropout rate')

    parser.add_argument(
        '--lm-loss-weight',
        type=float, required=False, default=1.0,
        help='LM task loss weight in overall loss, combined with Next Sentence Prediction loss')

    parser.add_argument(
        '--activation-checkpointing',
        action='store_true',
        help='Enable activation checkpointing')

    parser.add_argument(
        '--describe-logs-object',
        action='store_true',
        help='When flag is set, will log a description of the logs object for '
             'demonstration or debugging purposes')

    parser.add_argument(
        '--inspect-sliding-windows',
        action='store_true',
        help='When flag is set, will log info related to the sliding metric windows used for '
             'demonstration or debugging purposes')

    return parser


def describe_config(
    pretrained_model: str,
    max_conversations: int | None,
    num_choices: int | None,
    sequence_length_outlier_threshold: float,
    force_generate_samples: bool,
    lr_warmup_schedule: bool,
    lr_warmup_epochs: int,
    weight_decay: float,
    dropout_rate: float,
    lm_loss_weight: float,
    activation_checkpointing: bool,
    describe_logs_object: bool,
    inspect_sliding_windows: bool,
    logger: logging.Logger | None = None,
    **kwargs,
) -> None:
    """
    Log persona chatbot training configuration.

    :param pretrained_model: HuggingFace pre-trained model name.
    :param max_conversations: Max conversations to use (None = all).
    :param num_choices: Number of reply choices.
    :param sequence_length_outlier_threshold: Fraction of samples to discard.
    :param force_generate_samples: Force regenerate samples.
    :param lr_warmup_schedule: Use LR warmup schedule.
    :param lr_warmup_epochs: LR warmup epochs.
    :param weight_decay: Weight decay.
    :param dropout_rate: Dropout rate.
    :param lm_loss_weight: LM loss weight.
    :param activation_checkpointing: Enable activation checkpointing.
    :param describe_logs_object: Log description of logs object.
    :param inspect_sliding_windows: Inspect sliding metric windows.
    :param logger: Logger to use.
    :param kwargs: Additional arguments passed to base describe_config.
    """
    describe_base_config(logger=logger, **kwargs)

    logger.info(f"  pretrained_model: {pretrained_model}")
    logger.info(f"  max_conversations: {max_conversations}")
    logger.info(f"  num_choices: {num_choices}")
    logger.info(f"  sequence_length_outlier_threshold: {sequence_length_outlier_threshold}")
    logger.info(f"  force_generate_samples: {force_generate_samples}")
    logger.info(f"  lr_warmup_schedule: {lr_warmup_schedule}")
    logger.info(f"  lr_warmup_epochs: {lr_warmup_epochs}")
    logger.info(f"  weight_decay: {weight_decay}")
    logger.info(f"  dropout_rate: {dropout_rate}")
    logger.info(f"  lm_loss_weight: {lm_loss_weight}")
    logger.info(f"  activation_checkpointing: {activation_checkpointing}")
    logger.info(f"  describe_logs_object: {describe_logs_object}")
    logger.info(f"  inspect_sliding_windows: {inspect_sliding_windows}")
