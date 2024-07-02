from examples.shared_args import create_arg_parser as create_base_arg_parser
from examples.shared_args import describe_args as describe_base_args


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

    parser.add_argument(
        '--no-batch-level-validation-loss-evaluation',
        action='store_true',
        help='When flag is set, no batch-level validation loss will be evaluated during training, only the '
             'validation loss over the complete validation set will be evaluated once per epoch')

    return parser


def describe_args(args, logger):
    logger.info(f"Pre-trained model name: {args.pretrained_model}")

    logger.info(f"Max. num. conversations to use (None = all): {args.max_conversations}")
    logger.info(f"Num. of reply choices : {args.num_choices}")

    logger.info(f"Force (re)generate multiple choice conversation samples: {args.force_generate_samples}")

    logger.info(f"Fraction of samples to discard to reduce max. sequence length: "
                f"{args.sequence_length_outlier_threshold}")

    logger.info(f"Use LR warmup schedule: {args.lr_warmup_schedule}")
    logger.info(f"LR warmup epochs (if warmup LR schedule used): {args.lr_warmup_epochs}")

    describe_base_args(args, logger)

    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Dropout rate: {args.dropout_rate}")
    logger.info(f"Weight of LM loss on complete loss: {args.lm_loss_weight}")
    logger.info(f"Activation checkpointing: {args.activation_checkpointing}")
    logger.info(f"Log a description of logs object: {args.describe_logs_object}")
    logger.info(f"Inspect sliding metric windows: {args.inspect_sliding_windows}")

    logger.info(f"No batch-level validation loss evaluation: {args.no_batch_level_validation_loss_evaluation}")
