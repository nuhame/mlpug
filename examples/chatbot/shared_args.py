from examples.shared_args import create_arg_parser as create_base_arg_parser
from examples.shared_args import describe_args as describe_base_args


def create_arg_parser(description="Finetune GPT2 as persona aware chatbot"):
    parser = create_base_arg_parser(description=description)

    parser.add_argument(
        '--pretrained-model',
        type=str, required=False, default='gpt2',
        help='Huggingface pre-trained model name of GPT2 model')

    parser.add_argument(
        '--num-choices',
        type=int, required=False, default=None,
        help='Num. of reply choices')

    parser.add_argument(
        '--batch-chunk-size',
        type=int, required=False, default=None,
        help='Weight decay')

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
        type=float, required=False, default=2.0,
        help='LM task loss weight in overall loss, combined with Next Sentence Prediction loss')

    return parser


def describe_args(args, logger):
    logger.info(f"Pre-trained model name: {args.pretrained_model}")

    logger.info(f"Num. of reply choices : {args.num_choices}")

    describe_base_args(args, logger)

    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Dropout rate: {args.dropout_rate}")
    logger.info(f"Weight of LM loss on complete loss: {args.lm_loss_weight}")
