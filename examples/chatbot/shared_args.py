from examples.shared_args import create_arg_parser as create_base_arg_parser
from examples.shared_args import describe_args as describe_base_args


def create_arg_parser(description="Finetune GPT2 as persona aware chatbot"):
    parser = create_base_arg_parser(description=description)

    parser.add_argument(
        '--pretrained-model',
        type=str, required=False, default='gpt2',
        help='Huggingface pre-trained model name of GPT2 model')

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

    describe_base_args(args, logger)


