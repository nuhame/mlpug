from examples.persona_chatbot.shared_args import create_arg_parser as create_base_arg_parser
from examples.persona_chatbot.shared_args import describe_args as describe_base_args


def create_arg_parser(description="Finetune GPT2 as persona aware chatbot using PyTorch"):
    parser = create_base_arg_parser(description=description)

    parser.add_argument(
        '--num-dataloader-workers',
        type=int, required=False, default=2,
        help='Number of dataloader workers.')

    return parser


def describe_args(args, logger):
    logger.info(f"Num. dataloader workers: {args.num_dataloader_workers}")

    describe_base_args(args, logger)
