import argparse

from mlpug.examples.chatbot.shared import create_argument_parser

if __name__ == "__main__":

    parser = create_argument_parser()

    args = parser.parse_args()

    ##################################################
    #
    # [START] Setup
    #
    ##################################################

    # ############ Conversations dataset #############
    # Default word tokens

    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    dataset_path = args.dataset_path
    base_dataset_filename = args.base_dataset_filename
    logger.info(f"dataset_path : {dataset_path}")
    logger.info(f"base_dataset_filename : {base_dataset_filename}")

    ##################################################

