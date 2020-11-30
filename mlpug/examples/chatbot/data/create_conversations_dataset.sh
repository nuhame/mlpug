#!/usr/bin/env bash

# Run this script from the root of the mlpug package directory (mlpug/mlpug/):
# mlpug/examples/chatbot/data/create_conversations_dataset.sh
python mlpug/examples/chatbot/chatbot_data_reference.py --corpus-path ~/Projects/Nuhame/data/cornell-movie-dialogs-corpus \
                                                        --output-file  ~/Projects/Nuhame/data/cmdc-sentence-pairs-with-voc-max-len-40-min-word-occurance-3-26112020.pickle \
                                                        --max-sequence-length 40 \
                                                        --min-word-count 3
