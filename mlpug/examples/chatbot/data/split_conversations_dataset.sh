#!/usr/bin/env bash

# Run this script from the root of the mlpug package directory (mlpug/mlpug/):
# mlpug/examples/chatbot/data/split_conversations_dataset.sh
python mlpug/examples/chatbot/split_conversations_dataset.py --conversations-dataset  ~/Projects/Nuhame/data/cmdc-sentence-pairs-with-voc-max-len-40-min-word-occurance-3-26112020.pickle \
                                                             --split 0.7 0.3
