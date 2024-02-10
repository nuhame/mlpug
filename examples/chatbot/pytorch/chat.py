import os

import argparse
from typing import List, Optional, Callable

import torch
from botshop.simple_bot import chat_with
from transformers import PreTrainedTokenizer, GPT2Tokenizer, GPT2DoubleHeadsModel

from botshop import ModelEvaluatorBase, IOProcessorBase, SimpleBot
from botshop.pytorch import BasicConversationEngine
from botshop.pytorch.utils import select_max

from basics.logging_utils import log_exception
from basics.logging import get_logger

import mlpug.pytorch as mlp

from examples.chatbot.special_tokens import SPECIAL_TOKENS_MAPPING
from examples.legacy.chatbot.tensorflow.original_transformer_tutorial.evaluation import Chatbot


def build_tokenizer_and_model_from(model_path: str, device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cpu")

    logger.info(f'Loading model checkpoint ...')
    checkpoint = torch.load(model_path, map_location=device)

    tokenizer = GPT2Tokenizer.from_pretrained(**checkpoint['hyper_parameters'])
    orig_num_tokens = len(tokenizer)
    num_special_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS_MAPPING)

    model = GPT2DoubleHeadsModel.from_pretrained(**checkpoint['hyper_parameters'])
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_special_tokens)

    model.load_state_dict(checkpoint['model'])

    return tokenizer, model


def load_persona_data(persona_file_path: str):
    with open(persona_file_path) as file:
        bot_personality = [line.rstrip() for line in file]

    return [line for line in bot_personality if len(line) > 0]

class ModelEvaluator(ModelEvaluatorBase):

    def __init__(
            self,
            model,
            speaker2_id
    ):
        super().__init__()

        self._model = model
        self._speaker2_id = speaker2_id

    def reset_state(self):
        pass

    def update_context(self, inputs, conversation_context, conversation_start):
        """

        Updates the conversation context with new conversation input data

        :param inputs: Dict:
        {
            "input_ids": <(new) input ids>,
            "token_type_ids": <(new) token_type_ids>
        }

        :param conversation_context: Dict, can be empty initially, and can be filled with
                                     context data when calling this method

        :param conversation_start: Boolean

        :return: None
        """
        if conversation_start:
            conversation_context["input_ids"] = inputs["input_ids"]
            conversation_context["token_type_ids"] = inputs["token_type_ids"]
        else:
            conversation_context["input_ids"] += inputs["input_ids"]
            conversation_context["token_type_ids"] += inputs["token_type_ids"]

    def predict_next_token(self, previous_token, prediction_context, conversation_context):
        """

        :param previous_token: Previous token, initially None
        :param prediction_context: Dict, can be used to store intermediate state
        :param conversation_context: Dict, conversation context, updated after each turn using `update_context`
            {
                "input_ids": <(new) input ids>,
                "token_type_ids": <(new) token_type_ids>
            }
        :return: model output logits
        """
        if previous_token is None:
            input_ids = conversation_context["input_ids"]
            token_type_ids = conversation_context["token_type_ids"]
        else:
            input_ids = prediction_context["input_ids"] + previous_token
            token_type_ids = prediction_context["token_type_ids"] + self._speaker2_id

        prediction_context["input_ids"] = input_ids
        prediction_context["token_type_ids"] = token_type_ids

        outputs = self._model(input_ids=input_ids, token_type_ids=token_type_ids)

        return outputs.logits[:, -1, :].squeeze()


class IOProcessor(IOProcessorBase):

    def __init__(
        self,
        tokenizer_func: Callable,
        detokenizer_func: Callable,
        bot_personality: List[str],
        bos: str = "<bos>",
        eos: str = "<eos>",
        speaker1: str = "<speaker1>",  # The user
        speaker2: str = "<speaker2>",  # The bot
        name: Optional[str] = None
    ):
        super().__init__(pybase_logger_name=name)

        self._tokenizer_func = tokenizer_func
        self._detokenizer_func = detokenizer_func

        self._personality_sequence_ids = self._tokenizer_func(' '.join(bot_personality))

        self.bos_id = self._get_token_id_of(bos)
        self.eos_id = self._get_token_id_of(eos)

        self.speaker1_id = self._get_token_id_of(speaker1)
        self.speaker2_id = self._get_token_id_of(speaker2)

        self._input_ids = None
        self._token_type_ids = None

    def reset_state(self):
        self._input_ids = None
        self._token_type_ids = None

    def process_inputs(self, inputs, conversation_start):
        """

        :param inputs: Dict with one or more different types of inputs
         {
            "chats": <list of input chats, interleaving uer and bot chats, starting with a user chat>,
            "is_user": <list, boolean indicating whether the chat is from the user or not>
         }
        :param conversation_start: Bool, if the given inputs are the initial inputs of a new
                                   conversation

        :return:
        """
        if conversation_start:
            self._input_ids = [self.bos_id] + self._personality_sequence_ids
            self._token_type_ids = [self.speaker2_id] * (len(self._personality_sequence_ids) + 1)

        chats, is_user = self._get_input_chats(inputs, conversation_start)

        for chat, chat_is_from_user in zip(chats, is_user):
            speaker_id = self.speaker1_id if chat_is_from_user else self.speaker2_id

            chat_ids = [speaker_id] + self._tokenizer_func(chat)

            self._input_ids += chat_ids
            self._token_type_ids += [speaker_id] * len(chat_ids)

        return {
            "input_ids": self._input_ids.copy(),
            "token_type_ids": self._token_type_ids.copy()
        }

    def process_response(self, response, scores=None, stop_sequence=None):
        """

        :param response:
        :param scores: response scores
        :param stop_sequence: Detected stop sequence

        :return: processed response, scores
        """
        return self._detokenizer_func(response), scores

    def _get_input_chats(self, inputs, conversation_start):
        """

        Get the raw input chats (and other inputs if any) from the inputs dict
        return the selected inputs split as sequence_inputs and other_inputs.

        sequence_inputs will be further processed, other_inputs not

        :param inputs: Dict with one or more different types of inputs
            {
                "chats": <list of input chats, interleaving uer and bot chats, starting with a user chat>,
                "is_user": <list, boolean indicating whether the chat is from the user or not>
            }
        :param conversation_start: Bool, if the given inputs are the initial inputs of a new
                                   conversation

        :return: sequence_inputs
        """
        if conversation_start:
            return inputs["chats"], inputs["is_user"]
        else:
            # return last bot response and follow-up user chat
            return inputs["chats"][-2:], inputs["is_user"][-2:]

    def _get_token_id_of(self, token) -> int:
        token_id = self._tokenizer_func(token)

        if len(token_id) != 1:
            raise ValueError(f'String {token} does not represent a single token in your tokenizer.')

        return token_id[0]

def create_arg_parser(description="Chat with persona aware chatbot"):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--model-path',
        type=str, required=True,
        help='Path to the trained model.')

    parser.add_argument(
        '--persona-file-path',
        type=str, required=True,
        help='Path to a file containing a newline separated list of sentences describing the bot persona.'
    )

    return parser


def describe_args(args, logger):
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Persona file path: {args.persona_file_path}")


if __name__ == '__main__':
    # ############# SETUP LOGGING #############
    mlp.logging.use_fancy_colors()
    logger = get_logger(os.path.basename(__file__))
    # ########################################

    # ############## PARSE ARGS ##############
    parser = create_arg_parser()

    args = parser.parse_args()

    describe_args(args, logger)

    tokenizer, model = build_tokenizer_and_model_from(args.model_path)

    bot_personality = load_persona_data(args.persona_file_path)

    io_processor = IOProcessor(
        tokenizer_func=tokenizer.encode,
        detokenizer_func=tokenizer.decode,
        bot_personality=bot_personality
    )

    model_evaluator = ModelEvaluator(model, speaker2_id=io_processor.speaker2_id)

    conversation_engine = BasicConversationEngine(
        io_processor,
        model_evaluator,
        select_token_func=select_max,
        sequence_end_token=io_processor.speaker1_id
    )

    chatbot = SimpleBot(conversation_engine)

    chat_with(chatbot)
