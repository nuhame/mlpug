import os

from functools import partial
from typing import List, Optional, Callable, Tuple

import argparse

import torch
from botshop.simple_bot import chat_with
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

from botshop import ModelEvaluatorBase, IOProcessorBase, SimpleBot
from botshop.pytorch import BasicConversationEngine
from botshop.pytorch.utils import select_max, filter_top_p, random_sample

from basics.logging import get_logger

import mlpug.pytorch as mlp

from examples.persona_chatbot.special_tokens import SPECIAL_TOKENS_MAPPING, SPECIAL_TOKENS

# ############# SETUP LOGGING #############
mlp.logging.use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))
# ########################################


def get_default_device() -> torch.device:
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    if cuda_available:
        return torch.device("cuda")
    elif mps_available:
        return torch.device("mps")
    else:
        return torch.device("cpu")


def build_tokenizer_and_model_from(
        model_path: str,
        device: Optional[torch.device] = None,
        logger=None
) -> Tuple[GPT2Tokenizer, GPT2DoubleHeadsModel]:
    if logger is None:
        logger = module_logger

    if device is None:
        device = get_default_device()

    logger.info(f"Using device: {device}")

    logger.info(f'Loading model checkpoint ...')
    checkpoint = torch.load(model_path, map_location=device)

    hyper_params = checkpoint['hyper_parameters']
    pretrained_model_name_or_path = hyper_params["pretrained_model"]

    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    orig_num_tokens = len(tokenizer)
    num_special_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS_MAPPING)

    model = GPT2DoubleHeadsModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_special_tokens)

    model.load_state_dict(checkpoint['model'])
    model = model.to(device)

    return tokenizer, model


def load_persona_data(persona_file_path: str):
    with open(persona_file_path) as file:
        bot_personality = [line.rstrip() for line in file]

    return [line for line in bot_personality if len(line) > 0]


def create_token_selection_func(top_p=0.70, sample_temp=0.9):
    if sample_temp <= 0.0:
        return select_max

    return partial(select_top_p_random, top_p=top_p, sample_temp=sample_temp)


def select_top_p_random(logits, top_p=0.70, sample_temp=0.9):
    logits = logits/sample_temp

    logits, top_p_index = filter_top_p(logits, top_p=top_p)
    p, selected_token = random_sample(logits)

    return p, top_p_index[selected_token]


def sequence_end_detector_func(
        response: List[int],
        scores: List[float],
        end_token_ids: List[int]
) -> Optional[List[int]]:

    if response[-1] in end_token_ids:
        return [response[-1]]
    else:
        return None


class ModelEvaluator(ModelEvaluatorBase):

    def __init__(
            self,
            model,
            speaker2_id
    ):
        super().__init__()

        self._model = model
        self._speaker2_id = speaker2_id

        self._device = self._model.device

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

        input_ids = torch.LongTensor(inputs["input_ids"]).to(self._device)
        token_type_ids = torch.LongTensor(inputs["token_type_ids"]).to(self._device)
        if conversation_start:
            conversation_context["input_ids"] = input_ids
            conversation_context["token_type_ids"] = token_type_ids
        else:
            conversation_context["input_ids"] = torch.cat(
                (conversation_context["input_ids"], input_ids)
            )
            conversation_context["token_type_ids"] = torch.cat(
                (conversation_context["token_type_ids"], token_type_ids)
            )

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
            # Prompt start of bot response
            previous_token = self._speaker2_id

            input_ids = conversation_context["input_ids"]
            token_type_ids = conversation_context["token_type_ids"]
        else:
            input_ids = prediction_context["input_ids"]
            token_type_ids = prediction_context["token_type_ids"]

        input_ids = torch.cat(
            (input_ids, torch.LongTensor([previous_token]).to(self._device))
        )
        token_type_ids = torch.cat(
            (token_type_ids, torch.LongTensor([self._speaker2_id]).to(self._device))
        )

        prediction_context["input_ids"] = input_ids
        prediction_context["token_type_ids"] = token_type_ids

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, token_type_ids=token_type_ids)

        return outputs.logits[-1, :].squeeze()


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
        input_ids = []
        token_type_ids = []
        if conversation_start:
            input_ids = [self.bos_id] + self._personality_sequence_ids
            token_type_ids = [self.speaker2_id] * (len(self._personality_sequence_ids) + 1)

        chats, is_user = self._get_input_chats(inputs, conversation_start)

        for chat, chat_is_from_user in zip(chats, is_user):
            speaker_id = self.speaker1_id if chat_is_from_user else self.speaker2_id

            chat_ids = [speaker_id] + self._tokenizer_func(chat)

            input_ids += chat_ids
            token_type_ids += [speaker_id] * len(chat_ids)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids
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

    parser.add_argument(
        '--sample-temp',
        type=float, required=False, default=0.9,
        help='Sampling temperature for random token sampling. When <= 0, no sampling is performed.'
    )

    parser.add_argument(
        '--top-p',
        type=float, required=False, default=0.7,
        help='top-p (nucleus) value to pre-select most likely tokens for random sampling'
    )

    return parser


def describe_args(args, logger):
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Persona file path: {args.persona_file_path}")
    logger.info(f"Sample temp.: {args.sample_temp}")
    logger.info(f"Top-p: {args.top_p}")


if __name__ == '__main__':
    logger = module_logger

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

    special_token_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    sequence_end_detector = partial(sequence_end_detector_func, end_token_ids=special_token_ids)

    select_token_func = create_token_selection_func(top_p=args.top_p, sample_temp=args.sample_temp)
    logger.info(f"Select token functions: {select_token_func}")

    conversation_engine = BasicConversationEngine(
        io_processor,
        model_evaluator,
        select_token_func=select_token_func,
        sequence_end_detector_func=sequence_end_detector,
        max_response_length=40
    )

    chatbot = SimpleBot(conversation_engine)

    chat_with(chatbot)
