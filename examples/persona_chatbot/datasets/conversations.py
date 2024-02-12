from typing import Optional, Callable, Tuple, List

import numpy as np

from mlpug.base import Base


ConversationSample = Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]


class ConversationSampleFactory(Base):

    def __init__(self,
                 tokenizer_func: Callable,
                 bos: str = "<bos>",
                 eos: str = "<eos>",
                 speaker1: str = "<speaker1>",  # The user
                 speaker2: str = "<speaker2>",  # The bot
                 ignore_label: str = -100,
                 name: Optional[str] = None):

        super().__init__(pybase_logger_name=name)

        self._tokenizer_func = tokenizer_func

        self._ignore_label = ignore_label

        self._bos_id = self._get_token_id_of(bos)
        self._eos_id = self._get_token_id_of(eos)

        self._speaker1_id = self._get_token_id_of(speaker1)
        self._speaker2_id = self._get_token_id_of(speaker2)

    def __call__(self,
                 personality: List[str],
                 chat_history: List[str],
                 candidate_reply: str,
                 is_real_reply: bool) -> ConversationSample:
        """

        input_ids:
        <bos>
        <personality sequences, seperated by spaces>
        <speaker1><history 1><speaker2><history 2><speaker 1><history 3> ...
        <speaker2><candidate_reply><eos>

        token_type_ids:
        <speaker2>
        <speaker2>.... (until end of personality sequences)
        <speaker1>....<speaker2>....<speaker1>... (until end of last history sequence)
        <speaker2>.... (until end of sequence)

        token_label_ids:
        -100
        -100... (until end of personality sequences)
        -100... (until end of last history sequence)
        # if is_real_reply:
        -100<candidate_reply>-100
        # else:
        -100... (until end of reply)

        :param personality: List with utterances describing bot personality
        :param chat_history: List with utterances describing the chat history, starting with chat of speaker1 (user)
                             and also ending with chat of speaker1
        :param candidate_reply: potential bot reply (speaker2)
        :param is_real_reply: True if candidate_reply is the real reply

        :return: To be used with GPT2DoubleHeadsModel
                 (
                    input_ids,        # input_ids
                    token_type_ids,   # token_type_ids
                    token_label_ids,  # labels
                    last_token_idx,   # mc_token_ids
                    reply_class       # 0 = not real reply, 1 = real reply. ==> mc_labels
                 )
        """

        input_ids = [self._bos_id]
        token_type_ids = [self._speaker2_id]
        token_label_ids = [self._ignore_label]

        personality_sequence_ids = self._tokenizer_func(' '.join(personality))

        input_ids += personality_sequence_ids
        token_type_ids += [self._speaker2_id]*len(personality_sequence_ids)
        token_label_ids += [self._ignore_label]*len(personality_sequence_ids)

        for chat_idx, chat in enumerate(chat_history):
            speaker_id = self._speaker2_id if chat_idx % 2 else self._speaker1_id

            chat_ids = [speaker_id] + self._tokenizer_func(chat)

            input_ids += chat_ids
            token_type_ids += [speaker_id] * len(chat_ids)
            token_label_ids += [self._ignore_label] * len(chat_ids)

        candidate_reply_ids = self._tokenizer_func(candidate_reply)
        reply_ids = [self._speaker2_id] + candidate_reply_ids + [self._eos_id]

        input_ids += reply_ids
        token_type_ids += [self._speaker2_id] * len(reply_ids)
        if is_real_reply:
            # Also predict EOS token, such that during inference we can recognize the end of the reply.
            token_label_ids += [self._ignore_label] + candidate_reply_ids + [self._eos_id]
        else:
            token_label_ids += [self._ignore_label]*len(reply_ids)

        last_token_idx = len(input_ids)-1

        reply_class = int(is_real_reply)

        # Storing as numpy arrays really makes a huge difference when pickling and unpickling the data
        return np.array(input_ids), np.array(token_type_ids), np.array(token_label_ids), last_token_idx, reply_class

    def _get_token_id_of(self, token) -> int:
        token_id = self._tokenizer_func(token)

        if len(token_id) != 1:
            raise ValueError(f'String {token} does not represent a single token in your tokenizer.')

        return token_id[0]
