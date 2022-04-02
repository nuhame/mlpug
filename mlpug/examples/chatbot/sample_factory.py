from typing import Optional, Callable, List

from itertools import chain

from mlpug.base import Base


class ChatSampleFactory(Base):

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

        self._bos = bos
        self._eos = eos

        self._speaker1 = speaker1
        self._speaker2 = speaker2

        self._ignore_label = ignore_label

        self._bos_id = self._tokenizer_func(self._bos)[0]
        self._eos_id = self._tokenizer_func(self._eos)[0]

        self._speaker1_id = self._tokenizer_func(self._speaker1)[0]
        self._speaker2_id = self._tokenizer_func(self._speaker2)[0]

    def __call__(self,
                 personality: List[str],
                 chat_history: List[str],
                 candidate_reply: str,
                 is_real_reply: bool):
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
        -100.... (until end of personality sequences)
        -100.... (until end of last history sequence)
        -100<candidate_reply>-100

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

        personality_sequence = ' '.join(personality)

        input_ids += self._tokenizer_func(personality_sequence)
        token_type_ids += [self._speaker2_id]*len(personality_sequence)
        token_label_ids += [self._ignore_label]*len(personality_sequence)

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
        token_label_ids += [self._ignore_label] + candidate_reply_ids + [self._ignore_label]

        last_token_idx = len(input_ids)-1

        reply_class = int(is_real_reply)

        return input_ids, token_type_ids, token_label_ids, last_token_idx, reply_class
