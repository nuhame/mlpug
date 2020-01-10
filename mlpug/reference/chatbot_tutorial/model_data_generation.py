import os
import itertools

import torch

from basics.logging import get_logger

logger = get_logger(os.path.basename(__file__))


def indexesFromSentence(voc, sentence, EOS_token):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(sentences, voc, PAD_token, EOS_token):
    indexes_batch = [indexesFromSentence(voc, sentence, EOS_token) for sentence in sentences]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Batch dimension should be second (in order to partition over multiple GPUs)
    lengths = lengths.unsqueeze(0)

    padList = zeroPadding(indexes_batch, PAD_token)
    padVar = torch.LongTensor(padList)

    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(sentences, voc, PAD_token, EOS_token):
    indexes_batch = [indexesFromSentence(voc, sentence, EOS_token) for sentence in sentences]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, PAD_token)
    mask = binaryMatrix(padList, PAD_token)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch, PAD_token, EOS_token):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc, PAD_token, EOS_token)
    output, mask, max_target_len = outputVar(output_batch, voc, PAD_token, EOS_token)
    return inp, lengths, output, mask, max_target_len


class IndexedSentencePairsDataset(torch.utils.data.Dataset):
    def __init__(self, sentence_pairs, voc, EOS_token):
        super(IndexedSentencePairsDataset).__init__()

        self.sentence_pairs = sentence_pairs
        self.voc = voc

        self.EOS_token = EOS_token

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        input_sentence = self.sentence_pairs[idx][0]
        output_sentence = self.sentence_pairs[idx][1]

        return self._map_to_indices(input_sentence), self._map_to_indices(output_sentence)

    def _map_to_indices(self, sentence):
        return indexesFromSentence(self.voc, sentence, self.EOS_token)


def create_sentence_pairs_collate_fn(PAD_token, fixed_sequence_length=None):
    """

    `fixed_sequence_length` is important when training on TPU

    :param PAD_token:
    :type PAD_token:
    :param fixed_sequence_length:
    :type fixed_sequence_length:
    :return:
    :rtype:
    """

    if fixed_sequence_length:
        logger.info(f"Using fixed sequence lengths of {fixed_sequence_length} tokens.")

    def collate_fn(indexed_sentence_pairs):
        # Why is the sort required?
        # ==> This is a CuDNN requirement
        # ==> https://discuss.pytorch.org/t/why-lengths-should-be-given-in-sorted-order-in-pack-padded-sequence/3540
        # ==> Apparently solved now?
        indexed_sentence_pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

        input_batch, output_batch = [], []
        for pair in indexed_sentence_pairs:
            input_batch.append(pair[0])
            output_batch.append(pair[1])

        # ############# PROCESS INPUT BATCH #############
        input_lengths = torch.tensor([len(indexed_sentence) for indexed_sentence in input_batch], dtype=torch.short)
        # Batch dimension should be second (in order to partition over multiple GPUs)
        input_lengths = input_lengths.unsqueeze(0)

        if fixed_sequence_length:
            padded_input_batch = torch.ones(fixed_sequence_length, len(input_batch), dtype=torch.long) * PAD_token
            for idx, indexed_sentence in enumerate(input_batch):
                padded_input_batch[0:len(indexed_sentence), idx] = torch.LongTensor(indexed_sentence)
        else:
            padded_input_batch = zeroPadding(input_batch, PAD_token)
            padded_input_batch = torch.LongTensor(padded_input_batch)

        ################################################

        # ############# PROCESS OUTPUT BATCH ############
        if fixed_sequence_length:
            max_output_len = fixed_sequence_length
            padded_output_batch = torch.ones(fixed_sequence_length, len(output_batch), dtype=torch.long) * PAD_token
            for idx, indexed_sentence in enumerate(output_batch):
                padded_output_batch[0:len(indexed_sentence), idx] = torch.LongTensor(indexed_sentence)
            output_mask = padded_output_batch != PAD_token
        else:
            max_output_len = max([len(indexed_sentence) for indexed_sentence in output_batch])
            padded_output_batch = zeroPadding(output_batch, PAD_token)

            output_mask = binaryMatrix(padded_output_batch, PAD_token)
            output_mask = torch.BoolTensor(output_mask)

            padded_output_batch = torch.LongTensor(padded_output_batch)
        ################################################

        return padded_input_batch, input_lengths, padded_output_batch, output_mask, max_output_len

    return collate_fn
