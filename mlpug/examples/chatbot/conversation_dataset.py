"""
Conversation dataset class and utils
"""


# From https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
# but not PyTorch specific
def indexesFromSentence(voc, sentence, EOS_token):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


class IndexedSentencePairsDataset:
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

