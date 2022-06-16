import torch

from mlpug.base import Base
from mlpug.batch_chunking import ChunkableTupleBatchDim0


class BatchCollator(Base):

    def __init__(self, pad_token_idx, ignore_label_idx=-100, name=None):
        super().__init__(pybase_logger_name=name)

        self._pad_token_idx = pad_token_idx
        self._ignore_label_idx = ignore_label_idx

    def __call__(self, batch_samples):
        """

        :param batch_samples:

                One sample in the batch, list with num_choices conversation samples:
                (
                    input_ids,        # 0: input_ids
                    token_type_ids,   # 1: token_type_ids
                    token_label_ids,  # 2: labels
                    last_token_idx,   # 3: mc_token_ids
                    reply_class       # 4: 0 = not real reply, 1 = real reply. ==> mc_labels
                )

        :return:
        """

        batch_size = len(batch_samples)
        num_choices = len(batch_samples[0])

        if any([len(sample) != num_choices for sample in batch_samples]):
            raise ValueError("Number of choices should be the same for all samples in the batch")

        max_seq_len = max([max([len(sample[choice_idx][0]) for choice_idx in range(num_choices)])
                           for sample in batch_samples])

        input_ids_batch = self._pad_token_idx*torch.ones(
            (batch_size, num_choices, max_seq_len),
            dtype=torch.long)
        token_type_ids_batch = self._pad_token_idx*torch.ones(
            (batch_size, num_choices, max_seq_len),
            dtype=torch.long)
        token_labels_ids_batch = self._ignore_label_idx*torch.ones(
            (batch_size, num_choices, max_seq_len),
            dtype=torch.long)

        last_token_idx_batch = torch.zeros((batch_size, num_choices), dtype=torch.long)
        reply_class_batch = torch.zeros((batch_size,), dtype=torch.long)

        for s_idx, sample_choices in enumerate(batch_samples):
            for c_idx, sample in enumerate(sample_choices):
                input_ids, token_type_ids, token_label_ids, last_token_idx, _ = sample

                input_ids_batch[s_idx, c_idx, :len(input_ids)] = torch.LongTensor(input_ids)
                token_type_ids_batch[s_idx, c_idx, :len(token_type_ids)] = torch.LongTensor(token_type_ids)
                token_labels_ids_batch[s_idx, c_idx, :len(token_label_ids)] = torch.LongTensor(token_label_ids)

                last_token_idx_batch[s_idx] = last_token_idx

            reply_class_batch[s_idx] = [sample[4] for sample in sample_choices].index(1)

        return ChunkableTupleBatchDim0(input_ids_batch,
                                       token_type_ids_batch,
                                       token_labels_ids_batch,
                                       last_token_idx_batch,
                                       reply_class_batch)
