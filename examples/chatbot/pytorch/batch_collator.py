import torch


class BatchCollator:

    def __init__(self, pad_token_idx, ignore_label_idx=-100):
        self._pad_token_idx = pad_token_idx
        self._ignore_label_idx = ignore_label_idx

    def __call__(self, batch_samples):
        """

        :param batch_samples:

                One sample in the batch:
                (
                    input_ids,        # input_ids
                    token_type_ids,   # token_type_ids
                    token_label_ids,  # labels
                    last_token_idx,   # mc_token_ids
                    reply_class       # 0 = not real reply, 1 = real reply. ==> mc_labels
                )

        :return:
        """

        batch_size = len(batch_samples)

        max_seq_len = max([len(sample[0]) for sample in batch_samples])

        input_ids_batch = self._pad_token_idx*torch.ones((batch_size, max_seq_len), dtype=torch.long)
        token_type_ids_batch = self._pad_token_idx*torch.ones((batch_size, max_seq_len), dtype=torch.long)
        token_labels_ids_batch = self._ignore_label_idx*torch.ones((batch_size, max_seq_len), dtype=torch.long)

        last_token_idx_batch = torch.zeros((batch_size, 1), dtype=torch.long)
        reply_class_batch = torch.zeros((batch_size, 1), dtype=torch.long)

        for sample_idx, sample in enumerate(batch_samples):
            input_ids, token_type_ids, token_label_ids, last_token_idx, reply_class = sample

            input_ids_batch[sample_idx, :len(input_ids)] = torch.LongTensor(input_ids)
            token_type_ids_batch[sample_idx, :len(token_type_ids)] = torch.LongTensor(token_type_ids)
            token_labels_ids_batch[sample_idx, :len(token_label_ids)] = torch.LongTensor(token_label_ids)

            last_token_idx_batch[sample_idx, 0] = last_token_idx
            reply_class_batch[sample_idx, 0] = reply_class

        return input_ids_batch, token_type_ids_batch, token_labels_ids_batch, last_token_idx_batch, reply_class_batch
