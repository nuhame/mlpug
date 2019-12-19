import torch


def cross_entropy(inp, target):
    """
    Calculate cross entropy per sample

    :param inp: (batch_size, voc_size)
    :param target: (batch_size,)

    :return: cross entropy loss

    loss : (batch_size,) loss, per sample in the batch
    """
    return -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze())


def masked_average_loss(per_sample_loss, mask):
    """

    :param per_sample_loss: (seq_length, batch_size)
    :param mask: (seq_length, batch_size)

    :return: loss, total_elements

    loss: scalar (as tensor): average loss over all samples
    total_elements: scalar (as tensor): number of samples used based on mask

    """
    loss = per_sample_loss.masked_select(mask).mean()

    return loss
