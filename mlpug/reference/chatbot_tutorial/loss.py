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


def masked_loss(per_sample_loss, mask, average_loss=True):
    """

    :param per_sample_loss: (seq_length, batch_size)
    :param mask: (seq_length, batch_size)
    :param average_loss: If True, calculates the average loss over the samples, else sums the sample losses

    :return: loss

    loss: scalar (as tensor): average, or summed, loss over all samples

    """
    loss = per_sample_loss.masked_select(mask)

    return loss.mean() if average_loss else loss.sum()
