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


def masked_loss(per_sample_loss, mask):
    """

    :param per_sample_loss: (seq_length, batch_size)
    :param mask: (seq_length, batch_size)

    :return: average loss, summed loss, num_samples
    """
    loss = per_sample_loss.masked_select(mask)

    num_samples = mask.sum()
    loss_sum = loss.sum()

    loss = loss_sum/num_samples

    return loss, loss_sum, num_samples
