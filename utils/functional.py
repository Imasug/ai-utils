import mxnet as mx
from tqdm import tqdm


def calc_mean_std(dataset, batch_size=2):
    """
    :param dataset:
        Dataset of tuple of data and target.
        The data must be NDArray(3 * H * W).
        Each H and W  must be same.
    :param batch_size:
    :return:
    """
    data, _ = dataset.__getitem__(0)
    _, h, w = data.shape

    data_loader = mx.gluon.data.DataLoader(dataset, batch_size=batch_size)

    data_sum = mx.nd.zeros(shape=(3,))
    data_sum_sq = mx.nd.zeros(shape=(3,))

    for data, _ in tqdm(data_loader):
        data_sum += data.sum(axis=[0, 2, 3])
        data_sum_sq += (data ** 2).sum(axis=[0, 2, 3])

    count = len(dataset) * h * w

    mean = data_sum / count
    var = data_sum_sq / count - mean ** 2
    std = var.sqrt()

    mean = [round(v.item(), 3) for v in mean.asnumpy()]
    std = [round(v.item(), 3) for v in std.asnumpy()]

    return mean, std
