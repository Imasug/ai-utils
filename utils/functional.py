import mxnet as mx


def calc_mean_std(dataset, batch_size=2):
    """
    :param dataset:
        Dataset of NDArray(3 * H * W). Each H and W  must be same.
    :param batch_size:
    :return:
    """
    _, h, w = dataset.__getitem__(0).shape

    data_loader = mx.gluon.data.DataLoader(dataset, batch_size=batch_size)

    data_sum = mx.nd.zeros(shape=(3,))
    data_sum_sq = mx.nd.zeros(shape=(3,))

    for data in data_loader:
        data_sum += data.sum(axis=[0, 2, 3])
        data_sum_sq += (data ** 2).sum(axis=[0, 2, 3])

    count = len(dataset) * h * w

    mean = data_sum / count
    var = data_sum_sq / count - mean ** 2
    std = var.sqrt()

    mean = [round(v.item(), 3) for v in mean.asnumpy()]
    std = [round(v.item(), 3) for v in std.asnumpy()]

    return mean, std
