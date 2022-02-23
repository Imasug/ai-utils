import numpy as np
from tqdm import tqdm


# TODO どういう状態のデータセットに対して平均、標準偏差を取得すべきか？
def calc_img_mean_std(dataset):
    array_sum = np.zeros(shape=(3,))
    array_sum_sq = np.zeros(shape=(3,))

    mean = 0.0
    std = 0.0
    area = 0
    count = len(dataset)
    bar = tqdm(dataset)

    for i, (img, _) in enumerate(bar, start=1):
        array = np.array(img).astype(np.int64) / 255
        h, w, _ = array.shape
        area += h * w
        array_sum += array.sum(axis=(0, 1))
        array_sum_sq += (array ** 2).sum(axis=(0, 1))

        if i % 100 == 0 or i == count:
            mean = array_sum / area
            var = array_sum_sq / area - mean ** 2
            std = np.sqrt(var)
            bar.set_description(f'mean: {mean}, std: {std}')

    return mean, std
