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


class SegMetrics:

    def __init__(self, stats: np.ndarray):
        self.stats = stats

    @classmethod
    def create(
            cls,
            target: np.ndarray,
            prediction: np.ndarray,
            cls_num: int
    ):
        stats = []
        for c in range(0, cls_num):
            c_target = (target == c)
            c_prediction = (prediction == c)
            correct = np.count_nonzero(c_target * c_prediction)
            target_total = np.count_nonzero(c_target)
            prediction_total = np.count_nonzero(c_prediction)
            total = target_total + prediction_total - correct
            stats.append([correct, target_total, total])
        return cls(np.array(stats))

    def get_pixel_accuracy(self):
        correct, target_total, _ = self.stats.sum(axis=0)
        return correct / target_total

    def get_mean_accuracy(self):
        accuracy = np.array([])
        for stat in self.stats:
            correct, target_total, _ = stat
            if target_total == 0:
                continue
            accuracy = np.append(accuracy, correct / target_total)
        return accuracy.mean()

    def get_mean_iou(self):
        iou = np.array([])
        for stat in self.stats:
            correct, _, total = stat
            if total == 0:
                continue
            iou = np.append(iou, correct / total)
        return iou.mean()

    def get_mean_f1_score(self):
        f1_score = np.array([])
        for stat in self.stats:
            correct, _, total = stat
            if total == 0:
                continue
            f1_score = np.append(f1_score, 2 * correct / (total + correct))
        return f1_score.mean()

    def __add__(self, other):
        stats = self.stats + other.stats
        return SegMetrics(stats)


def get_seg_metrics(
        target: np.ndarray,
        prediction: np.ndarray,
        cls_num: int
) -> SegMetrics:
    """
    :param target: B, C, H, W
    :param prediction: B, C, H, W
    :param cls_num:
    :return:
    """
    return SegMetrics.create(target, prediction, cls_num)
