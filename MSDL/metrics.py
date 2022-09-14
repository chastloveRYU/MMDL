import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def precision(self):
        return np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)

    def recall(self):
        return np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)

    def F1_Score(self, method=None):
        score = 2 * self.precision() * self.recall() / (self.precision() + self.recall())
        if method is not None:
            if method == 'macro_average':
                return np.nanmean(score)
            elif method == 'micro_average':
                return self.Accuracy()
        else:
            return score

    def Kappa_Coefficient(self):
        num_classes = self.confusion_matrix.shape[0]
        pe = 0
        for i in range(num_classes):
            pe += self.confusion_matrix[i, :].sum() * self.confusion_matrix[:, i].sum()
        pe = pe / self.confusion_matrix.sum() ** 2
        kappa = (self.Accuracy() - pe) / (1 - pe)

        return kappa

    def Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # Acc = np.nanmean(Acc)
        return Acc

    def Mean_Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return IoU

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == "__main__":
    evaluator = Evaluator(6)

    truth = np.array([1, 3, 3, 2, 1])
    pred = np.array([1, 2, 3, 2, 3])

    evaluator.add_batch(truth, pred)

    acc = evaluator.Pixel_Accuracy()
    kappa = evaluator.Kappa_Coefficient()
    print(acc)
    print(kappa)

    #
    # truth
    #

    ########pred########
