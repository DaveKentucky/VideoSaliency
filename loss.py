import torch
from torch import nn
import numpy as np
import time


def normalize(s_map):
    """
    Normalize saliency map for evaluation metrics calculation

    :param s_map: input map (either predicted or ground truth)
    :type s_map: torch.Tensor
    :return: normalized map
    :rtype: torch.Tensor
    """
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map


class VideoSaliencyLoss(nn.Module):
    def __init__(self):
        super(VideoSaliencyLoss, self).__init__()

    def forward(self, pred, gt, fix):
        """
        Calculate the metrics.

        :param pred: predicted saliency map
        :type pred: torch.Tensor
        :param gt: ground truth saliency map
        :type gt: torch.Tensor
        :param fix: fixation map
        :type fix: torch.Tensor
        :return: loss metrics values (loss, AUC-Judd, SIM)
        :rtype: (torch.Tensor, float, torch.Tensor)
        """
        st = time.time()
        loss_auc = self.auc_Judd(pred, fix, True)
        print(f'AUC time: {((time.time() - st) / 60):.2f}')
        st = time.time()
        loss_sim = self.similarity(pred, gt)
        print(f'SIM time: {((time.time() - st) / 60):.2f}')
        loss = loss_auc + loss_sim
        return loss, loss_auc, loss_sim

    def similarity(self, pred, gt):
        """
        Calculates Similarity measure (SIM).

        :param pred: predicted saliency map
        :type pred: torch.Tensor
        :param gt: ground truth fixation map
        :type gt: torch.Tensor
        :return: SIM measure value
        :rtype: float
        """
        batch_size = pred.size(0)
        w = pred.size(1)
        h = pred.size(2)

        pred = normalize(pred)
        gt = normalize(gt)

        pred_sum = torch.sum(pred.view(batch_size, -1), 1)
        pred_expand = pred_sum.view(batch_size, 1, 1).expand(batch_size, w, h)

        gt_sum = torch.sum(gt.view(batch_size, -1), 1)
        gt_expand = gt_sum.view(batch_size, 1, 1).expand(batch_size, w, h)

        pred = pred / (pred_expand * 1.0)
        gt = gt / (gt_expand * 1.0)

        pred = pred.view(batch_size, -1)
        gt = gt.view(batch_size, -1)
        return torch.mean(torch.sum(torch.min(pred, gt), -1))

    def auc_Judd(self, pred, fix, show_plot=False):
        """
        Calculates Area Under the Curve measure (AUC) following T. Judd's implementation.
        :param pred: predicted saliency map
        :type pred: torch.Tensor
        :param fix: fixation map
        :type fix: torch.Tensor
        :param show_plot: if the result curve should be plotted
        :type show_plot: bool
        :return: AUC-Judd measure value
        :rtype: float
        """
        if len(pred.size()) == 3:
            pred = pred[0, :, :]
            fix = fix[0, :, :]
        pred = pred.cpu()
        fix = fix.cpu()
        pred = pred.detach().numpy()
        fix = fix.detach().numpy()

        # resize saliency map to fixation map size
        if not np.shape(pred) == np.shape(fix):
            from cv2.cv2 import resize
            pred = resize(pred, np.shape(fix))

        # normalize the saliency map
        pred = (pred - pred.min()) / (pred.max() - pred.min())

        # flatten the maps
        pred_flat = pred.flatten()
        fix_flat = fix.flatten()

        fixations = pred_flat[fix_flat > 0]
        num_fixations = len(fixations)
        num_pixels = len(pred_flat)

        all_threshes = sorted(fixations, reverse=True)
        tp = np.zeros((num_fixations + 2))  # true positives
        fp = np.zeros(num_fixations + 2)    # false positives
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1

        for i in range(num_fixations):
            thresh = all_threshes[i]
            above_thresh = sum(x >= thresh for x in pred_flat)  # total number of saliency map values above threshold
            # ratio saliency map values at fixation locations
            tp[i + 1] = float(i + 1) / num_fixations
            # ratio saliency map values at not fixated locations
            fp[i + 1] = float(above_thresh - i) / (num_pixels - num_fixations)

        score = np.trapz(tp, fp)    # trapezoidal rule application
        all_threshes = np.insert(all_threshes, 0, 0)
        all_threshes = np.append(all_threshes, 1)

        if show_plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.matshow(pred, cmap='gray')
            ax.set_title('Saliency map with fixations to be predicted')
            [y, x] = np.nonzero(fix)
            s = np.shape(pred)
            plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
            plt.plot(x, y, 'ro')

            ax = fig.add_subplot(1, 2, 2)
            plt.plot(fp, tp, '.b-')
            ax.set_title('Area under ROC curve: ' + str(score))
            plt.axis((0, 1, 0, 1))
            plt.show()

        return score
