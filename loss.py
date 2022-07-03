import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
from cv2.cv2 import resize


def normalize(s_map):
    """
    Normalize saliency map tensor for evaluation metrics calculation

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


def normalize_numpy(s_map):
    """
    Normalize saliency map numpy array for evaluation metrics calculation

    :param s_map: input map (either predicted or ground truth)
    :type s_map: numpy.ndarray
    :return: normalized map
    :rtype: numpy.ndarray
    """
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)) * 1.0)
    return norm_s_map


class VideoSaliencyLoss(nn.Module):
    def __init__(self, mode='train'):
        super(VideoSaliencyLoss, self).__init__()
        self.mode = mode

    def forward(self, pred, gt, fix):
        """
        Calculate the metrics.

        :param pred: predicted saliency map
        :type pred: torch.Tensor
        :param gt: ground truth saliency map
        :type gt: torch.Tensor
        :param fix: fixation map
        :type fix: torch.Tensor
        :return: loss metrics values (loss, SIM, NSS)
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        loss_sim = self.similarity(pred, gt).cpu()
        loss_nss = self.nss(pred, fix).cpu()
        loss = torch.FloatTensor([0.0]).cpu()
        loss += loss_sim - loss_nss
        if self.mode == 'evaluate':
            loss_auc = self.auc_Judd(pred, fix)
            loss_cc = self.cc(pred, gt)
            return loss_sim, loss_nss, loss_auc, loss_cc
        return loss, loss_sim, loss_nss

    def similarity(self, pred, gt):
        """
        Calculates Similarity measure (SIM).

        :param pred: predicted saliency map
        :type pred: torch.Tensor
        :param gt: ground truth fixation map
        :type gt: torch.Tensor
        :return: SIM measure value
        :rtype: torch.Tensor
        """
        pred = pred.cuda()
        gt = gt.cuda()

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

    def auc_Judd(self, pred, gt, show_plot=False):
        """
        Calculates Area Under the Curve measure (AUC) following T. Judd's implementation.

        :param pred: predicted saliency map
        :type pred: torch.Tensor
        :param gt: ground truth fixation map
        :type gt: torch.Tensor
        :param show_plot: if the result curve should be plotted
        :type show_plot: bool
        :return: AUC-Judd measure value
        :rtype: float
        """
        pred = pred.cpu()
        gt = gt.cpu()

        # resize predicted saliency map if the sizes don't match
        if pred.size() != gt.size():
            pred = pred.squeeze(0).numpy()
            pred = torch.FloatTensor(resize(pred, (gt.size(2), gt.size(1)))).unsqueeze(0)

        # get a single frame from video clip data
        if len(pred.size()) == 3:
            pred = pred[0, :, :]
            gt = gt[0, :, :]

        pred = pred.detach().numpy()
        gt = gt.detach().numpy()

        # resize saliency map to fixation map size
        if not np.shape(pred) == np.shape(gt):
            pred = resize(pred, np.shape(gt))

        # normalize the saliency map
        pred = (pred - pred.min()) / (pred.max() - pred.min())

        # flatten the maps
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()

        fixations = pred_flat[gt_flat > 0]
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
            [y, x] = np.nonzero(gt)
            s = np.shape(pred)
            plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
            plt.plot(x, y, 'ro')

            ax = fig.add_subplot(1, 2, 2)
            plt.plot(fp, tp, '.b-')
            ax.set_title('Area under ROC curve: ' + str(score))
            plt.axis((0, 1, 0, 1))
            plt.show()

        return score

    def nss(self, pred, gt):
        """
        Calculates Normalized Scanpath Saliency measure (NSS).

        :param pred: predicted saliency map
        :type pred: torch.Tensor
        :param gt: ground truth fixation map
        :type gt: torch.Tensor
        :return: NSS measure value
        :rtype: torch.Tensor
        """
        # resize predicted saliency map if the sizes don't match
        if pred.size() != gt.size():
            pred = pred.cpu()
            pred = pred.squeeze(0).numpy()
            pred = torch.FloatTensor(resize(pred, (gt.size(2), gt.size(1)))).unsqueeze(0)

        pred = pred.cuda()
        gt = gt.cuda()

        batch_size = pred.size(0)
        w = pred.size(1)
        h = pred.size(2)

        pred_mean = torch.mean(pred.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
        pred_std = torch.std(pred.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

        pred = (pred - pred_mean) / pred_std
        pred = torch.sum((pred * gt).view(batch_size, -1), 1)
        count = torch.sum(gt.view(batch_size, -1), 1)
        return torch.mean(pred / count)

    def cc(self, pred, gt):
        """
        Calculates Correlation Coefficient measure (CC).

        :param pred: predicted saliency map
        :type pred: torch.Tensor
        :param gt: ground truth saliency map
        :type gt: torch.Tensor
        :return: NSS measure value
        :rtype: torch.Tensor
        """
        assert pred.size() == gt.size()
        batch_size = pred.size(0)
        w = pred.size(1)
        h = pred.size(2)

        pred_mean = torch.mean(pred.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
        pred_std = torch.std(pred.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

        mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
        std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

        pred = (pred - pred_mean) / pred_std
        gt = (gt - mean_gt) / std_gt

        ab = torch.sum((pred * gt).view(batch_size, -1), 1)
        aa = torch.sum((pred * pred).view(batch_size, -1), 1)
        bb = torch.sum((gt * gt).view(batch_size, -1), 1)

        return torch.mean(ab / (torch.sqrt(aa * bb)))
