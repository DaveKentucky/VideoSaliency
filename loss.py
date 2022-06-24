import torch
from torch import nn
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
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)) * 1.0)
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
        :return: loss metrics values (loss, AUC-Judd, SIM, NSS)
        :rtype: (torch.Tensor, float, float, float)
        """
        loss_auc = self.auc_Judd(pred, fix)
        loss_sim = self.similarity(pred, gt)
        loss_nss = self.nss(pred, fix)
        loss = torch.FloatTensor([0.0]).cpu()
        loss += loss_auc + loss_sim + loss_nss
        return loss, loss_auc, loss_sim, loss_nss

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
        pred = pred.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        pred = normalize_numpy(pred)
        gt = normalize_numpy(gt)

        pred = pred / (np.sum(pred) * 1.0)
        gt = gt / (np.sum(gt) * 1.0)

        if len(pred.shape) == 3:
            sim_arr = np.empty(pred.shape[0])
            for i in range(gt.shape[0]):
                i_gt = gt[i, :, :].copy()
                i_pred = pred[i, :, :].copy()
                sim_arr[i] = self.get_sim(i_pred, i_gt)

            return np.mean(sim_arr)

        return self.get_sim(pred, gt)

    def get_sim(self, pred, gt):
        x, y = np.nonzero(gt > 0)
        sim = 0.0
        for i in zip(x, y):
            sim += min(gt[i[0], i[1]], pred[i[0], i[1]])

        return sim

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
        pred = pred.cpu()
        fix = fix.cpu()

        # resize predicted saliency map if the sizes don't match
        if pred.size() != fix.size():
            pred = pred.squeeze(0).numpy()
            pred = torch.FloatTensor(resize(pred, (fix.size(2), fix.size(1)))).unsqueeze(0)

        # get a single frame from video clip data
        if len(pred.size()) == 3:
            pred = pred[0, :, :]
            fix = fix[0, :, :]

        pred = pred.detach().numpy()
        fix = fix.detach().numpy()

        # resize saliency map to fixation map size
        if not np.shape(pred) == np.shape(fix):
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

    def nss(self, pred, fix):
        """
        Calculates Normalized Scanpath Saliency measure (NSS).

        :param pred: predicted saliency map
        :type pred: torch.Tensor
        :param fix: ground truth fixation map
        :type fix: torch.Tensor
        :return: NSS measure value
        :rtype: float
        """
        # resize predicted saliency map if the sizes don't match
        if pred.size() != fix.size():
            pred = pred.cpu()
            pred = pred.squeeze(0).numpy()
            pred = torch.FloatTensor(resize(pred, (fix.size(2), fix.size(1)))).unsqueeze(0)
            pred = pred.cuda()
            gt = fix.cuda()

        pred = pred.cpu().detach().numpy()
        fix = fix.cpu().detach().numpy()
        # gt = gt / 255

        if len(pred.shape) == 3:
            nss_arr = np.empty(pred.shape[0])
            for i in range(fix.shape[0]):
                i_pred = pred[i, :, :]
                i_fix = fix[i, :, :]
                nss_arr[i] = self.get_nss(i_pred, i_fix)

            return np.mean(nss_arr)

        return self.get_nss(pred, fix)

    def get_nss(self, pred, fix):
        x, y = np.nonzero(fix)
        pred_norm = (pred - np.mean(pred)) / np.std(pred)
        tmp = np.empty(x.shape[0])
        for i, val in enumerate(zip(x, y)):
            tmp[i] = pred_norm[val[0], val[1]]

        return np.mean(tmp)
