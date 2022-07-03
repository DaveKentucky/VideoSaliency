import argparse

import ViNet.model
import TASED_Net.model

from torch.utils.data import DataLoader

from dataset import DHF1KDataset
from model import VideoSaliencyModel
from train import prepare_sample
from utils import *
from loss import VideoSaliencyLoss
import os


parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='Model that should be validated. (ViNet, TASED_Net, VideoSaliency)')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def evaluate():
    args = parser.parse_args()
    model_name = args.model
    if model_name == 'ViNet':
        len_temporal = 32
        model = ViNet.model.VideoSaliencyModel(
            transformer_in_channel=32,
            nhead=4,
            use_upsample=True,
            num_hier=3,
            num_clips=len_temporal)
        model.load_state_dict(torch.load('ViNet/ViNet_DHF1K.pt'))
    elif model_name == 'TASED_Net':
        file_weight = 'TASED_Net/TASED_updated.pt'
        len_temporal = 32
        model = TASED_Net.model.TASED_v2()

        # load the weight file and copy the parameters
        if os.path.isfile(file_weight):
            print('loading weight file')
            weight_dict = torch.load(file_weight)
            model_dict = model.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        print(' size? ' + name, param.size(), model_dict[name].size())
                else:
                    print(' name? ' + name)
            print(' loaded')
        else:
            print('weight file?')
    elif model_name == 'VideoSaliency':
        len_temporal = 8
        model = VideoSaliencyModel()
        model.load_state_dict(torch.load('VideoSaliency.pt'))
    else:
        print('Invalid model name.')
        return

    path_data = 'E:/szkolne/praca_magisterska/ACLNet-Pytorch/validation'

    criterion = VideoSaliencyLoss(mode='evaluate')
    validation_dataset = DHF1KDataset(path_data, len_temporal, mode='validate')
    loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    with torch.no_grad():
        model.eval()
        sim_sum, nss_sum, auc_sum, cc_sum = 0, 0, 0, 0
        num_samples = len(loader)

        for (idx, sample) in enumerate(loader):
            print(f' Processing sample {idx + 1}...')
            clips, gt, fixations = prepare_sample(sample, device, gt_to_device=False)

            prediction = model(clips)
            gt = gt.squeeze(0).numpy()
            prediction = prediction.cpu().squeeze(0).detach().numpy()
            prediction = cv.resize(prediction, (gt.shape[1], gt.shape[0]))
            prediction = blur(prediction).unsqueeze(0).cuda()
            gt = torch.FloatTensor(gt).unsqueeze(0).cuda()
            # print(prediction.size())
            # print(gt.size())
            assert prediction.size() == gt.size()

            sim, nss, auc, cc = criterion(prediction, gt, fixations)
            print(f'  SIM: {sim.item():.3f}, NSS: {nss:.3f}, AUC: {auc.item():.3f}, CC: {cc.item():.3f}')
            sim_sum += sim.item()
            nss_sum += nss.item()
            auc_sum += auc.item()
            cc_sum += cc.item()

        avg_sim = sim_sum / num_samples
        avg_nss = nss_sum / num_samples
        avg_auc = auc_sum / num_samples
        avg_cc = cc_sum / num_samples
        print(f'SIM: {avg_sim:.3f}\n'
              f'NSS: {avg_nss:.3f}\n'
              f'AUC: {avg_auc:.3f}\n'
              f'CC: {avg_cc:.3f}\n')


if __name__ == '__main__':
    evaluate()
