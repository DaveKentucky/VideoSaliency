import argparse

import ViNet_model
import TASED_Net.model

from torch.utils.data import DataLoader

from dataset import DHF1KDataset, UCFDataset, GazeBaseDataset
from model import VideoSaliencyModel
from train import prepare_sample
from utils import *
from loss import VideoSaliencyLoss
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='Model that should be validated. (ViNet, TASED_Net, VS_NSS, VS_SIM)')
parser.add_argument('dataset', type=str, help='Dataset used for validation. (DHF1K, UCF, GazeBase)')
parser.add_argument('--criterion', type=str, default='metrics', help='Criterion which should be evaluated. (metrics, time)')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def evaluate():
    args = parser.parse_args()

    model_name = args.model
    if model_name == 'ViNet':
        len_temporal = 32
        model = ViNet_model.VideoSaliencyModel(
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
    elif model_name == 'VS_NSS':
        len_temporal = 8
        model = VideoSaliencyModel()
        model.load_state_dict(torch.load('weights/model_weights020.pt'))
    elif model_name == 'VS_SIM':
        len_temporal = 8
        model = VideoSaliencyModel()
        model.load_state_dict(torch.load('weights/sim_weights019.pt'))
    else:
        print('Invalid model name.')
        return

    path_data_DHF1K = 'E:/szkolne/praca_magisterska/ACLNet-Pytorch/validation'
    path_data_UCF = 'E:/szkolne/praca_magisterska/DHF1K_dataset/testing'
    path_data_GazeBase = 'E:/Szkolne/praca_magisterska/GazeBase/Frames'

    eval_criterion = args.criterion
    if eval_criterion == 'metrics':
        criterion = VideoSaliencyLoss(mode='evaluate')

    dataset_name = args.dataset
    if dataset_name == 'DHF1K':
        validation_dataset = DHF1KDataset(path_data_DHF1K, len_temporal, mode='validate')
    elif dataset_name == 'UCF':
        validation_dataset = UCFDataset(path_data_UCF, len_temporal, mode='validate')
    elif dataset_name == 'GazeBase':
        validation_dataset = GazeBaseDataset(path_data_GazeBase, len_temporal)
    else:
        print('Invalid dataset name.')
        return
    loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    with torch.no_grad():
        model.eval()
        sim_sum, nss_sum, auc_sum, cc_sum = 0, 0, 0, 0
        num_samples = len(loader)
        total_time, total_frames = 0, 0

        for (idx, sample) in enumerate(loader):
            start_time = time.time()
            print(f' Processing sample {idx + 1}...')
            clips, gt, fixations = prepare_sample(sample, device, gt_to_device=False)

            prediction = model(clips)
            if eval_criterion == 'metrics':
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
            elif eval_criterion == 'time':
                total_time += (time.time() - start_time)
                total_frames += len_temporal
                print(f'  time: {time.time() - start_time}')

        if eval_criterion == 'metrics':
            avg_sim = sim_sum / num_samples
            avg_nss = nss_sum / num_samples
            avg_auc = auc_sum / num_samples
            avg_cc = cc_sum / num_samples
            print(f'SIM: {avg_sim:.3f}\n'
                  f'NSS: {avg_nss:.3f}\n'
                  f'AUC: {avg_auc:.3f}\n'
                  f'CC: {avg_cc:.3f}\n')
        elif eval_criterion == 'time':
            print(f'total time: {total_time / 60} min\n'
                  f'FPS: {total_frames / total_time}')


if __name__ == '__main__':
    evaluate()
