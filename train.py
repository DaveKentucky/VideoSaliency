import argparse
import os
import time
from datetime import timedelta
from itertools import islice

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import DHF1KDataset
from loss import VideoSaliencyLoss
from model import VideoSaliencyModel

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path',
                    default='E:/szkolne/praca_magisterska/ACLNet-Pytorch/train',
                    type=str,
                    help='path to training data')
parser.add_argument('--output_path', default='./result', type=str, help='path for output files')


def main():
    args = parser.parse_args()

    # set constants
    file_weight = './S3D_kinetics400.pt'
    len_temporal = 8
    batch_size = 4
    epochs = 20

    # set input and output path strings
    path_input = args.train_data_path
    path_output = args.output_path
    path_output = os.path.join(path_output, time.strftime("%m-%d_%H-%M-%S"))
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    model = VideoSaliencyModel()

    # load dataset
    train_dataset = DHF1KDataset(path_input, len_temporal)

    # load the weight file for encoder network
    if os.path.isfile(file_weight):
        print('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.encoder.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if 'base.' in name:
                bn = int(name.split('.')[1])
                sn_list = [0, 5, 8, 14]
                sn = sn_list[0]
                if sn_list[1] <= bn < sn_list[2]:
                    sn = sn_list[1]
                elif sn_list[2] <= bn < sn_list[3]:
                    sn = sn_list[2]
                elif bn >= sn_list[3]:
                    sn = sn_list[3]
                name = '.'.join(name.split('.')[2:])
                name = 'base%d.%d.' % (sn_list.index(sn) + 1, bn - sn) + name
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print(' size? ' + name, param.size(), model_dict[name].size())
            else:
                print(' name? ' + name)

        print(' loaded')
        model.encoder.load_state_dict(model_dict)
    else:
        print('weight file?')

    # load model to GPU
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print('CUDA is not available on your device or no compatible GPUs were found.')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    print('Successfully loaded the model to GPU!')

    # set parameters for training
    params = []
    for key, value in dict(model.named_parameters()).items():
        if 'conv' in key:
            params += [{'params': [value], 'key':key+'(new)'}]
        else:
            params += [{'params': [value], 'lr':0.001, 'key':key}]

    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=2e-7)
    criterion = VideoSaliencyLoss()

    # train the model
    model.train()
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()

    avg_loss, avg_auc, avg_sim, avg_nss = 0, 0, 0, 0

    for i in range(epochs):
        for (idx, sample) in enumerate(loader):
            clips = sample[0]
            annotations = sample[1]
            fixations = sample[2]
            clips = clips.to(device)
            clips = clips.permute((0, 2, 1, 3, 4))
            annotations = annotations.to(device)
            optimizer.zero_grad()

            prediction = model(clips)
            # print(prediction.size())
            # print(annotations.size())
            assert prediction.size() == annotations.size()

            loss, loss_auc, loss_sim, loss_nss = criterion(prediction, annotations, fixations)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_auc += loss_auc.item()
            avg_sim += loss_sim.item()
            avg_nss += loss_nss.item()

        print(f'epoch: {i + 1}\n'
              f'loss: {(avg_loss / len(loader)):.3f}\n'
              f'SIM: {(avg_sim / len(loader)):.3f}\n'
              f'AUC: {(avg_auc / len(loader)):.3f}\n'
              f'NSS: {(avg_nss / len(loader)):.3f}\n'
              f'total time: {((time.time() - start_time) / 60):.2f} minutes')
        avg_loss = 0

        weights_file = f'model_weights{(i + 1):03}.pt'
        torch.save(model.state_dict(), os.path.join('weights', weights_file))


    # i, step = 0, 0
    # loss_sum = 0
    # start_time = time.time()
    #
    # for clip, annotation in islice(loader, epochs):
    #     with torch.set_grad_enabled(True):
    #         output = model(clip.cuda())
    #         print(f'output: {output.shape}, gt: {annotation.shape}')
    #         loss = criterion(output, annotation.cuda())
    #
    #     loss_sum += loss.detach().item()
    #     loss.backward()
    #     optimizer.step()
    #     # optimizer.zero_grad()
    #
    #     if (i + 1) % 10 == 0:
    #         step += 1
    #         print(f'iteration: [{step}/{epochs}], loss: {loss_sum / 10}, '
    #               f'time: {timedelta(seconds=int(time.time() - start_time))}', flush=True)
    #         loss_sum = 0
    #
    #         if step % 10 == 0:
    #             torch.save(model.state_dict(), os.path.join(path_output, f'iter{step:03}.pt'))
    #
    #     i += 1


if __name__ == '__main__':
    main()
