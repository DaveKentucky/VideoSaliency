import argparse
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import DHF1KDataset
from loss import VideoSaliencyLoss
from model import VideoSaliencyModel
from utils import load_model_to_device

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path',
                    default='E:/szkolne/praca_magisterska/ACLNet-Pytorch/train',
                    type=str,
                    help='path to training data')
parser.add_argument('--output_path', default='./result', type=str, help='path for output files')
parser.add_argument('--load_weight_file', default='', type=str, help='path to pretrained model state dict file')


def main():
    args = parser.parse_args()

    # set constants
    file_weight = './S3D_kinetics400.pt'
    len_temporal = 8
    batch_size = 3
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
    if not os.path.isfile(file_weight):
        print('Invalid weight file for encoder network.')

    print(f'Loading encoder network weights from {file_weight}...')
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

    model.encoder.load_state_dict(model_dict)
    print(' Encoder network weights loaded!')

    # load the weight file for decoder network
    if not args.load_weight_file == '':
        print(f'\nLoading decoder network weights from {args.load_weight_file}...')
        model.load_state_dict(torch.load(args.load_weight_file))
        print(' Decoder network weights loaded!')

    # load model to GPU
    model, device = load_model_to_device(model)

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
    print('\nStarting training...')
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
            # loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_auc += loss_auc.item()
            avg_sim += loss_sim.item()
            avg_nss += loss_nss.item()

        print(f'\nepoch: {i + 1}\n'
              f'loss: {(avg_loss / len(loader)):.3f}\n'
              f'SIM: {(avg_sim / len(loader)):.3f}\n'
              f'AUC: {(avg_auc / len(loader)):.3f}\n'
              f'NSS: {(avg_nss / len(loader)):.3f}\n'
              f'total time: {((time.time() - start_time) / 60):.2f} minutes')
        avg_loss = 0

        weights_file = f'model_weights{(i + 1):03}.pt'
        torch.save(model.state_dict(), os.path.join('weights', weights_file))


if __name__ == '__main__':
    main()
