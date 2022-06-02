import argparse
import os
import time
from datetime import timedelta
from itertools import islice

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import DHF1KDataset
from model import VideoSaliencyModel
from utils import KLDivLoss

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
    len_temporal = 32
    batch_size = 8
    num_iterations = 200

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
        return
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()
    print('Successfully loaded the model to GPU!')

    # set parameters for training
    params = []
    for key, value in dict(model.named_parameters()).items():
        if 'conv' in key:
            params += [{'params': [value], 'key':key+'(new)'}]
        else:
            params += [{'params': [value], 'lr':0.001, 'key':key}]

    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=2e-7)
    criterion = KLDivLoss()

    # train the model
    model.train()
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    i, step = 0, 0
    loss_sum = 0
    start_time = time.time()

    for clip, annotation in islice(loader, num_iterations):
        with torch.set_grad_enabled(True):
            output = model(clip.cuda())
            loss = criterion(output, annotation.cuda())

        loss_sum += loss.detach().item()
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()

        if (i + 1) % 10 == 0:
            step += 1
            print(f'iteration: [{step}/{num_iterations}], loss: {loss_sum / 10}, '
                  f'time: {timedelta(seconds=int(time.time() - start_time))}', flush=True)
            loss_sum = 0

            if step % 10 == 0:
                torch.save(model.state_dict(), os.path.join(path_output, f'iter{step:03}.pt'))

        i += 1


if __name__ == '__main__':
    main()
