import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import DHF1KDataset
from loss import VideoSaliencyLoss
from model import VideoSaliencyModel
from utils import load_model_to_device, blur
import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path',
                    default='E:/szkolne/praca_magisterska/ACLNet-Pytorch/train',
                    type=str,
                    help='path to training data')
parser.add_argument('--validation_data_path',
                    default='E:/szkolne/praca_magisterska/ACLNet-Pytorch/validation',
                    type=str,
                    help='path to validation data')
parser.add_argument('--output_path', default='result', type=str, help='path for output files')
parser.add_argument('--S3D_weights_file', default='S3D_kinetics400.pt', type=str, help='path to S3D network weights file')
parser.add_argument('--model_weights_file', default='', type=str, help='path to full model weights file')
parser.add_argument('--loss_file', default='loss.npy', type=str, help='path to numpy file with loss values')


def main():
    args = parser.parse_args()

    # set constants
    len_temporal = 8    # number of frames in operated clip
    batch_size = 3      # number of samples operated by the model at once
    epochs = 20

    # set input and output path strings
    path_train = args.train_data_path
    path_validate = args.validation_data_path
    path_output = args.output_path
    path_loss = args.loss_file
    # path_output = os.path.join(path_output, time.strftime("%m-%d_%H-%M-%S"))
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    model = VideoSaliencyModel()

    # load dataset
    train_dataset = DHF1KDataset(path_train, len_temporal)
    validation_dataset = DHF1KDataset(path_validate, len_temporal, mode='validate')

    # load the weight file for encoder network
    file_weight = args.S3D_weights_file
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
    file_weight_check = args.model_weights_file
    if not file_weight_check == '':
        print(f'\nLoading decoder network weights from {file_weight_check}...')
        model.load_state_dict(torch.load(file_weight_check))
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

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    best_loss = get_best_loss(path_loss)
    for i in range(epochs):
        # train the model
        loss_train = train(model, train_loader, optimizer, criterion, i + 1, device)

        # validate the model
        loss_val = validate(model, val_loader, criterion, i + 1, device)

        if loss_val[0] <= best_loss:
            best_loss = loss_val[0]
            # save the model
            weights_file = f'model_weights{(1 + (i if file_weight_check == "" else i + int(file_weight_check.split(".")[-2][-3:]))):03}.pt'
            torch.save(model.state_dict(), os.path.join('weights', weights_file))

        save_loss(np.asarray(loss_val), path_loss)


def train(model, loader, optimizer, criterion, epoch, device):
    print(f'\nStarting training model at epoch {epoch}\n')
    model.train()
    start_time = time.time()
    loss_sum, sim_sum = 0, 0
    num_samples = len(loader)

    for (idx, sample) in enumerate(loader):
        print(f' TRAIN: Processing sample {idx + 1}...')
        clips, gt, fixations = prepare_sample(sample, device, gt_to_device=True)
        optimizer.zero_grad()

        prediction = model(clips)
        # print(prediction.size())
        # print(gt.size())
        assert prediction.size() == gt.size()

        loss, loss_sim = criterion(prediction, gt, fixations)
        loss.backward()
        optimizer.step()
        print(f'   loss: {loss.item():.3f}, SIM: {loss_sim:.3f}')
        loss_sum += loss.item()
        sim_sum += loss_sim.item()

    avg_loss = loss_sum / num_samples
    avg_sim = sim_sum / num_samples
    print(f'\nepoch: {epoch}\n'
          f'loss: {avg_loss:.3f}\n'
          f'SIM: {avg_sim:.3f}\n'
          f'training time: {((time.time() - start_time) / 60):.2f} minutes')
    return avg_loss, avg_sim


def validate(model, loader, criterion, epoch, device):
    print(f'\nStarting validating model at epoch {epoch}')
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        loss_sum, sim_sum = 0, 0
        num_samples = len(loader)

        for (idx, sample) in enumerate(loader):
            print(f' VAL: Processing sample {idx + 1}...')
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

            loss, loss_sim = criterion(prediction, gt, fixations)
            print(f'   loss: {loss.item():.3f}, SIM: {loss_sim:.3f}')
            loss_sum += loss.item()
            sim_sum += loss_sim.item()

        avg_loss = loss_sum / num_samples
        avg_sim = sim_sum / num_samples
        print(f'\nepoch: {epoch}\n'
              f'loss: {avg_loss:.3f}\n'
              f'SIM: {avg_sim:.3f}\n'
              f'validation time: {((time.time() - start_time) / 60):.2f} minutes')
        return avg_loss, avg_sim


def prepare_sample(sample, device, gt_to_device):
    clips = sample[0]
    gt = sample[1]
    fixations = sample[2]
    clips = clips.to(device)
    clips = clips.permute((0, 2, 1, 3, 4))
    if gt_to_device:
        gt.to(device)
    return clips, gt, fixations


def save_loss(loss_arr, filename):
    size = loss_arr.shape[0]
    loss_arr = np.reshape(loss_arr, (1, size))
    if os.path.isfile(filename):
        arr = np.load(filename)
        loss_arr = np.concatenate((arr, loss_arr), axis=0)

    np.save(filename, loss_arr)


def get_best_loss(filename):
    if os.path.isfile(filename):
        arr = np.load(filename)
        best_loss = np.min(arr[:, 0])
    else:
        best_loss = np.inf
    return best_loss


if __name__ == '__main__':
    # import cv2 as cv
    # import numpy as np
    # train_dataset = DHF1KDataset('E:/szkolne/praca_magisterska/ACLNet-Pytorch/train', 1)
    # loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # for sample in loader:
    #     clip = sample[0]
    #     annt = sample[1]
    #     clip = clip.permute(0, 1, 3, 4, 2)
    #     clip = clip[0, 0, :, :, :].cpu().detach().numpy()
    #     annt = annt.permute(1, 2, 0)
    #     annt = annt.cpu().detach().numpy()
    #     annt = cv.cvtColor(annt, cv.COLOR_GRAY2BGR)
    #     images = np.concatenate((clip, annt), axis=1)
    #     cv.imshow("image", images)
    #     cv.waitKey(0)
    main()
