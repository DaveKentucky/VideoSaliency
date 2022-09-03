import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat

from utils import torch_transform_image


def resize_fixation(fixation, width=384, height=224):
    """
    Resizes fixation map.

    :param fixation: fixation map binary matrix (scipy.io.loadmat(mat_file)['I'])
    :type fixation: numpy.ndarray
    :param width: desired fixation map width
    :type width: int
    :param height: desired fixation map height
    :type height: int
    :return: resized fixation matrix
    :rtype: numpy.ndarray
    """
    out = np.zeros((height, width))
    height_sf = height / fixation.shape[0]  # height scale factor
    width_sf = width / fixation.shape[1]    # width scale factor

    coords = np.argwhere(fixation)
    for coord in coords:
        row = int(np.round(coord[0] * height_sf))
        col = int(np.round(coord[1] * width_sf))
        if row == height:
            row -= 1
        if col == width:
            col -= 1
        out[row, col] = 1

    return out


class DHF1KDataset(Dataset):
    def __init__(self, path, len_snippet, mode='train'):
        self.path = path
        self.len_snippet = len_snippet
        self.mode = mode
        if self.mode == 'train':
            self.video_names = os.listdir(path)
            self.list_num_frame = [len(os.listdir(os.path.join(path, d, 'images'))) for d in self.video_names]
        elif self.mode == 'validate':
            self.list_num_frame = []
            for v in os.listdir(path):
                for i in range(0, len(os.listdir(os.path.join(path, v, 'images'))) - self.len_snippet, 4 * self.len_snippet):
                    self.list_num_frame.append((v, i))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        if self.mode == 'train':
            file_name = self.video_names[idx]
            start_idx = np.random.randint(0, self.list_num_frame[idx] - self.len_snippet + 1)
        elif self.mode == 'validate':
            file_name, start_idx = self.list_num_frame[idx]

        path_clip = os.path.join(self.path, file_name, 'images')
        path_annotation = os.path.join(self.path, file_name, 'maps')
        path_fixation = os.path.join(self.path, file_name, 'fixation', 'maps')
        clip_img = []
        clip_annotation = []
        clip_fixation = []

        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, f'{(start_idx + i + 1):04}.png')).convert('RGB')

            annotation = np.array(Image.open(os.path.join(path_annotation, f'{(start_idx + i + 1):04}.png')).convert('L'))
            annotation = annotation.astype(float)
            annotation = cv.resize(annotation, (384, 224))
            if np.max(annotation) > 1.0:
                annotation = annotation / 255.0

            fixation = loadmat(os.path.join(path_fixation, f'{(start_idx + i + 1):04}.mat'))['I']
            fixation = resize_fixation(fixation)

            clip_img.append(torch_transform_image(img))
            clip_annotation.append(torch.FloatTensor(annotation))
            clip_fixation.append(torch.from_numpy(fixation.copy()))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        clip_annotation = torch.FloatTensor(torch.stack(clip_annotation, dim=0))

        # print(clip_img.shape)
        return clip_img, clip_annotation[-1], clip_fixation[-1]


class UCFDataset(Dataset):
    def __init__(self, path, len_snippet, mode='train'):
        self.path = path
        self.len_snippet = len_snippet
        self.mode = mode
        if self.mode == 'train':
            self.video_names = os.listdir(path)
            self.list_num_frame = [len(os.listdir(os.path.join(path, d, 'images'))) for d in self.video_names]
        elif self.mode == 'validate':
            self.list_num_frame = []
            for v in os.listdir(path):
                for i in range(0, len(os.listdir(os.path.join(path, v, 'images'))) - self.len_snippet, self.len_snippet):
                    self.list_num_frame.append((v, i))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        if self.mode == 'train':
            file_name = self.video_names[idx]
            start_idx = np.random.randint(0, self.list_num_frame[idx] - self.len_snippet + 1)
        elif self.mode == 'validate':
            file_name, start_idx = self.list_num_frame[idx]

        path_clip = os.path.join(self.path, file_name, 'images')
        path_annotation = os.path.join(self.path, file_name, 'maps')
        path_fixation = os.path.join(self.path, file_name, 'fixation', 'maps')
        clip_img = []
        clip_annotation = []
        clip_fixation = []

        list_clips = os.listdir(path_clip)
        list_clips.sort()
        list_sal_clips = os.listdir(path_annotation)
        list_sal_clips.sort()
        list_fix_clips = os.listdir(path_fixation)
        list_fix_clips.sort()

        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, list_clips[start_idx + i])).convert('RGB')

            annotation = np.array(Image.open(os.path.join(path_annotation, list_sal_clips[start_idx+i])).convert('L'))
            annotation = annotation.astype(float)
            annotation = cv.resize(annotation, (384, 224))
            if np.max(annotation) > 1.0:
                annotation = annotation / 255.0

            fixation = loadmat(os.path.join(path_fixation, list_fix_clips[start_idx + i]))['I']
            fixation = resize_fixation(fixation)

            clip_img.append(torch_transform_image(img))
            clip_annotation.append(torch.FloatTensor(annotation))
            clip_fixation.append(torch.from_numpy(fixation.copy()))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        clip_annotation = torch.FloatTensor(torch.stack(clip_annotation, dim=0))

        # print(clip_img.shape)
        return clip_img, clip_annotation[-1], clip_fixation[-1]


class GazeBaseDataset(Dataset):
    def __init__(self, path, len_snippet):
        self.path = path
        self.len_snippet = len_snippet
        self.list_num_frame = []
        for v in os.listdir(path):
            for i in range(0, len(os.listdir(os.path.join(path, v, 'images'))) - self.len_snippet, self.len_snippet):
                self.list_num_frame.append((v, i))

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        file_name, start_idx = self.list_num_frame[idx]

        path_clip = os.path.join(self.path, file_name, 'images')
        path_annotation = os.path.join(self.path, file_name, 'maps')
        path_fixation = os.path.join(self.path, file_name, 'fixations')
        clip_img = []
        clip_annotation = []
        clip_fixation = []

        list_clips = os.listdir(path_clip)[100:]
        list_clips.sort()
        list_sal_clips = os.listdir(path_annotation)[100:]
        list_sal_clips.sort()
        list_fix_clips = os.listdir(path_fixation)[100:]
        list_fix_clips.sort()

        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, list_clips[start_idx + i])).convert('RGB')

            annotation = np.array(Image.open(os.path.join(path_annotation, list_sal_clips[start_idx+i])).convert('L'))
            annotation = annotation.astype(float)
            annotation = cv.resize(annotation, (384, 224))
            if np.max(annotation) > 1.0:
                annotation = annotation / 255.0

            fixation = np.array(Image.open(os.path.join(path_fixation, list_fix_clips[start_idx+i])).convert('L'))
            fixation = fixation.astype(float)
            fixation = cv.resize(fixation, (384, 224))
            if np.max(fixation) > 1.0:
                fixation = fixation / 255.0

            clip_img.append(torch_transform_image(img))
            clip_annotation.append(torch.FloatTensor(annotation))
            clip_fixation.append(torch.from_numpy(fixation.copy()))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        clip_annotation = torch.FloatTensor(torch.stack(clip_annotation, dim=0))

        # print(clip_img.shape)
        return clip_img, clip_annotation[-1], clip_fixation[-1]