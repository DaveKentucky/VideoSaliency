import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat


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
            r -= 1
        if col == width:
            c -= 1
        out[row, col] = 1

    return out


class DHF1KDataset(Dataset):
    def __init__(self, path, len_snippet):
        self.path = path
        self.len_snippet = len_snippet
        self.video_names = os.listdir(path)
        self.list_num_frame = [len(os.listdir(os.path.join(path, d, 'images'))) for d in self.video_names]
        self.transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        file_name = self.video_names[idx]
        start_idx = np.random.randint(0, self.list_num_frame[idx] - self.len_snippet + 1)

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

            clip_img.append(self.transform(img))
            clip_annotation.append(torch.FloatTensor(annotation))
            clip_fixation.append(torch.from_numpy(fixation.copy()))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
        clip_annotation = torch.FloatTensor(torch.stack(clip_annotation, dim=0))

        # print(clip_img.shape)
        return clip_img, clip_annotation[-1], clip_fixation[-1]
