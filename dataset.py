import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


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
        clip_img = []
        clip_annotation = []

        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, f'{(start_idx + i + 1):04}.png')).convert('RGB')

            annotation = np.array(Image.open(os.path.join(path_annotation, f'{(start_idx + i + 1):04}.png')).convert('L'))
            annotation = annotation.astype(float)
            annotation = cv.resize(annotation, (384, 224))
            if np.max(annotation) > 1.0:
                annotation = annotation / 255.0

            clip_annotation.append(torch.FloatTensor(annotation))
            clip_img.append(self.transform(img))

            clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
            clip_annotation = torch.FloatTensor(torch.stack(clip_annotation, dim=0))
            return clip_img, clip_annotation
