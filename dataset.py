import os
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms


class DHF1KDataset(Dataset):
    def __init__(self, path, len_snippet):
        self.path = path
        self.len_snippet = len_snippet
        self.video_names = os.listdir(path)
        self.list_num_frame = [len(os.listdir(os.path.join(path, d, 'images'))) for d in self.video_names]
        self.transform = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )

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
            img = cv.imread(os.path.join(path_clip, '%04d.png' % (start_idx + i + 1)))
            img = cv.resize(img, (384, 224))

            annotation = cv.imread(os.path.join(path_annotation, '%04d.png' % (start_idx + i + 1)), 0)
            annotation = annotation.astype(float)
            annotation = cv.resize(annotation, (384, 224))
            if np.max(annotation) > 1.0:
                annotation = annotation / 255.0

            clip_annotation.append(torch.FloatTensor(annotation))
            clip_img.append(self.transform(img))

            clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
            clip_annotation = torch.FloatTensor(torch.stack(clip_annotation, dim=0))
            return clip_img, clip_annotation
