import torchvision
import torch
import numpy as np
import os

from torch.utils.data import Dataset

DATA_MODES = ['train', 'val', 'test']


class AntispoofDataset(Dataset):

    def __init__(self, path_videos, loader=torchvision.datasets.folder.default_loader,
                 transform=None, mode='train'):
        super().__init__()

        self.path_videos = path_videos
        self.transform = transform
        self.loader = loader
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

    def __getitem__(self, index):
        video_info = self.path_videos[index]
        frames = os.listdir(video_info['path'])
        images = []

        for frame in frames:
            images.append(self.loader(os.path.join(video_info['path'], frame)))

            if self.transform is not None:
                images[-1] = self.transform(images[-1])

        if self.mode == 'train':
            return torch.Tensor([np.array(image) for image in images]), video_info['label']
        else:
            return torch.Tensor([np.array(image) for image in images])

    def __len__(self):
        return len(self.path_videos)