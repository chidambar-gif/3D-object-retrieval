import torch
from pathlib import Path
from torch.utils.data import Dataset
from .image import load_img
from loaders import load_pt



class OSMN40_train(Dataset):
    def __init__(self, phase, object_list):
        super().__init__()
        assert phase in ('train', 'val')
        self.phase = phase
        self.object_list = object_list

    def __getitem__(self, index):
        p = Path(self.object_list[index]['path'])
        lbl = self.object_list[index]['label']
        # # image
        img = load_img(p/'image', self.phase=='train')
        # point cloud
        pt = load_pt(p/'pointcloud', self.phase=='train')

        return img, pt, lbl

    def __len__(self):
        return len(self.object_list)

class OSMN40_retrive(Dataset):
    def __init__(self, object_list):
        super().__init__()
        self.object_list = object_list

    def __getitem__(self, index):
        p = Path(self.object_list[index])
        # # image
        img = load_img(p/'image')
        # point cloud
        pt = load_pt(p/'pointcloud')

        return img, pt

    def __len__(self):
        return len(self.object_list)


if __name__ == '__main__':
    pass

