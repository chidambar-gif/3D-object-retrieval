import torch
import torch.nn as nn
from .image import MVCNN
from .dgcnn import DGCNN
import torch.nn.functional as F

class UniModel(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model_img = MVCNN(n_class, n_view=12)
        self.model_pt = DGCNN(n_class)

        self.linear1 = nn.Linear(2048+512, 1024, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4= nn.Linear(256,n_class)

    def forward(self, data, global_ft=False):

        img, pt  = data
        ft_pt=self.model_pt(pt)
        ft_img=self.model_img(img)

        total_img_pt_ft = torch.cat((ft_img,ft_pt),1)

        x = F.leaky_relu(self.bn6(self.linear1(total_img_pt_ft)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = F.leaky_relu(self.bn3(self.linear3(x)),negative_slope=0.2)
        x = self.linear4(x)

        if global_ft:
            return (x),(total_img_pt_ft)
        else:
            return x

