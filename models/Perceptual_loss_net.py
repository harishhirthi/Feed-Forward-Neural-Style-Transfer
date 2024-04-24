from collections import namedtuple
import torch
import torch.nn as nn
from torchvision.models import vgg16

"""Class to create feature maps from pre-trained vgg16 model."""
class Vgg16_pretrained(nn.Module):

    def __init__(self, requires_grad: bool = False, show_progress: bool = False):
        super().__init__()

        vgg16_pretrained = vgg16(pretrained = True, progress = show_progress).eval()
        vgg16_features = vgg16_pretrained.features
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'] # Layers used to get feature maps

        self.slice_1 = nn.Sequential()
        self.slice_2 = nn.Sequential()
        self.slice_3 = nn.Sequential()
        self.slice_4 = nn.Sequential()

        for i in range(4):
            self.slice_1.add_module(str(i), vgg16_features[i])
        for i in range(4, 9):
            self.slice_2.add_module(str(i), vgg16_features[i])
        for i in range(9, 16):
            self.slice_3.add_module(str(i), vgg16_features[i])
        for i in range(16, 23):
            self.slice_4.add_module(str(i), vgg16_features[i])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> namedtuple:
        out = self.slice_1(x)
        relu1_2 = out
        out = self.slice_2(out)
        relu2_2 = out
        out = self.slice_3(out)
        relu3_3 = out
        out = self.slice_4(out)
        relu4_3 = out

        vgg16_outputs = namedtuple("VGG16Outputs", self.layer_names)
        out = vgg16_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out
        

Perceptual_Loss_Net = Vgg16_pretrained