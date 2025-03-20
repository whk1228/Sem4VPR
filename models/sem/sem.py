import torch
import torch.nn as nn
from models.semantic_masker.DeepLabV3Plus import network


class SemNet(nn.Module):
    def __init__(self,num_classes=19, output_stride=16):
        super(SemNet, self).__init__()
        self.model = network.modeling.__dict__["deeplabv3plus_resnet101"](num_classes, output_stride)
        checkpoint = torch.load("/root/autodl-tmp/SemVG/model/semantic_masker"
                            "/DeepLabV3Plus/mlwu_utils/pretrained_weight/best"
                            "_deeplabv3plus_resnet101_cityscapes_os16.pth")
        self.model.load_state_dict(checkpoint["model_state"])



    def forward(self, x):
        with torch.no_grad():
            sem = self.model(x)
        return sem
