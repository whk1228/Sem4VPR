import torch
import torch.nn as nn
from models.semantic_masker.DeepLabV3Plus import network



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.register_buffer()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.relu1 = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class BasicBlockSem(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(BasicBlockSem, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_planes)  #通道注意力模块
        self.sa = SpatialAttention() # 空间注意力模块

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        # Channel Attention Module
        # out = self.ca(out) * out  # 通道注意力
        out = self.sa(out) * out  # 空间注意力

        out = self.relu(out)

        return out
class SemNetAtt(nn.Module):
    def __init__(self,num_classes=19, output_stride=16):
        super(SemNetAtt, self).__init__()
        self.model = network.modeling.__dict__["deeplabv3plus_resnet101"](num_classes, output_stride)
        checkpoint = torch.load("/root/autodl-tmp/SemVG/model/semantic_masker"
                                "/DeepLabV3Plus/mlwu_utils/pretrained_weight/best"
                                "_deeplabv3plus_resnet101_cityscapes_os16.pth")
        self.model.load_state_dict(checkpoint["model_state"])

    def forward(self, x):
        with torch.no_grad():
             res = self.model(x) #(B, H, W)

        return res
