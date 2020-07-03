# Mostly borrowed from https://github.com/ZijunDeng/pytorch-semantic-segmentation
import torch.nn.functional as F
from torch import nn
from torchvision import models

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # nn.init.kaiming_normal(module.weight) # initialization used originally in Resnet
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

# Much borrowed from https://github.com/ycszen/pytorch-ss/blob/master/gcn.py
class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = int( (kernel_size[0] - 1) / 2 )
        pad1 = int( (kernel_size[1] - 1) / 2 )
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim,  out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim,  out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = x + residual
        return out


class GCN(nn.Module):
    def __init__(self, num_classes, num_levels=4):
        super(GCN, self).__init__()

        self.num_levels = num_levels

        # resnet = models.resnet152(pretrained=True)
        resnet = models.resnet101(pretrained=True)

        # Resnet-GCN not implemented, instead original Resnet layers are used
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        kernel_size = 7 # set this value according to the smallest resolution which depends upon the image size and the number of scales in the net
        self.gcm1 = _GlobalConvModule(2048, num_classes, (kernel_size, kernel_size))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (kernel_size, kernel_size))
        self.gcm3 = _GlobalConvModule(512,  num_classes, (kernel_size, kernel_size))
        self.gcm4 = _GlobalConvModule(256,  num_classes, (kernel_size, kernel_size))

        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)

        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4,
                           self.brm1, self.brm2, self.brm3, self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)

    def forward(self, x):

        if self.num_levels == 2:
                                    # if x: 512
            fm0 = self.layer0(x)    # 256
            fm1 = self.layer1(fm0)  # 128
            fm2 = self.layer2(fm1)  # 64

            gcfm1 = self.brm3(self.gcm3(fm2))  # 64
            gcfm2 = self.brm4(self.gcm4(fm1))  # 128

            fs1 = self.brm7(F.upsample_bilinear(gcfm1, fm1.size()[2:]) + gcfm2)  # 128
            fs2 = self.brm8(F.upsample_bilinear(fs1,   fm0.size()[2:]))          # 256
            out = self.brm9(F.upsample_bilinear(fs2,     x.size()[2:]))          # 512

            return out

        elif self.num_levels == 3:
                                    # if x: 512
            fm0 = self.layer0(x)    # 256
            fm1 = self.layer1(fm0)  # 128
            fm2 = self.layer2(fm1)  # 64
            fm3 = self.layer3(fm2)  # 32

            gcfm1 = self.brm2(self.gcm2(fm3))  # 32
            gcfm2 = self.brm3(self.gcm3(fm2))  # 64
            gcfm3 = self.brm4(self.gcm4(fm1))  # 128

            fs1 = self.brm6(F.upsample_bilinear(gcfm1, fm2.size()[2:]) + gcfm2)  # 64
            fs2 = self.brm7(F.upsample_bilinear(fs1,   fm1.size()[2:]) + gcfm3)  # 128
            fs3 = self.brm8(F.upsample_bilinear(fs2,   fm0.size()[2:]))          # 256
            out = self.brm9(F.upsample_bilinear(fs3,     x.size()[2:]))          # 512

            return out

        else:
                                    # if x: 512
            fm0 = self.layer0(x)    # 256
            fm1 = self.layer1(fm0)  # 128
            fm2 = self.layer2(fm1)  # 64
            fm3 = self.layer3(fm2)  # 32
            fm4 = self.layer4(fm3)  # 16

            gcfm1 = self.brm1(self.gcm1(fm4))  # 16
            gcfm2 = self.brm2(self.gcm2(fm3))  # 32
            gcfm3 = self.brm3(self.gcm3(fm2))  # 64
            gcfm4 = self.brm4(self.gcm4(fm1))  # 128

            fs1 = self.brm5(F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2)  # 32
            fs2 = self.brm6(F.upsample_bilinear(fs1,   fm2.size()[2:]) + gcfm3)  # 64
            fs3 = self.brm7(F.upsample_bilinear(fs2,   fm1.size()[2:]) + gcfm4)  # 128
            fs4 = self.brm8(F.upsample_bilinear(fs3,   fm0.size()[2:]))          # 256
            out = self.brm9(F.upsample_bilinear(fs4,     x.size()[2:]))          # 512

            return out

class ResnetFCN(nn.Module):

    def __init__(self, num_classes):
        super(ResnetFCN, self).__init__()

        # Load the model and change the last layer
        fcn = models.segmentation.fcn_resnet101(pretrained=True)

        conv_classifier = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        conv_auxiliar   = nn.Conv2d(256, num_classes, kernel_size=(1,1), stride=(1,1))

        nn.init.xavier_uniform_(conv_classifier.weight)
        if conv_classifier.bias is not None:
            conv_classifier.bias.data.zero_()

        nn.init.xavier_uniform_(conv_auxiliar.weight)
        if conv_auxiliar.bias is not None:
            conv_auxiliar.bias.data.zero_()

        fcn.classifier[4]     = conv_classifier
        fcn.aux_classifier[4] = conv_auxiliar

        self.fcn = fcn

    def forward(self, x):

        return self.fcn(x)['out']


class DeepLabV3(nn.Module):

    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()

        # Load the model and change the last layer
        net = models.segmentation.deeplabv3_resnet101(pretrained=True)

        conv_classifier = nn.Conv2d(256, num_classes, kernel_size=(1,1), stride=(1,1))
        conv_auxiliar   = nn.Conv2d(256, num_classes, kernel_size=(1,1), stride=(1,1))

        nn.init.xavier_uniform_(conv_classifier.weight)
        if conv_classifier.bias is not None:
            conv_classifier.bias.data.zero_()

        nn.init.xavier_uniform_(conv_auxiliar.weight)
        if conv_auxiliar.bias is not None:
            conv_auxiliar.bias.data.zero_()

        net.classifier[4]     = conv_classifier
        net.aux_classifier[4] = conv_auxiliar

        self.net = net

    def forward(self, x):

        return self.net(x)['out']

