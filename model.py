import torch.nn as nn
import torch
from torchsummary import summary

__all__ = ['Cfcn']


class Conv1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Conv3x3(nn.Module):
    def __init__(self, _in, _out, stride=1, dilation=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(_in, _out, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Deconv3x3(nn.Module):
    def __init__(self, _in, _out, stride=2, dilation=1):
        super(Deconv3x3, self).__init__()
        self.conv = nn.ConvTranspose2d(_in, _out, kernel_size=3, stride=stride, padding=dilation)
        self.bn = nn.BatchNorm2d(_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CSP_layer(nn.Module):
    def __init__(self, in_channel, out_channel, block, blocks, stride):
        super(CSP_layer, self).__init__()
        self.channels = in_channel
        self.layer = self.make_layer(block, out_channel, blocks)
        self.conv1 = Conv1x1(out_channel, out_channel)
        self.conv2 = Conv1x1(out_channel*2, out_channel)
        self.downsample = Conv3x3(in_channel, out_channel, stride=stride)

    def make_layer(self, block, channel, blocks):
        layers = []
        downsample = None

        front = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        back = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        layers.append(front)

        for _ in range(blocks):
            layers.append(block(channel, channel))

        layers.append(back)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.downsample(x)
        identity = x

        x = self.layer(x)

        identity = self.conv1(identity)
        x = torch.cat([x, identity], dim=1)
        out = self.conv2(x)

        return out


class Cfcn(nn.Module):
    def __init__(self, block, layers, class_num):
        super(Cfcn, self).__init__()
        self.channels = 32

        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_layer1 = self.make_layer(block, 32, layers[0])
        self.conv_layer2 = self.make_layer(block, 64, layers[1], stride=2)
        self.conv_layer3 = self.make_layer(block, 128, layers[2], stride=2)
        self.conv_layer4 = self.make_layer(block, 256, layers[3], stride=2)

        self.dilconv1_1 = Conv3x3(256, 256, dilation=1)
        self.dilconv1_2 = Conv3x3(256, 256, dilation=2)
        self.dilconv1_5 = Conv3x3(256, 256, dilation=4)
        self.dilconv2_1 = Conv3x3(256, 256, dilation=1)
        self.dilconv2_2 = Conv3x3(256, 256, dilation=2)
        self.dilconv3_1 = Conv3x3(256, 256, dilation=1)

        self.up_conv4 = Conv1x1(256, 256)
        self.up_deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.up_conv3 = Conv1x1(128, 128)
        self.up_deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.up_conv2 = Conv1x1(64, 64)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, class_num, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)

    def make_layer(self, block, channel, blocks, stride=1):
        layers = [CSP_layer(self.channels, channel, block, blocks, stride)]
        self.channels = channel
        return nn.Sequential(*layers)

    def forward(self, _x):
        x = self.conv1(_x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv_layer1(x)
        x1 = x

        x = self.conv_layer2(x)
        x2 = self.up_conv2(x)

        x = self.conv_layer3(x)
        x3 = self.up_conv3(x)

        x = self.conv_layer4(x)

        d1x = self.dilconv1_1(x)
        d1x = self.dilconv1_2(d1x)
        d1x = self.dilconv1_5(d1x)
        d2x = self.dilconv2_1(x)
        d2x = self.dilconv2_2(d2x)
        d3x = self.dilconv3_1(x)
        x4 = d1x + d2x + d3x + x
        x4 = self.up_conv4(x4)

        up_x = self.up_deconv3(x4, output_size=x3.size())
        up_x = self.relu(up_x)
        x3 = x3 + up_x
        up_x = self.up_deconv2(x3, output_size=x2.size())
        up_x = self.relu(up_x)
        x2 = x2 + up_x

        x = self.deconv3(x, output_size=x3.size())
        x = self.relu(x)
        x = torch.cat([x, x3], dim=1)

        x = self.deconv2(x, output_size=x2.size())
        x = self.relu(x)
        x = torch.cat([x, x2], dim=1)
        x = self.relu(x)

        x = self.deconv1(x, output_size=x1.size())
        x = self.relu(x)

        out = self.conv2(x)

        return out


def cfcn(num_class):
    return Cfcn(BasicBlock, [9, 9, 9, 9], num_class)


if __name__ == '__main__':
    model = cfcn(2)
    print(summary(model, (3, 256, 256), batch_size=32, device='cpu'))
