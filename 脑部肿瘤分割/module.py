from torch import nn
from torchvision.models import resnext50_32x4d
from my_dataset import device
import torch

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()

        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convrelu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)

        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                         stride=2, padding=1, output_padding=0)

        self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)

        return x


class ResNeXtUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # 加载预训练的ResNeXt50模型
        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        filters = [256, 512, 1024, 2048]  # 对应各编码器阶段的输出通道数

        # 编码器部分（下采样）
        self.encoder0 = nn.Sequential(*self.base_layers[:3])  # 初始卷积+BN+ReLU
        self.encoder1 = nn.Sequential(*self.base_layers[4])  # 第1个残差块组
        self.encoder2 = nn.Sequential(*self.base_layers[5])  # 第2个残差块组
        self.encoder3 = nn.Sequential(*self.base_layers[6])  # 第3个残差块组
        self.encoder4 = nn.Sequential(*self.base_layers[7])  # 第4个残差块组

        # 解码器部分（上采样）
        self.decoder4 = DecoderBlock(filters[3], filters[2])  # 对应encoder4的上采样
        self.decoder3 = DecoderBlock(filters[2], filters[1])  # 对应encoder3的上采样
        self.decoder2 = DecoderBlock(filters[1], filters[0])  # 对应encoder2的上采样
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 对应encoder1的上采样

        # 最终分类器
        self.last_conv0 = ConvRelu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)

    def forward(self, x):
        # 下采样路径（编码器）
        x = self.encoder0(x)  # 初始特征提取
        e1 = self.encoder1(x)  # 第1个下采样阶段
        e2 = self.encoder2(e1)  # 第2个下采样阶段
        e3 = self.encoder3(e2)  # 第3个下采样阶段
        e4 = self.encoder4(e3)  # 第4个下采样阶段，特征图尺寸最小

        # 上采样路径（解码器）+ 跳跃连接
        d4 = self.decoder4(e4) + e3  # 上采样并融合encoder3的特征
        d3 = self.decoder3(d4) + e2  # 上采样并融合encoder2的特征
        d2 = self.decoder2(d3) + e1  # 上采样并融合encoder1的特征
        d1 = self.decoder1(d2)  # 最终上采样

        # 最终分类器
        out = self.last_conv0(d1)
        out = self.last_conv1(out)
        out = torch.sigmoid(out)  # 将输出值压缩到[0,1]区间

        return out


rx50 = ResNeXtUNet(n_classes=1).to(device)

if __name__ == '__main__':
    output = rx50(torch.randn(1, 3, 256, 256).to(device))
    print(output.shape)