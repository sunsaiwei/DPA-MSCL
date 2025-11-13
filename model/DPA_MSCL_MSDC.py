import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out


class MCNN_1D(nn.Module):
    def __init__(self, features, dim_in=256, dropout=0):
        super(MCNN_1D, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv1d(1, 16, 1, padding=0),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(3, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(5, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU()
        ])

        # self.seblock = SEBlock(48, reduction=16)
        # self.sablock = SAblock(48, groups=16)
        self.ecablock = ECAblock(3)

        self.output_layer = nn.Sequential(
            # Add dropout here
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(48 * features, dim_in),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        convs = []
        for i in range(0, len(self.conv), 3):
            conv_layer = self.conv[i]
            batch_norm = self.conv[i + 1]
            activation = self.conv[i + 2]
            #将不同尺度数据放入不同的卷积层中进行处理
            conv = activation(batch_norm(conv_layer(x[i // 3])))
            #处理后放入convs列表中
            convs.append(conv)
        x = torch.cat(convs, dim=1)
        x = self.ecablock(x)
        # x = self.sablock(x)
        x = self.output_layer(x)
        return x
# class SAblock(nn.Module):
#     """一维版通道-空间注意力模块"""
#
#     def __init__(self, channel, groups=16):
#         super(SAblock, self).__init__()
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 修改为1D池化
#
#         # 参数维度调整为1D (1, channels, 1)
#         self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1))
#         self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1))
#         self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1))
#         self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1))
#
#         self.sigmoid = nn.Sigmoid()
#         self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
#
#     @staticmethod
#     def channel_shuffle(x, groups):
#         b, c, l = x.shape  # 修改为1D长度维度
#
#         x = x.reshape(b, groups, -1, l)      # [B, G, C/G, L]
#         x = x.permute(0, 2, 1, 3)            # [B, C/G, G, L]
#         x = x.reshape(b, -1, l)              # 合并通道维度
#         return x
#
#     def forward(self, x):
#         b, c, l = x.shape  # 输入形状改为[B, C, L]
#
#         # 分组处理
#         x = x.reshape(b * self.groups, -1, l)  # [B*G, C/G, L]
#         x_0, x_1 = x.chunk(2, dim=1)          # 分割通道
#
#         # 通道注意力分支
#         xn = self.avg_pool(x_0)               # [B*G, C/(2G), 1]
#         xn = self.cweight * xn + self.cbias   # 可学习参数
#         xn = x_0 * self.sigmoid(xn)           # 通道注意力图
#
#         # 空间注意力分支
#         xs = self.gn(x_1)                     # 组归一化
#         xs = self.sweight * xs + self.sbias
#         xs = x_1 * self.sigmoid(xs)           # 空间注意力图
#
#         # 合并结果
#         out = torch.cat([xn, xs], dim=1)      # [B*G, C/G, L]
#         out = out.reshape(b, -1, l)           # [B, C, L]
#
#         out = self.channel_shuffle(out, 2)    # 通道混洗
#         return out

class ECAblock(nn.Module):
    def __init__(self, k_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                            padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, L)
        y = self.avg_pool(x)          # (B, C, 1)
        y = y.permute(0, 2, 1)        # (B, 1, C)
        y = self.conv(y)              # 跨通道卷积
        y = self.sigmoid(y.permute(0, 2, 1))  # (B, C, 1)
        return x * y.expand_as(x)


# class SEBlock(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1)
#         return x * y.expand_as(x)



class BCLModel(nn.Module):
    """backbone + projection head"""
    def __init__(self, features, num_classes=5,  dropout=0, head='mlp', dim_in=256, feat_dim=128, use_norm=True):
        super(BCLModel, self).__init__()
        self.encoder = MCNN_1D(features, dim_in, dropout)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        if use_norm:
            self.fc = NormedLinear(dim_in, num_classes)
        else:
            self.fc = nn.Linear(dim_in, num_classes)
        self.head_fc = nn.Sequential(nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
                                     nn.Linear(dim_in, feat_dim))

    def forward(self, x):
        feat = self.encoder(x)
        feat_mlp = F.normalize(self.head(feat), dim=1)
        logits = self.fc(feat)  # 批次中每个元素对应一个类别的未归一化预测得分（还没有转化为概率分布），shape为(batch_size, num_classes)
        centers_logits = F.normalize(self.head_fc(self.fc.weight.T), dim=1)  # 这是对分类层权重进行转置后得到的规范化类别中心表示
        return feat_mlp, logits, centers_logits


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels)
        )

        # 用于匹配输入和输出通道的跳跃连接
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # 残差连接
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        # 添加残差（跳跃连接）
        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, num_classes, features):
        super(ResNet1D, self).__init__()

        self.layer1 = ResidualBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 输入通道为1
        self.layer2 = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 分类层
        self.output_layer = nn.Sequential(
            # Add dropout here
            nn.Flatten(),
            nn.Linear(64 * features, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)

        )

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.output_layer(x)

        return x

class Classifier(nn.Module):
    """可训练的分类头"""

    def __init__(self, num_classes, feature_size=256):
        super().__init__()
        self.output = ResNet1D(num_classes, feature_size)
        # 强制将全连接层参数转为 float32
        self.output = self.output.to(dtype=torch.float32)

    def forward(self, x):
        return self.output(x)

# class Classifier(nn.Module):
#     """可训练的分类头"""
#
#     def __init__(self, num_classes, feature_size=256):
#         super().__init__()
#         self.fc = nn.Linear(feature_size, num_classes)
#         # 强制将全连接层参数转为 float32
#         self.fc = self.fc.to(dtype=torch.float32)
#
#     def forward(self, x):
#         return self.fc(x)