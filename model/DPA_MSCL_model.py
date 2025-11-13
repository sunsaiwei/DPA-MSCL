import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out


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



class ResNet1D(nn.Module):
    def __init__(self,  features, dim_in=256, dropout=0):
        super(ResNet1D, self).__init__()

        self.layer1 = ResidualBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 输入通道为1
        self.layer2 = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ecablock = ECAblock(5)


        # 输出层
        self.output_layer = nn.Sequential(
            # Add dropout here
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(64 * features, 256),
            nn.BatchNorm1d(dim_in),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, dim_in))

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.ecablock(x)
        x = self.output_layer(x)

        return x




class MCNN_1D_1(nn.Module):
    def __init__(self, features, dim_in=256, dropout=0):
        super(MCNN_1D_1, self).__init__()
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
        x = self.output_layer(x)
        return x



class MCNN_1D_2(nn.Module):
    def __init__(self, features, dim_in=256, dropout=0):
        super(MCNN_1D_2, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv1d(3, 16, 1, padding=0),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(5, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(7, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU()
        ])
        self.ecablock = ECAblock(5)


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
        x = self.output_layer(x)
        return x

# 平衡对比学习中的第二个分支
class BCLModel(nn.Module):
    """backbone + projection head"""
    def __init__(self, features, num_classes=5, dropout=0, head='mlp', dim_in=256, feat_dim=128, use_norm=True):
        super(BCLModel, self).__init__()
        self.encoder1 = ResNet1D(features, dim_in, dropout)
        self.encoder2 = MCNN_1D_1(features, dim_in, dropout)
        self.encoder3 = MCNN_1D_2(features, dim_in, dropout)
        self.weights = nn.Parameter(torch.randn(2), requires_grad=True)
        if head == 'linear':
            self.head1 = nn.Linear(dim_in, feat_dim)
            self.head2 = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
            self.head2 = nn.Sequential(
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
        dynamic_weights = F.softmax(self.weights, dim=0)
        feat1 = self.encoder1(x[0])
        feat2 = self.encoder2(x[1])
        feat3 = self.encoder3(x[2])

        z2 = F.normalize(self.head1(feat2), dim=1)
        z3 = F.normalize(self.head2(feat3), dim=1)
        # z2 *= dynamic_weights[0]
        # z3 *= dynamic_weights[1]

        z = torch.cat([z2.unsqueeze(1), z3.unsqueeze(1)], dim=1)
        
        logits = self.fc(feat1)  # 批次中每个元素对应一个类别的未归一化预测得分（还没有转化为概率分布），shape为(batch_size, num_classes)
        centers_logits = F.normalize(self.head_fc(self.fc.weight.T), dim=1)  # 这是对分类层权重进行转置后得到的规范化类别中心表示
        return z, logits, centers_logits


class Classifier(nn.Module):
    """可训练的分类头"""

    def __init__(self, num_classes, dropout=0, feature_size=256*2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, num_classes))
        # 强制将全连接层参数转为 float32
        self.fc = self.fc.to(dtype=torch.float32)

    def forward(self, x):
        return self.fc(x)


# class CNN_1D_Classifier(nn.Module):
#     def __init__(self, num_classes, dropout=0, feature_size=256*2):
#         super(CNN_1D_Classifier, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(1, 32, 3, padding=1),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(32, 32, 3, padding=1),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU()
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(32, 64, 3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU()
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv1d(64, 64, 3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU()
#         )
#         self.output_layer = nn.Sequential(
#             # Add dropout here
#             nn.Flatten(),
#             nn.Linear(64 * feature_size, 256),
#             nn.BatchNorm1d(256),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, num_classes)
#
#         )
#
#     def forward(self, x):
#         # 将x增加一个通道维度
#         x = x.unsqueeze(1)  # 假设输入x的形状为(batch_size, features)，需要增加一个通道维度
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.output_layer(x)
#         return x



