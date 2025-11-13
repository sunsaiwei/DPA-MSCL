"""
这是一个工具包，包含了一些辅助函数，包括一些关键函数和类
"""
import pickle
import math

import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt

# 解析命令行参数

class MultiView_Dataset(Dataset):
    def __init__(self, dataset_view1, dataset_view2, dataset_view3):
        """
        dataset_view1, dataset_view2, dataset_view3: 分别代表三个视角的数据集，
        每个数据集均需满足 __len__ 和 __getitem__ 方法，并且样本的索引对齐。
        """
        assert len(dataset_view1) == len(dataset_view2) == len(dataset_view3), \
            "三个数据集的样本数量必须相等"
        self.dataset1 = dataset_view1
        self.dataset2 = dataset_view2
        self.dataset3 = dataset_view3

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        data1, label1 = self.dataset1[idx]
        data2, label2 = self.dataset2[idx]
        data3, label3 = self.dataset3[idx]
        # 如果三个视角的标签一定一致，可以只取一个
        label = label1
        # 返回一个包含三个视角数据的元组，以及对应的标签
        return (data1, data2, data3), label

class MultiScaleDataset(Dataset):
    """
       多尺度数据集类，将不同尺度的特征数据组织成 PyTorch 数据集格式。

       作用：
       - 该类用于存储和组织多尺度特征数据，每个样本在不同尺度（例如 1、3、5）下的特征会被封装成一个元组，并与对应的标签配对，以便模型在训练时能够同时利用不同尺度的信息。

       数据格式：
       - self.data_by_scale: 一个列表，每个元素是不同尺度的特征数据，每个尺度的数据样本数量相同。
       - self.labels: 一个列表，存储所有样本的标签，每个样本的不同尺度数据共享同一个标签。

       参数：
       - data_scale_features: 以不同尺度组织的特征数据，列表格式，每个列表元素对应一个尺度的样本集。
       - labels: 样本对应的标签列表。

       方法：
       - __len__(): 返回数据集中样本的数量（即标签的数量）。
       - __getitem__(index): 返回索引 index 处的样本，包括 (尺度1样本, 尺度3样本, 尺度5样本) 及对应的标签。

       示例：
       假设 data_by_scale 结构如下：
       ```
       data_by_scale = [
           [样本0_尺度1, 样本1_尺度1, ..., 样本N_尺度1], # 尺度1
           [样本0_尺度3, 样本1_尺度3, ..., 样本N_尺度3], # 尺度3
           [样本0_尺度5, 样本1_尺度5, ..., 样本N_尺度5]  # 尺度5
       ]
       labels = [标签0, 标签1, ..., 标签N]
       ```
       当 index = 2 时，返回：
       ```
       (样本2_尺度1, 样本2_尺度3, 样本2_尺度5), 标签2
       ```
       这样模型可以同时接收多个尺度的数据进行训练。
       """
    def __init__(self, data_scale_features, labels):
        self.data_by_scale = data_scale_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return tuple(data[index] for data in self.data_by_scale), self.labels[index]


def convert_to_tensor(data_list):
    """
    将列表中的数据转换为 PyTorch 浮点张量（Tensor）。

    参数：
    - data_list: 包含多个数组或列表的数据（list of list/array）。

    返回：
    - 转换后的 PyTorch 张量列表（list of torch.Tensor）。
    """
    return [torch.tensor(data, dtype=torch.float32) for data in data_list]



class MultiScale_Dataset(Dataset):
    """
       多尺度数据集类，将不同尺度的特征数据组织成 PyTorch 数据集格式。

       作用：
       - 该类用于存储和组织多尺度特征数据，每个样本在不同尺度（例如 1、3、5）下的特征会被封装成一个元组，并与对应的标签配对，以便模型在训练时能够同时利用不同尺度的信息。

       数据格式：
       - self.data_by_scale: 一个张量，每个元素是不同尺度的特征数据，每个尺度的数据样本数量相同。
       - self.labels: 一个列表，存储所有样本的标签，每个样本的不同尺度数据共享同一个标签。

       参数：
       - data_scale_features: 以不同尺度组织的特征数据，列表格式，每个列表元素对应一个尺度的样本集。
       - labels: 样本对应的标签列表。

       方法：
       - __len__(): 返回数据集中样本的数量（即标签的数量）。
       - __getitem__(index): 返回索引 index 处的样本，包括 (尺度1样本, 尺度3样本, 尺度5样本) 及对应的标签。

       示例：
       假设 data_by_scale 结构如下：
       ```
       data_by_scale = [
           [样本0_尺度1, 样本1_尺度1, ..., 样本N_尺度1], # 尺度1
           [样本0_尺度3, 样本1_尺度3, ..., 样本N_尺度3], # 尺度3
           [样本0_尺度5, 样本1_尺度5, ..., 样本N_尺度5]  # 尺度5
       ]
       labels = [标签0, 标签1, ..., 标签N]
       ```
       当 index = 2 时，返回：
       ```
       (样本2_尺度1, 样本2_尺度3, 样本2_尺度5), 标签2
       ```
       这样模型可以同时接收多个尺度的数据进行训练。
       """
    def __init__(self, data_scale_features, labels):
        self.data_by_scale = data_scale_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return tuple(data[index] for data in self.data_by_scale), self.labels[index]

def prepare_multiscale1_data(features):
    """
    生成不同尺度（1、3、5）的特征数据，并转换为 PyTorch 张量。

    参数：
    - features: 原始特征数据（NumPy 数组），形状为 (样本数, 特征数)。

    处理过程：
    1. 创建 new_data_scale_features，用于存储不同尺度的特征数据：
       - 尺度1：当前样本的特征。
       - 尺度3：当前样本 + 前后各一个样本（边界处用零填充）。
       - 尺度5：当前样本 + 前后各两个样本（边界处用零填充）。
    2. 遍历所有样本，为每个样本生成不同尺度的数据，并存入相应的列表中。
    3. 将列表转换为 PyTorch 张量，以便后续模型使用。

    返回：
    - new_data_by_scale: 包含不同尺度特征的 PyTorch 张量列表。
    """
    scales = [1, 3, 5]
    #创建一个列表new_data_by_scale，其中包含三个空列表，用于存储在不同尺度上的数据。这三个子列表对应于scales列表中的三个尺度。
    new_data_scale_features = [[] for _ in scales]
    for i in range(len(features)):
        #对于尺度1，直接将数据中的每个点添加到new_data_by_scale[0]中。
        new_data_scale_features[0].append(features[i])

        #对于尺度3，根据数据点的位置在new_data_by_scale[1]中构建包含当前点及其前一个和后一个点的数据。
        # 对于第一个数据点，没有前一个点，用零填充。对于最后一个数据点，没有后一个点，同样用零填充。
        if i == 0:
            new_data_scale_features[1].append(np.vstack([np.zeros(features.shape[1]), features[i], features[min(i + 1, len(features) - 1)]]))
        elif i == len(features) - 1:
            new_data_scale_features[1].append(np.vstack([features[max(i - 1, 0)], features[i], np.zeros(features.shape[1])]))
        else:
            new_data_scale_features[1].append(np.vstack([features[i - 1], features[i], features[i + 1]]))

        #对于尺度5，根据数据点的位置在new_data_by_scale[2]中构建包含当前点及其前两个和后两个点的数据。
        # 对于第一个数据点，没有前两个点，用零填充。对于最后一个数据点，没有后两个点，同样用零填充。
        left_padding_2 = np.zeros(features.shape[1]) if i - 2 < 0 else features[i - 2]
        left_padding_1 = np.zeros(features.shape[1]) if i - 1 < 0 else features[i - 1]
        right_padding_1 = np.zeros(features.shape[1]) if i + 1 >= len(features) else features[i + 1]
        right_padding_2 = np.zeros(features.shape[1]) if i + 2 >= len(features) else features[i + 2]
        new_data_scale_features[2].append(
            np.vstack([left_padding_2, left_padding_1, features[i], right_padding_1, right_padding_2]))

    #将数据从Numpy数组的形式转换为张量
    new_data_by_scale = convert_to_tensor(new_data_scale_features)
    return new_data_by_scale

def prepare_multiscale2_data(features):
    """
    生成不同尺度（3、5、7）的特征数据，并转换为 PyTorch 张量。

    参数：
    - features: 原始特征数据（NumPy 数组），形状为 (样本数, 特征数)。

    处理过程：
    1. 创建 new_data_scale_features，用于存储不同尺度的特征数据：
       - 尺度3：当前样本及其前后各一个样本（边界处用零填充）
       - 尺度5：当前样本及其前后各两个样本（边界处用零填充）
       - 尺度7：当前样本及其前后各三个样本（边界处用零填充）
    2. 遍历所有样本，为每个样本生成不同尺度的数据，并存入相应的列表中。
    3. 将列表转换为 PyTorch 张量，以便后续模型使用。

    返回：
    - new_data_by_scale: 包含不同尺度特征的 PyTorch 张量列表。
    """
    scales = [3, 5, 7]
    new_data_scale_features = [[] for _ in scales]
    num_samples, num_features = features.shape

    for i in range(num_samples):
        # 处理尺度3（前后各1个样本）
        if i == 0:
            scale3_data = np.vstack([
                np.zeros(num_features),
                features[i],
                features[min(i+1, num_samples-1)]
            ])
        elif i == num_samples - 1:
            scale3_data = np.vstack([
                features[max(i-1, 0)],
                features[i],
                np.zeros(num_features)
            ])
        else:
            scale3_data = np.vstack([
                features[i-1],
                features[i],
                features[i+1]
            ])
        new_data_scale_features[0].append(scale3_data)

        # 处理尺度5（前后各2个样本）
        left2 = features[i-2] if i-2 >=0 else np.zeros(num_features)
        left1 = features[i-1] if i-1 >=0 else np.zeros(num_features)
        right1 = features[i+1] if i+1 < num_samples else np.zeros(num_features)
        right2 = features[i+2] if i+2 < num_samples else np.zeros(num_features)
        scale5_data = np.vstack([left2, left1, features[i], right1, right2])
        new_data_scale_features[1].append(scale5_data)

        # 处理尺度7（前后各3个样本）
        left3 = features[i-3] if i-3 >=0 else np.zeros(num_features)
        left2 = features[i-2] if i-2 >=0 else np.zeros(num_features)
        left1 = features[i-1] if i-1 >=0 else np.zeros(num_features)
        right1 = features[i+1] if i+1 < num_samples else np.zeros(num_features)
        right2 = features[i+2] if i+2 < num_samples else np.zeros(num_features)
        right3 = features[i+3] if i+3 < num_samples else np.zeros(num_features)
        scale7_data = np.vstack([
            left3, left2, left1,
            features[i],
            right1, right2, right3
        ])
        new_data_scale_features[2].append(scale7_data)

    # 转换为张量
    new_data_by_scale = convert_to_tensor(new_data_scale_features)
    return new_data_by_scale

def generate_multiview_data(features, labels, test_size, random_state):
    """
    生成多视角数据，并进行标准化和训练/测试集划分。

    作用：
    - 该方法首先对输入的原始数据进行多视角数据生成，生成不同尺度的特征数据。
    - 对每个尺度的数据进行标准化，以保证不同尺度的数据分布一致。
    - 将数据按照 test_size 的比例划分为训练集和测试集，以供模型训练和评估。

    参数：
    - data: 原始特征数据（Tensor）。
    - labels: 样本对应的标签（列表）。

    返回：
    - train_test_data: 列表，包含以下四个部分：
      1. X_train: 训练集特征（包含不同尺度）。
      2. X_test: 测试集特征（包含不同尺度）。
      3. y_train: 训练集标签。
      4. y_test: 测试集标签。



    示例：
    ```
    train_test_data = generate_multiscale_data(data, labels)
    X_train, X_test, y_train, y_test = train_test_data
    ```
    """
    X_multiview_train, X_multiview_test = [], []
    X_train1, X_test1, y_train1, y_test1 = [], [], [], []
    X_train2, X_test2, y_train2, y_test2 = [], [], [], []
    new_data_by_scale1 = prepare_multiscale1_data(features)# 多尺度[1,3,5]
    new_data_by_scale2 = prepare_multiscale2_data(features)# 多尺度[3,5,7]
    k = 1
    j = 3
    X_train, X_test, y_train, y_test = [], [], [], []
    train_test_data = []
    # 生成多视角数据：每个样本是一个列表，视角1：原生数据；视角2：多尺度数据1:；视角3：多尺度数据2
    features_tensor = torch.tensor(features, dtype=torch.float32)
    X_raw_train, X_raw_test, y_raw_train, y_raw_test = (
        train_test_split(features_tensor, labels, test_size=test_size, random_state=random_state))
    X_multiview_train.append(X_raw_train)
    X_multiview_test.append(X_raw_test)
    y_multiview_train = y_raw_train
    y_multiview_test =y_raw_test

    # 将输入的多尺度数据进行预处理，并划分成训练集和测试集
    for i in range(len(new_data_by_scale1)):
        reshaped_data = new_data_by_scale1[i] \
            .reshape(new_data_by_scale1[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
        # 将每个尺度的数据变换为维度为(N,C,L)的张量，其中N为样本数，C为通道数，L为特征长度
        if i == 0:
            new_data_by_scale1[i] = new_data_by_scale1[i].unsqueeze(1)
        else:
            new_data_by_scale1[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale1[i].size(0), k, new_data_by_scale1[i].size(2))
        k = k + 2

        X_multiscale_train, X_multiscale_test, y_multiscale_train, y_multiscale_test = (
            train_test_split(new_data_by_scale1[i], labels, test_size=test_size, random_state=random_state))
        X_train1.append(X_multiscale_train)
        X_test1.append(X_multiscale_test)
        y_train1.append(y_multiscale_train)
        y_test1.append(y_multiscale_test)
    X_multiview_train.append(X_train1)
    X_multiview_test.append(X_test1)


    for i in range(len(new_data_by_scale2)):
        reshaped_data = new_data_by_scale2[i] \
            .reshape(new_data_by_scale2[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
        # 将每个尺度的数据变换为维度为(N,C,L)的张量，其中N为样本数，C为通道数，L为特征长度

        new_data_by_scale2[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale2[i].size(0), j, new_data_by_scale2[i].size(2))
        j = j + 2

        X_multiscale_train, X_multiscale_test, y_multiscale_train, y_multiscale_test = (
            train_test_split(new_data_by_scale2[i], labels, test_size=test_size, random_state=random_state)
        )
        X_train2.append(X_multiscale_train)
        X_test2.append(X_multiscale_test)
        y_train2.append(y_multiscale_train)
        y_test2.append(y_multiscale_test)
    X_multiview_train.append(X_train2)
    X_multiview_test.append(X_test2)


    return X_multiview_train, X_multiview_test, y_multiview_train, y_multiview_test

def generate_multiview_blind(features, labels):
    """
    生成盲井的多视角数据，不进行训练集和测试集划分

    参数：
    - features: 原始特征数据（Tensor/Array）
    - labels: 样本对应的标签（列表）

    返回：
    - multiview_data: 列表，包含三个视角的数据：
      1. 原始数据视图
      2. 多尺度数据视图1（尺度[1,3,5]）
      3. 多尺度数据视图2（尺度[3,5,7]）
    - labels: 原始标签（保持原样）
    """
    # 生成多尺度基础数据
    new_data_by_scale1 = prepare_multiscale1_data(features)
    new_data_by_scale2 = prepare_multiscale2_data(features)

    # 初始化多视角容器
    multiview_data = []

    # 第一视角：原始数据标准化
    scaler_raw = preprocessing.StandardScaler()
    X_raw = scaler_raw.fit_transform(features)
    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32)
    multiview_data.append(X_raw_tensor)

    # 第二视角：多尺度数据1（1,3,5）
    view2_data = []
    k = 1  # 初始通道数
    for i in range(len(new_data_by_scale1)):
        # 标准化处理
        reshaped_data = new_data_by_scale1[i].reshape(new_data_by_scale1[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(reshaped_data)

        # 形状重构
        scaled_tensor = torch.FloatTensor(scaled_data)
        if i == 0:
            scaled_tensor = scaled_tensor.unsqueeze(1)
        else:
            scaled_tensor = scaled_tensor.reshape(
                new_data_by_scale1[i].size(0),
                k,
                new_data_by_scale1[i].size(2)
            )
        k += 2

        view2_data.append(scaled_tensor)
    multiview_data.append(view2_data)

    # 第三视角：多尺度数据2（3,5,7）
    view3_data = []
    j = 3  # 初始通道数
    for i in range(len(new_data_by_scale2)):
        # 标准化处理
        reshaped_data = new_data_by_scale2[i].reshape(new_data_by_scale2[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(reshaped_data)

        # 形状重构
        scaled_tensor = torch.FloatTensor(scaled_data).reshape(
            new_data_by_scale2[i].size(0),
            j,
            new_data_by_scale2[i].size(2)
        )
        j += 2

        view3_data.append(scaled_tensor)
    multiview_data.append(view3_data)

    return multiview_data, labels


def generate_multiscale_data(data, labels, args):
    """
    生成多尺度数据，并进行标准化和训练/测试集划分。

    作用：
    - 该方法首先对输入的原始数据进行多尺度转换，生成不同尺度的特征数据。
    - 对每个尺度的数据进行标准化，以保证不同尺度的数据分布一致。
    - 将数据按照 test_size 的比例划分为训练集和测试集，以供模型训练和评估。

    参数：
    - data: 原始特征数据（Tensor）。
    - labels: 样本对应的标签（列表）。

    返回：
    - train_test_data: 列表，包含以下四个部分：
      1. X_train: 训练集特征（包含不同尺度）。
      2. X_test: 测试集特征（包含不同尺度）。
      3. y_train: 训练集标签。
      4. y_test: 测试集标签。

    处理过程：
    1. 使用 `prepare_multiscale_data(data)` 生成不同尺度的特征数据，每个尺度的数据存储在 `new_data_by_scale` 中。
    2. 遍历 `new_data_by_scale`:
       - 先将数据重塑为二维数组，以便进行标准化处理。
       - 使用 `StandardScaler` 进行标准化，使数据均值为 0，方差为 1。
       - 重新调整数据形状，使其符合 (N, C, L) 形式：
         - N: 样本数
         - C: 通道数（不同尺度的通道数不同）
         - L: 每个样本的特征长度
    3. 使用 `train_test_split` 将每个尺度的数据随机划分为训练集（70%）和测试集（30%）。
    4. `X_train` 和 `X_test` 分别存储所有尺度的训练数据和测试数据。
    5. `y_train` 和 `y_test` 存储划分后的标签。
    6. 由于所有尺度的数据共享相同的标签，`y_train` 和 `y_test` 各包含三个相同的标签列表。

    示例：
    ```
    train_test_data = generate_multiscale_data(data, labels)
    X_train, X_test, y_train, y_test = train_test_data
    ```
    """
    new_data_by_scale = prepare_multiscale_data(data)
    j = 1
    X_train, X_test, y_train, y_test = [], [], [], []
    train_test_data = []
    # 将输入的多尺度数据进行预处理，并划分成训练集和测试集
    for i in range(len(new_data_by_scale)):
        # 把每个尺度的数据重塑为一个二维数组以便于标准化
        reshaped_data = new_data_by_scale[i] \
            .reshape(new_data_by_scale[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
        # 将每个尺度的数据变换为维度为(N,C,L)的张量，其中N为样本数，C为通道数，L为特征长度
        if i == 0:
            new_data_by_scale[i] = new_data_by_scale[i].unsqueeze(
                1)  # 对于尺度1在第二个维度上增加一个大小为1的维度，因为unsqueeze()函数新增维度的大小固定为1
        else:
            new_data_by_scale[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale[i].size(0), j, new_data_by_scale[i].size(2))
        j = j + 2
        # 将每个尺度的数据划分为训练集和测试集
        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(new_data_by_scale[i], labels,
                                                                                test_size=args.test_size,
                                                                                random_state=args.random_state)
        X_train.append(X_train_data)
        X_test.append(X_test_data)
        y_train.append(y_train_data)
        y_test.append(y_test_data)
    train_test_data.append(X_train)
    train_test_data.append(X_test)
    train_test_data.append(y_train)
    train_test_data.append(y_test)
    return train_test_data

def prepare_multiscale_data(features):
    """
    生成不同尺度（1、3、5）的特征数据，并转换为 PyTorch 张量。

    参数：
    - features: 原始特征数据（NumPy 数组），形状为 (样本数, 特征数)。

    处理过程：
    1. 创建 new_data_scale_features，用于存储不同尺度的特征数据：
       - 尺度1：当前样本的特征。
       - 尺度3：当前样本 + 前后各一个样本（边界处用零填充）。
       - 尺度5：当前样本 + 前后各两个样本（边界处用零填充）。
    2. 遍历所有样本，为每个样本生成不同尺度的数据，并存入相应的列表中。
    3. 将列表转换为 PyTorch 张量，以便后续模型使用。

    返回：
    - new_data_by_scale: 包含不同尺度特征的 PyTorch 张量列表。
    """
    scales = [1, 3, 5]
    #创建一个列表new_data_by_scale，其中包含三个空列表，用于存储在不同尺度上的数据。这三个子列表对应于scales列表中的三个尺度。
    new_data_scale_features = [[] for _ in scales]
    for i in range(len(features)):
        #对于尺度1，直接将数据中的每个点添加到new_data_by_scale[0]中。
        new_data_scale_features[0].append(features[i])

        #对于尺度3，根据数据点的位置在new_data_by_scale[1]中构建包含当前点及其前一个和后一个点的数据。
        # 对于第一个数据点，没有前一个点，用零填充。对于最后一个数据点，没有后一个点，同样用零填充。
        if i == 0:
            new_data_scale_features[1].append(np.vstack([np.zeros(features.shape[1]), features[i], features[min(i + 1, len(features) - 1)]]))
        elif i == len(features) - 1:
            new_data_scale_features[1].append(np.vstack([features[max(i - 1, 0)], features[i], np.zeros(features.shape[1])]))
        else:
            new_data_scale_features[1].append(np.vstack([features[i - 1], features[i], features[i + 1]]))

        #对于尺度5，根据数据点的位置在new_data_by_scale[2]中构建包含当前点及其前两个和后两个点的数据。
        # 对于第一个数据点，没有前两个点，用零填充。对于最后一个数据点，没有后两个点，同样用零填充。
        left_padding_2 = np.zeros(features.shape[1]) if i - 2 < 0 else features[i - 2]
        left_padding_1 = np.zeros(features.shape[1]) if i - 1 < 0 else features[i - 1]
        right_padding_1 = np.zeros(features.shape[1]) if i + 1 >= len(features) else features[i + 1]
        right_padding_2 = np.zeros(features.shape[1]) if i + 2 >= len(features) else features[i + 2]
        new_data_scale_features[2].append(
            np.vstack([left_padding_2, left_padding_1, features[i], right_padding_1, right_padding_2]))

    #将数据从Numpy数组的形式转换为张量
    new_data_by_scale = convert_to_tensor(new_data_scale_features)
    return new_data_by_scale


def generate_multiscale_blind(features, labels):
    """
    生成多尺度数据（盲井预测场景），并进行标准化处理。

    作用：
    - 该方法对输入的原始特征数据进行多尺度转换，生成不同尺度的特征数据。
    - 对每个尺度的数据进行标准化，以保证数据分布一致，提升模型的泛化能力。
    - 返回处理后的多尺度数据和对应的标签，供模型进行训练或预测。

    参数：
    - features: 原始特征数据（Tensor）。
    - labels: 样本对应的标签（列表）。

    返回：
    - new_data_by_scale: 处理后的多尺度特征数据（列表，每个元素对应一个尺度）。
    - labels: 与输入数据对应的标签（未修改）。
    """
    new_data_by_scale = prepare_multiscale_data(features)
    j=1
    for i in range(len(new_data_by_scale)):
        reshaped_data = new_data_by_scale[i] \
            .reshape(new_data_by_scale[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
        if i == 0:
            new_data_by_scale[i] = new_data_by_scale[i].unsqueeze(1)
        else:
            new_data_by_scale[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale[i].size(0), j, new_data_by_scale[i].size(2))
        j = j + 2
    return new_data_by_scale, labels





def get_cls_num_list(y_train, num_classes):
    # 创建按类别分组的容器
    class_data = [[] for _ in range(num_classes)]

    # 遍历所有训练样本的标签
    for i in range(len(y_train)):
        y = y_train[i]
        class_data[y].append(i)  # 将样本索引存入对应类别的列表

    # 生成类别数量统计列表
    cls_num_list = [len(class_data[i]) for i in range(num_classes)]
    return cls_num_list

# 定义warm-up调度函数
def warmup_lr_scheduler(args, optimizer, current_epoch):
    if current_epoch <= args.warm_epochs:
        lr = args.warmup_from + (args.warmup_to - args.warmup_from) * \
             current_epoch / args.warm_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer


def write_file(file_path,predicted):
    """
    该函数用于将预测结果写入指定的文件中。每个预测结果以换行符分隔，逐行写入文件。

    参数：
    file_path (str)：目标文件的路径，预测结果将被写入到该文件中。

    predicted (iterable)：包含预测结果的可迭代对象（如列表或 PyTorch 张量）。每个元素表示一个预测结果，item() 方法将用于获取预测值。

    返回值：
    该方法没有返回值。它的作用是将预测结果逐行写入文件。
    """
    with open(file_path, "w") as file:
        for prediction in predicted:
            file.write(f"{prediction.item()}\n")
        file.close()

def get_confusion_matrix(trues, preds):
    """
    计算并归一化混淆矩阵。

    作用：
    - 该方法计算真实标签与预测标签之间的混淆矩阵，并对其按行进行归一化，以便分析分类模型的表现。
    - 归一化后的矩阵表示每个类别的预测分布，方便直观理解分类器的错误率和正确率。

    参数：
    - trues: 真实标签（列表或数组）。
    - preds: 预测标签（列表或数组）。

    返回：
    - 归一化后的混淆矩阵（百分比格式，数值范围 0-100，保留两位小数）。
    """
    conf_matrix = confusion_matrix(trues, preds)
    # 按行归一化为小数（0-1范围）
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除以0
    conf_matrix_normalized = conf_matrix / row_sums
    return np.round(conf_matrix_normalized * 100, 2)  # 乘以100后保留两位小数

def save_metrics_plot(args, accuracy, precision, recall, f1, save_dir, model_name):
    """
    可视化评估指标并保存为图片
    参数:
        accuracy (float): 准确率
        precision (float): 精确率
        recall (float): 召回率
        f1 (float): F1分数
        save_dir (str): 保存路径
        model_name (str): 模型名称

    返回值:
        无返回值，该方法将绘制的评估指标保存为图片并展示。
    """
    # 创建绘图对象
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)

    # 隐藏坐标轴
    ax.axis('off')

    # 创建表格数据
    cell_text = [
        [f"{accuracy:.4f}"],
        [f"{precision:.4f}"],
        [f"{recall:.4f}"],
        [f"{f1:.4f}"]
    ]

    # 绘制表格
    table = ax.table(
        cellText=cell_text,
        rowLabels=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        colLabels=['Value'],
        loc='center',
        cellLoc='center'
    )

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # 设置标题
    plt.title(f'Evaluation Metrics - {model_name} on {args.dataset}', fontsize=14, pad=20)

    # 自动调整布局并保存
    plt.tight_layout()
    save_path = save_dir + f'{model_name}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()







class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 定义warm-up调度函数
def warmup_lr_scheduler(args, optimizer, current_epoch):
    if current_epoch <= args.warm_epochs:
        lr = args.warmup_from + (args.warmup_to - args.warmup_from) * \
             current_epoch / args.warm_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer
