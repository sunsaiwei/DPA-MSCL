"""
这是一个工具包，包含了获取各种数据集的方法
"""
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from utils.utils import *
import pandas as pd


def get_Hugoton_Panoma_multiscale(path, args):
    """
    获取Hugoton_Panoma数据集的多尺度训练集和测试集（一个样本中包含三个尺度的数据，并且只有一个岩性标签），数据加载器

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """

    # 从给定路径读取Excel文件并获取新疆地区的训练数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)

    # 数据清洗，删除数据中包含任意缺失值的行
    data_frame.dropna(inplace=True)
    # 选择要处理的列
    data_frame = data_frame.loc[:,
                 ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]

    # 分离标签得到特征
    X = data_frame.drop(labels='Facies', axis=1)

    # 对特征标准化
    max_min = preprocessing.StandardScaler()
    X = max_min.fit_transform(X)

    # 取得标签
    y = data_frame['Facies']
    y = y.values - 1  # 标签从1开始，这里将其转换为0开始
    y = torch.LongTensor(y)

    # 获得多尺度数据,
    train_test_data = generate_multiscale_data(X, y, args)

    # 这里为什么y_train和y_test是是取得的train_test_data[2][0]和train_test_data[3][0]看generate_multiscale_data中的注释
    # 分离训练集和测试集的特征和标签
    X_train, X_test, y_train, y_test = train_test_data[0], train_test_data[1], train_test_data[2][0], \
        train_test_data[3][0]

    # 将多尺度数据转化为自定义的多尺度数据集，方便构造数据加载器
    train_dataset = MultiScaleDataset(X_train, y_train)
    test_dataset = MultiScaleDataset(X_test, y_test)

    # 构造数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return X_train, X_test, train_dataloader, y_train, y_test, test_dataloader

def get_blind_HP_multiscale(path, blind_well1, blind_well2, args):
    """
    获取两个独立数据集的多尺度训练集和测试集（训练集和盲测集分别来自不同文件）

    参数：
    - path_train: 训练集Excel文件路径（str）
    - path_test: 测试集Excel文件路径（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """

    # 从给定路径读取Excel文件并获取盲测试数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)

    # 训练集：排除两个盲井的所有数据
    train_frame = data_frame[(data_frame['Well Name'] != blind_well1) &
                             (data_frame['Well Name'] != blind_well2)]
    # 盲测试集：使用两个盲井的数据
    blind_frame = data_frame[(data_frame['Well Name'] == blind_well1) |
                             (data_frame['Well Name'] == blind_well2)]

    # 数据预处理函数
    def preprocess(df):
        df = df.dropna()  # 删除空值
        features = df.loc[:, ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
        X = features.drop(labels='Facies', axis=1)
        # 对特征标准化
        standard_scaler = preprocessing.StandardScaler()
        X = standard_scaler.fit_transform(X)

        # 取得标签
        y = df['Facies'].values - 1  # 关键修改：将标签值偏移-1，使得标签值从0开始
        y = torch.LongTensor(y)
        return X, y

    # 处理训练集
    X_train, y_train = preprocess(train_frame)
    # 处理测试集
    X_test, y_test = preprocess(blind_frame)

    # 生成多尺度数据（假设generate_multiscale_blind返回张量）
    X_train, _ = generate_multiscale_blind(X_train, y_train)
    X_test, _ = generate_multiscale_blind(X_test, y_test)

    # 转换为PyTorch张量
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # 创建数据集
    train_dataset = MultiScaleDataset(X_train, y_train)
    test_dataset = MultiScaleDataset(X_test, y_test)

    # 创建数据加载器（假设batch_size在可见域中）
    train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                               batch_size=args.batch_size,
                               shuffle=False)


    return X_train, X_test, train_dataloader, y_train, y_test, test_dataloader

def get_daqing_multiscale(path, args):
    """
    获取部分大庆数据集的多尺度训练集和测试集（一个样本中包含三个尺度的数据，并且只有一个岩性标签），数据加载器

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """
    # 从给定路径读取Excel文件并获取训练数据，同时生成多尺度数据集和数据加载器
    data_frame = pd.read_excel(path)

    # 数据清洗，删除数据中包含任意缺失值的行
    data_frame.dropna(inplace=True)

    # 去除岩性为凝灰岩的数据，也就是Face=5的数据
    # data_frame = data_frame[data_frame['Face'] != 5]
    # 选择要处理的列
    data_frame = data_frame.loc[:,
                 ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish",
                  "Face"]]

    # 分离标签得到特征
    X = data_frame.drop(labels='Face', axis=1)
    X = X.values
    # 对特征标准化
    standard_scaler = preprocessing.StandardScaler()
    X = standard_scaler.fit_transform(X)

    # 取得标签
    y = data_frame['Face']
    y = y.values
    y = torch.LongTensor(y)

    # 获得多尺度数据,
    train_test_data = generate_multiscale_data(X, y, args)

    # 这里为什么y_train和y_test是是取得的train_test_data[2][0]和train_test_data[3][0]看generate_multiscale_data中的注释
    # 分离训练集和测试集的特征和标签
    X_train, X_test, y_train, y_test = train_test_data[0], train_test_data[1], train_test_data[2][0], \
        train_test_data[3][0]

    # 将多尺度数据转化为自定义的多尺度数据集，方便构造数据加载器
    train_dataset = MultiScaleDataset(X_train, y_train)
    test_dataset = MultiScaleDataset(X_test, y_test)

    # 构造数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return X_train, X_test, train_dataloader, y_train, y_test, test_dataloader

def get_blind_daqing_multiscale(path1, path2, args):
    """
    获取部分大庆数据集的多尺度训练集和盲测集（一个样本中包含三个尺度的数据，并且只有一个岩性标签），数据加载器

    参数：
    - path: Excel文件的路径（str）

    返回：
    - X_train: 训练集特征（numpy array）
    - y_train: 训练集岩性标签（Tensor）
    - train_dataloader: 训练集数据加载器（DataLoader）
    - X_test: 测试集特征（numpy array）
    - y_test: 测试集岩性标签（Tensor）
    - test_dataloader: 测试集数据加载器（DataLoader）
    """
    # 从给定路径读取Excel文件并获取训练数据，同时生成多尺度数据集和数据加载器
    train_frame = pd.read_excel(path1)
    # 数据清洗，删除数据中包含任意缺失值的行
    train_frame.dropna(inplace=True)

    # 从个给定路径读取Excel文件并获取盲井数据，同时生成多尺度数据集和数据加载器
    blind_frame = pd.read_excel(path2)
    # 数据清洗，删除数据中包含任意缺失值的行
    blind_frame.dropna(inplace=True)


    # 选择要处理的列
    train_frame = train_frame.loc[:,
                 ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish",
                  "Face"]]
    blind_frame = blind_frame.loc[:,
                 ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish",
                  "Face"]]

    # 处理训练集
    X_train = train_frame.drop(labels='Face', axis=1)
    X_train = X_train.values
    # 对特征标准化
    standard_scaler = preprocessing.StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)

    # 取得标签
    y_train = train_frame['Face']
    y_train = y_train.values
    y_train = torch.LongTensor(y_train)

    # 处理盲测集
    X_test = blind_frame.drop(labels='Face', axis=1)
    X_test = X_test.values
    # 对特征标准化
    X_test = standard_scaler.transform(X_test)

    # 取得标签
    y_test = blind_frame['Face']
    y_test = y_test.values
    y_test = torch.LongTensor(y_test)

    # 生成多尺度数据
    X_train, _ = generate_multiscale_blind(X_train, y_train)
    X_test, _ = generate_multiscale_blind(X_test, y_test)

    # 创建数据集
    train_dataset = MultiScaleDataset(X_train, y_train)
    test_dataset = MultiScaleDataset(X_test, y_test)

    # 创建数据加载器（假设batch_size在可见域中）
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return X_train, X_test, train_dataloader, y_train, y_test, test_dataloader
