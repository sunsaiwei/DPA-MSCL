"""
这是一个工具包，包含了获取各种数据集的方法
"""
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from utils.utils import *
import pandas as pd


def get_Hugoton_Panoma_MultiViewData(path, args):
    # 数据加载与预处理
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    data_frame = data_frame[["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]

    # 数据分割
    X = data_frame.drop("Facies", axis=1).values
    y = data_frame["Facies"].values - 1  # 标签从1开始，这里将其转换为0开始
    y = torch.LongTensor(y)
    # 多视角数据生成
    X_multiview_train, X_multiview_test, y_multiview_train, y_multiview_test = generate_multiview_data(X, y, args.test_size, args.random_state)

    # 构建原始数据集
    raw_train_dataset = TensorDataset(X_multiview_train[0], y_multiview_train)
    raw_test_dataset = TensorDataset(X_multiview_test[0], y_multiview_test)

    # 构建多尺度数据集1
    multi1_train_dataset = MultiScale_Dataset(X_multiview_train[1], y_multiview_train)
    multi1_test_dataset = MultiScale_Dataset(X_multiview_test[1], y_multiview_test)

    # 构建多尺度数据集2
    multi2_train_dataset = MultiScale_Dataset(X_multiview_train[2], y_multiview_train)
    multi2_test_dataset = MultiScale_Dataset(X_multiview_test[2], y_multiview_test)

    # 组装多视角数据集
    multiview_train_dataset = MultiView_Dataset(raw_train_dataset, multi1_train_dataset, multi2_train_dataset)
    multiview_test_dataset = MultiView_Dataset(raw_test_dataset, multi1_test_dataset, multi2_test_dataset)

    # 构建数据加载器
    train_dataloader = DataLoader(
        multiview_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        multiview_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return X_multiview_train, X_multiview_test, train_dataloader, y_multiview_train, y_multiview_test, test_dataloader

def get_blind_HP_MultiViewData(path, blind_well1, blind_well2, args):
    """
        获取盲井1数据集的多视角训练集和测试集（一个样本中包含三个尺度的数据，并且只有一个岩性标签），数据加载器

        参数：
        - path: Excel文件的路径（str）
        - blind_well1: 第一个盲井的名称（str）
        - blind_well2: 第二个盲井的名称（str）

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

    # 特征工程（训练集）
    train_data = train_frame.loc[:,
                 ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    # 分离标签得到特征
    X_train = train_data.drop(labels='Facies', axis=1).values

    # 取得标签
    y_train = train_frame['Facies'].values - 1
    y_train = torch.LongTensor(y_train)


    # 特征工程（盲测集）
    blind_data = blind_frame.loc[:,
                 ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    # 分离标签得到特征
    X_blind = blind_data.drop(labels='Facies', axis=1).values

    # 取得标签
    y_blind = blind_frame['Facies'].values - 1
    y_blind = torch.LongTensor(y_blind)



    # 多视角数据生成
    X_multiview_train, y_multiview_train = generate_multiview_blind(X_train, y_train)
    X_multiview_blind, y_multiview_blind = generate_multiview_blind(X_blind, y_blind)

    # 构建原始数据集
    raw_train_dataset = TensorDataset(X_multiview_train[0], y_multiview_train)
    raw_test_dataset = TensorDataset(X_multiview_blind[0], y_multiview_blind)

    # 构建多尺度数据集1
    multi1_train_dataset = MultiScale_Dataset(X_multiview_train[1], y_multiview_train)
    multi1_test_dataset = MultiScale_Dataset(X_multiview_blind[1], y_multiview_blind)

    # 构建多尺度数据集2
    multi2_train_dataset = MultiScale_Dataset(X_multiview_train[2], y_multiview_train)
    multi2_test_dataset = MultiScale_Dataset(X_multiview_blind[2], y_multiview_blind)

    # 组装多视角数据集
    multiview_train_dataset = MultiView_Dataset(raw_train_dataset, multi1_train_dataset, multi2_train_dataset)
    multiview_blind_dataset = MultiView_Dataset(raw_test_dataset, multi1_test_dataset, multi2_test_dataset)

    # 构建数据加载器
    train_dataloader = DataLoader(
        multiview_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        multiview_blind_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return X_multiview_train, X_multiview_blind, train_dataloader, y_multiview_train, y_multiview_blind, test_dataloader

def get_daqing_MultiViewData(path, args):
    # 数据加载与预处理
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    data_frame = data_frame[["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish", "Face"]]


    # 数据分割
    X = data_frame.drop("Face", axis=1).values
    y = data_frame["Face"].values
    y = torch.LongTensor(y)
    # 多视角数据生成
    X_multiview_train, X_multiview_test, y_multiview_train, y_multiview_test = generate_multiview_data(X, y, args.test_size, args.random_state)

    # 构建原始数据集
    raw_train_dataset = TensorDataset(X_multiview_train[0], y_multiview_train)
    raw_test_dataset = TensorDataset(X_multiview_test[0], y_multiview_test)

    # 构建多尺度数据集1
    multi1_train_dataset = MultiScale_Dataset(X_multiview_train[1], y_multiview_train)
    multi1_test_dataset = MultiScale_Dataset(X_multiview_test[1], y_multiview_test)

    # 构建多尺度数据集2
    multi2_train_dataset = MultiScale_Dataset(X_multiview_train[2], y_multiview_train)
    multi2_test_dataset = MultiScale_Dataset(X_multiview_test[2], y_multiview_test)

    # 组装多视角数据集
    multiview_train_dataset = MultiView_Dataset(raw_train_dataset, multi1_train_dataset, multi2_train_dataset)
    multiview_test_dataset = MultiView_Dataset(raw_test_dataset, multi1_test_dataset, multi2_test_dataset)

    # 构建数据加载器
    train_dataloader = DataLoader(
        multiview_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        multiview_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return X_multiview_train, X_multiview_test, train_dataloader, y_multiview_train, y_multiview_test, test_dataloader

def get_blind_daqing_MultiViewData(path1, path2, args):
    """
        获取盲井1数据集的多视角训练集和测试集（一个样本中包含三个尺度的数据，并且只有一个岩性标签），数据加载器

        参数：
        - path: Excel文件的路径（str）
        - blind_well1: 第一个盲井的名称（str）
        - blind_well2: 第二个盲井的名称（str）

        返回：
        - X_train: 训练集特征（numpy array）
        - y_train: 训练集岩性标签（Tensor）
        - train_dataloader: 训练集数据加载器（DataLoader）
        - X_test: 测试集特征（numpy array）
        - y_test: 测试集岩性标签（Tensor）
        - test_dataloader: 测试集数据加载器（DataLoader）
        """

    # 从给定路径读取Excel文件并获取盲测试数据，同时生成多尺度数据集和数据加载器

    # 训练集：排除两个盲井的所有数据
    train_frame = pd.read_excel(path1)
    train_frame.dropna(inplace=True)
    # 盲测试集：使用两个盲井的数据
    blind_frame = pd.read_excel(path2)
    blind_frame.dropna(inplace=True)

    # 特征工程（训练集）
    train_data = train_frame.loc[:,
                 ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish", "Face"]]
    # 分离标签得到特征
    X_train = train_data.drop(labels='Face', axis=1).values

    # 取得标签
    y_train = train_frame['Face'].values
    y_train = torch.LongTensor(y_train)


    # 特征工程（盲测集）
    blind_data = blind_frame.loc[:,
                 ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish", "Face"]]
    # 分离标签得到特征
    X_blind = blind_data.drop(labels='Face', axis=1).values

    # 取得标签
    y_blind = blind_frame['Face'].values
    y_blind = torch.LongTensor(y_blind)



    # 多视角数据生成
    X_multiview_train, y_multiview_train = generate_multiview_blind(X_train, y_train)
    X_multiview_blind, y_multiview_blind = generate_multiview_blind(X_blind, y_blind)

    # 构建原始数据集
    raw_train_dataset = TensorDataset(X_multiview_train[0], y_multiview_train)
    raw_test_dataset = TensorDataset(X_multiview_blind[0], y_multiview_blind)

    # 构建多尺度数据集1
    multi1_train_dataset = MultiScale_Dataset(X_multiview_train[1], y_multiview_train)
    multi1_test_dataset = MultiScale_Dataset(X_multiview_blind[1], y_multiview_blind)

    # 构建多尺度数据集2
    multi2_train_dataset = MultiScale_Dataset(X_multiview_train[2], y_multiview_train)
    multi2_test_dataset = MultiScale_Dataset(X_multiview_blind[2], y_multiview_blind)

    # 组装多视角数据集
    multiview_train_dataset = MultiView_Dataset(raw_train_dataset, multi1_train_dataset, multi2_train_dataset)
    multiview_blind_dataset = MultiView_Dataset(raw_test_dataset, multi1_test_dataset, multi2_test_dataset)

    # 构建数据加载器
    train_dataloader = DataLoader(
        multiview_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        multiview_blind_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return X_multiview_train, X_multiview_blind, train_dataloader, y_multiview_train, y_multiview_blind, test_dataloader