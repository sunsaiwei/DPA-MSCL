import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from utils.load_data import get_daqing_MultiViewData
from utils.utils import get_cls_num_list, write_file, get_confusion_matrix, save_metrics_plot, warmup_lr_scheduler
from model.DPA_MSCL_model import BCLModel,Classifier
from losses.BCLLoss import BalSCL
from losses.logitadjustLoss import LogitAdjust
from train.DPA_MSCL_train import BCL_train, train_model, evaluate, save_predictions

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


def parse_arguments():
    """
    解析命令行参数，主要用于设置超参数。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='daqing',
                        help='Dataset to use: Hugoton_Panoma or blind_HP or daqing or blind_daqing.')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=3024,
                        help='Random seed.')
    parser.add_argument('--random_state', type=int, default=124,
                        help='random_state for Dataset split.')
    parser.add_argument('--epochs1', type=int, default=300,
                        help='Number of epochs to Contrastive train.')
    parser.add_argument('--epochs2', type=int, default=80,
                        help='Number of epochs to Classification train.')
    parser.add_argument('--lr', type=float, default=0.006,
                        help='Initial learning rate.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR scheduler gamma.')
    parser.add_argument('--step-size', type=int, default=20,
                        help='LR scheduler step size.')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--data_path1', type=str, default="dataset/hp.xlsx",
                        help='Hugoton_Panoma_path.')
    parser.add_argument('--data_path2', type=str, default="dataset/daqing.xlsx",
                        help='part_Daqing_path.')
    parser.add_argument('--data_path3', type=str, default="dataset/daqing_train_data.xlsx",
                        help='daqing_train_data.')
    parser.add_argument('--data_path4', type=str, default="dataset/daqing_blind_data.xlsx",
                        help='daqing_blind_data.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test size for Dataset split.')

    # 添加warm-up相关参数
    parser.add_argument('--warm_epochs', type=int, default=5,
                        help='Number of warm-up epochs')
    parser.add_argument('--warmup_from', type=float, default=0.001,
                        help='Initial learning rate for warm-up')
    parser.add_argument('--warmup_to', type=float, default=0.01,
                        help='Final learning rate after warm-up')

    parser.add_argument('--alpha', default=1.0, type=float,
                        help='cross entropy loss weight')
    parser.add_argument('--beta', default=0.35, type=float,
                        help='supervised contrastive loss weight')
    parser.add_argument('--features1', type=int, default=7,
                        help='feature number for hp.')
    parser.add_argument('--features2', type=int, default=13,
                        help='feature number for daqing.')
    parser.add_argument('--num_classes1', type=int, default=9,
                        help='Number of classes or categories in the  hp dataset')
    parser.add_argument('--num_classes2', type=int, default=6,
                        help='Number of classes or categories in the  daqing dataset')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

args = parse_arguments()

def main():
    # for i in range(300):
    #     if (i > 0):
    #         args.random_state += 1
    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    if args.dataset == "daqing":
        x_train, x_test, data_train_loader, y_train, y_test, test_loader = get_daqing_MultiViewData(
            args.data_path2, args)
        args.features = args.features2
        args.num_classes = args.num_classes2
        cls_num_list = get_cls_num_list(y_train, args.num_classes)
        save_path = 'datasave/daqing/'
    else:
        raise ValueError("Invalid dataset name")

    """
    监督对比学习阶段
    """
    # 设置训练加载器
    train_loader = data_train_loader
    # 初始化模型
    supcon_model = BCLModel(args.features, args.num_classes)
    # 设置设备（GPU或CPU）
    device = torch.device("cuda:0" if args.cuda else "cpu")
    supcon_model.to(device)
    # 初始化优化器
    optimizer = optim.Adam(supcon_model.parameters(), lr=args.warmup_from)
    # 初始化学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # 初始化对比损失函数
    criterion_scl = BalSCL(cls_num_list, args.temp).to(device)
    criterion_ce = LogitAdjust(cls_num_list).to(device)  # 对数调整交叉熵

    # 创建SummaryWriter对象，指定日志保存目录
    tb_path = save_path + 'tensorboard/DPA-BCL'
    writer = SummaryWriter(log_dir=tb_path)

    # 训练对比学习模型
    for epoch in range(1, args.epochs1 + 1):
        # 应用warm-up学习率
        optimizer = warmup_lr_scheduler(args, optimizer, epoch)
        # train for one epoch
        time1 = time.time()
        loss = BCL_train(train_loader, supcon_model, criterion_ce, criterion_scl, optimizer, args, epoch)
        time2 = time.time()
        # 在tensorboard中记录学习率变化
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        # 记录训练损失
        writer.add_scalar('Loss/train', loss, epoch)
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
    # 关闭写入器
    writer.close()

    """
    对比学习后的特征表示进行t-SNE降维可视化
    """

    # 保持数据预处理流程相同（用于保证对比公平性）
    train_loader.dataset.transform = None  # 关闭数据增强
    model = supcon_model.eval()

    # 特征提取函数（需适配监督对比模型）
    def extract_features(model, loader):
        features, labels = [], []
        device = next(model.parameters()).device  # 自动获取模型所在设备
        with torch.no_grad():
            for inputs, targets in loader:
                # 处理多视图输入
                processed_inputs = [
                    inputs[0].to(device),  # 第一个视图（张量）
                    [x.to(device) for x in inputs[1]],  # 第二个视图（列表中的张量）
                    [x.to(device) for x in inputs[2]]  # 第三个视图（列表中的张量）
                ]
                targets = targets.to(device)

                feat, _, _ = model(processed_inputs)  # 使用处理后的输入
                features.append(feat.cpu())
                labels.append(targets.cpu())
        return torch.cat(features), torch.cat(labels)

    # 提取对比学习后的特征
    new_features, new_labels = extract_features(model, train_loader)

    # 替换原有特征处理流程（保持后续代码完全一致）
    # 在标准化前手动压平批次维度（同时保持特征维度）
    features_2d = new_features.view(-1, new_features.shape[-1])  # 自动推导样本总量
    features_daqing = StandardScaler().fit_transform(features_2d.numpy())

    labels_daqing = np.repeat(new_labels.numpy().astype(int), 2, axis=0)
    lithology_info = {
        0: ('SS', '#6a4c93', 'Gravelly Mudstone'),  # 薰衣草紫
        1: ('DS', '#1982c4', 'Gravelly Limestone'),  # 钴蓝色
        2: ('HS', '#8ac926', 'Medium-grained Sandstone'),  # 嫩绿色
        3: ('Hgs', '#ff595e', 'Dolomitic Mudstone'),  # 珊瑚红
        4: ('BS', '#ffca3a', 'Bioclastic Limestone'),  # 芥末黄
        5: ('T', '#a5a5a5', 'tuff')  # 灰色
    }
    
    # ===== 新增类间距计算模块 ===== #
    # 计算类中心
    unique_labels = np.unique(labels_daqing)
    class_centers = {}
    for label in unique_labels:
        mask = labels_daqing == label
        class_centers[label] = labels_daqing[mask].mean(axis=0)

    # 构建距离矩阵
    distance_matrix = np.zeros((len(unique_labels), len(unique_labels)))
    for i in unique_labels:
        for j in unique_labels:
            distance_matrix[i, j] = np.linalg.norm(class_centers[i] - class_centers[j])

    # 统计分析
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    print(f"[类间距离分析]\n"
          f"平均距离: {distance_matrix[triu_indices].mean():.4f}\n"
          f"最小距离: {distance_matrix[triu_indices].min():.4f}\n"
          f"最大距离: {distance_matrix[triu_indices].max():.4f}\n")

    # 可选：可视化距离热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=lithology_info.keys(),
                yticklabels=lithology_info.keys())
    plt.title("Inter-class Distance Matrix")
    plt.show()
    # ===== 结束新增模块 ===== #

    # 保持原有t-SNE参数和可视化代码完全不变
    tsne_daqing = TSNE(
        n_components=3,
        perplexity=40,
        learning_rate=200,
        random_state=args.seed,
    )
    features_daqing_3d = tsne_daqing.fit_transform(features_daqing)

    # 以下保持原始绘图代码完全一致...

    # 4. 增强可视化
    fig = plt.figure(figsize=(9.6, 6.4))
    ax = fig.add_subplot(111, projection='3d')

    # 岩性颜色配置（中等饱和度方案）


    # 保持后续代码完全不变...
    # 创建自定义颜色映射
    cmap = ListedColormap([v[1] for v in lithology_info.values()])

    # 绘制散点图（参数保持不变）
    sc = ax.scatter(
        features_daqing_3d[:, 0],
        features_daqing_3d[:, 1],
        features_daqing_3d[:, 2],
        c=labels_daqing,
        vmin=0,  # ← 新增这个
        vmax=5,  # ← 新增这个
        cmap=cmap,
        alpha=0.65,  # 降低透明度增强层次感
        s=28,  # 减小点尺寸
        edgecolors='none',
        linewidths=0.3,  # 边框粗细
        depthshade=True  # 保持深度阴影
    )

    # 后续所有图例、标签、布局设置保持原样...

    # 单图例系统
    # 独立图例框（右上方垂直排列）

    legend_elements = [plt.Line2D([0], [0],
                                  marker='o',
                                  color='w',
                                  label=v[0],
                                  markerfacecolor=v[1],
                                  markersize=8) for v in lithology_info.values()]

    # 添加图例并设置位置样式
    ax.legend(handles=legend_elements,
              loc='upper right',
              bbox_to_anchor=(0.94, 0.9),
              fontsize=10,  # 调小字体
              frameon=True,
              fancybox=True,
              framealpha=0.7,  # 调低框透明度
              borderpad=0.6,
              edgecolor='#404040')  # 添加边框颜色

    plt.tight_layout()
    plt.show()

    """
    分类训练阶段
    """

    # 初始化训练和测试准确率列表
    train_accs = []
    test_accs = []
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 初始化分类器模型
    classifier_model = Classifier(args.num_classes)
    # 初始化优化器
    optimizer = optim.Adam(classifier_model.parameters(), lr=args.lr)
    # 初始化学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # 初始化交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化最佳准确率和模型保存路径
    best_accuracy = 0
    best_model_path = save_path + f'{args.dataset}_best.pth'
    # 设置设备（GPU或CPU）
    device = torch.device("cuda:0" if args.cuda else "cpu")
    classifier_model.to(device)

    # 训练分类器模型
    for epoch in range(1, args.epochs2 + 1):
        # 训练模型
        train_model(supcon_model.encoder2, supcon_model.encoder3, classifier_model, criterion, optimizer,
                    data_train_loader,
                    device=device)
        # 计算训练准确率
        train_accuracy, _ = evaluate(supcon_model.encoder2, supcon_model.encoder3, classifier_model, x_train,
                                     y_train,
                                     device=device)
        # 计算测试准确率
        test_accuracy, predicted_test = evaluate(supcon_model.encoder2, supcon_model.encoder3, classifier_model,
                                                 x_test,
                                                 y_test,
                                                 device=device)
        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        # 打印训练和测试准确率
        print(
            f'Epoch: {epoch}/{args.epochs2}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # 如果当前测试准确率高于最佳准确率，保存模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(classifier_model.state_dict(), best_model_path)

        # 更新学习率调度器
        scheduler.step()

    # 加载最佳模型进行最终评估
    classifier_model.load_state_dict(torch.load(best_model_path))
    accuracy, predicted = evaluate(supcon_model.encoder2, supcon_model.encoder3, classifier_model, x_test, y_test,
                                   device=device)
    # 保存预测结果到文件
    path = save_path + 'y_pre/DPA-BCL.txt'
    write_file(path, predicted)

    # 计算精确率、召回率和F1分数
    precision = precision_score(y_test.cpu(), predicted.cpu(), average='macro')
    recall = recall_score(y_test.cpu(), predicted.cpu(), average='macro')
    f1 = f1_score(y_test.cpu(), predicted.cpu(), average='macro')
    conf_matrix = get_confusion_matrix(y_test.cpu(), predicted.cpu())

    # 可视化评估结果
    eval_save_path = save_path + f'model_evaluation/'
    save_metrics_plot(args, accuracy, precision, recall, f1, eval_save_path, "DPA-BCL")

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 7))

    # 假设你有一个岩性名称列表，例如：
    lithology_labels = ['SM', 'DM', 'HS', 'Hgs', 'BS', 'T']  # 替换为你的实际岩性名称

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='.2f',  # 显示两位小数
        cmap='YlOrBr',
        xticklabels=lithology_labels,
        yticklabels=lithology_labels
    )
    plt.title(f'Confusion Matrix for DPA-BCL on {args.dataset}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    confusion_matrix_save_path = save_path + f'confusion_matrix/'
    plt.savefig(confusion_matrix_save_path + f'DPA_BCL.png')
    plt.show()

    # 可视化训练和测试准确率
    plt.figure(figsize=(10, 7))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title(f'Accuracy Curves for DPA-BCL on {args.dataset}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_curve_save_path = save_path + f'accuracy_curve/'
    plt.savefig(accuracy_curve_save_path + f'DPA_BCL.png')
    plt.show()


# 如果当前脚本是主程序，运行main函数
if __name__ == '__main__':
    main()
