import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
import time

import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from utils.load_data import get_Hugoton_Panoma_MultiViewData
from utils.utils import get_cls_num_list, write_file, get_confusion_matrix, save_metrics_plot, warmup_lr_scheduler
from model.DPA_MSCL_model import BCLModel,Classifier
from losses.BCLLoss import BalSCL
from losses.logitadjustLoss import LogitAdjust
from train.DPA_MSCL_train import BCL_train, train_model, evaluate, save_predictions


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hp',
                        help='Dataset to use: Hugoton_Panoma or blind_HP or daqing or blind_daqing.')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=420,
                        help='Random seed.')
    parser.add_argument('--random_state', type=int, default=35,
                        help='random_state for Dataset split.')
    parser.add_argument('--epochs1', type=int, default=100,
                        help='Number of epochs to Contrastive train.')
    parser.add_argument('--epochs2', type=int, default=80,
                        help='Number of epochs to Classification train.')
    parser.add_argument('--lr', type=float, default=0.01,
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
    parser.add_argument('--blind_well1', type=str, default="SHANKLE",
                        help='Well name for blind1 well 1.')
    parser.add_argument('--blind_well2', type=str, default="STUART",
                        help='Well name for blind1 well 2.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

args = parse_arguments()

def main():
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    if args.dataset == "hp":
        x_train, x_test, data_train_loader, y_train, y_test, test_loader = get_Hugoton_Panoma_MultiViewData(args.data_path1, args)
        args.features = args.features1
        args.num_classes = args.num_classes1
        cls_num_list = get_cls_num_list(y_train, args.num_classes)
        save_path = 'datasave/hp/'
    else:
        raise ValueError("Invalid dataset name")




    train_loader = data_train_loader
    supcon_model = BCLModel(args.features, args.num_classes)
    total_params = sum(p.numel() for p in supcon_model.parameters())
    trainable_params = sum(p.numel() for p in supcon_model.parameters() if p.requires_grad)
    print(f"SupCon Model - Total params: {total_params}, Trainable params: {trainable_params}")
    device = torch.device("cuda:0" if args.cuda else "cpu")
    supcon_model.to(device)

    optimizer = optim.Adam(supcon_model.parameters(), lr=args.warmup_from)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion_scl = BalSCL(cls_num_list, args.temp).to(device)
    criterion_ce = LogitAdjust(cls_num_list).to(device)

    tb_path = save_path + 'tensorboard/DPA_BCL'
    writer = SummaryWriter(log_dir=tb_path)

    for epoch in range(1, args.epochs1 + 1):
        optimizer = warmup_lr_scheduler(args, optimizer, epoch)
        # train for one epoch
        time1 = time.time()
        loss = BCL_train(train_loader, supcon_model, criterion_ce, criterion_scl, optimizer, args, epoch)
        time2 = time.time()
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/train', loss, epoch)
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
    writer.close()



    train_loader.dataset.transform = None
    model = supcon_model.eval()

    def extract_features(model, loader):
        features, labels = [], []
        device = next(model.parameters()).device
        with torch.no_grad():
            for inputs, targets in loader:
                processed_inputs = [
                    inputs[0].to(device),
                    [x.to(device) for x in inputs[1]],
                    [x.to(device) for x in inputs[2]]
                ]
                targets = targets.to(device)

                feat, _, _ = model(processed_inputs)
                features.append(feat.cpu())
                labels.append(targets.cpu())
        return torch.cat(features), torch.cat(labels)

    new_features, new_labels = extract_features(model, train_loader)


    features_2d = new_features.view(-1, new_features.shape[-1])
    features_Hugoton_Panoma = StandardScaler().fit_transform(features_2d.numpy())

    labels_Hugoton_Panoma = np.repeat(new_labels.numpy().astype(int), 2, axis=0)


    lithology_info = {
        0: ('SS', '#4E79A7', 'Gravelly Sandstone'),
        1: ('CSiS', '#A0CBE8', 'Coarse Siltstone'),
        2: ('FSiS', '#F28E2B', 'Fine-grained Sandstone'),
        3: ('SiSH', '#E15759', 'Siliceous Shale'),
        4: ('MS', '#76B7B2', 'Marl stone'),
        5: ('WS', '#59A14F', 'Wacke stone'),
        6: ('D', '#EDC948', 'Dolo stone'),
        7: ('PS', '#B07AA1', 'Pack stone'),
        8: ('BS', '#FF9DA7', 'Bound stone')
    }

    unique_labels = np.unique(labels_Hugoton_Panoma)
    class_centers = {}
    for label in unique_labels:
        mask = labels_Hugoton_Panoma == label
        class_centers[label] = features_Hugoton_Panoma[mask].mean(axis=0)

    distance_matrix = np.zeros((len(unique_labels), len(unique_labels)))
    for i in unique_labels:
        for j in unique_labels:
            distance_matrix[i, j] = np.linalg.norm(class_centers[i] - class_centers[j])

    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    print(f"[类间距离分析]\n"
          f"平均距离: {distance_matrix[triu_indices].mean():.4f}\n"
          f"最小距离: {distance_matrix[triu_indices].min():.4f}\n"
          f"最大距离: {distance_matrix[triu_indices].max():.4f}\n")

    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=lithology_info.keys(),
                yticklabels=lithology_info.keys())
    plt.show()

    tsne_Hugoton_Panoma = TSNE(
        n_components=3,
        perplexity=10,
        learning_rate=500,
        random_state=args.seed
    )
    features_Hugoton_Panoma_3d = tsne_Hugoton_Panoma.fit_transform(features_Hugoton_Panoma)


    fig = plt.figure(figsize=(9.6, 6.4))
    ax = fig.add_subplot(111, projection='3d')




    cmap = ListedColormap([v[1] for v in lithology_info.values()])


    sc = ax.scatter(
        features_Hugoton_Panoma_3d[:, 0],
        features_Hugoton_Panoma_3d[:, 1],
        features_Hugoton_Panoma_3d[:, 2],
        c=labels_Hugoton_Panoma,
        vmin=0,
        vmax=8,
        cmap=cmap,
        alpha=0.65,
        s=28,
        edgecolors='none',
        linewidths=0.3,
        depthshade=True
    )



    legend_elements = [plt.Line2D([0], [0],
                                  marker='o',
                                  color='w',
                                  label=v[0],
                                  markerfacecolor=v[1],
                                  markersize=8) for v in lithology_info.values()]

    ax.legend(handles=legend_elements,
              loc='upper right',
              bbox_to_anchor=(0.94, 0.9),
              fontsize=10,
              frameon=True,
              fancybox=True,
              framealpha=0.7,
              borderpad=0.6,
              edgecolor='#404040')

    plt.tight_layout()
    plt.show()



    times = 1

    train_accs = []
    test_accs = []

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    classifier_model = Classifier(args.num_classes)
    total_params = sum(p.numel() for p in classifier_model.parameters())
    trainable_params = sum(p.numel() for p in classifier_model.parameters() if p.requires_grad)
    print(f"Classifier Model - Total params: {total_params}, Trainable params: {trainable_params}")
    optimizer = optim.Adam(classifier_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
    best_model_path = save_path + f'{args.dataset}_best.pth'
    device = torch.device("cuda:0" if args.cuda else "cpu")
    classifier_model.to(device)
    # 评估内存占用
    if torch.cuda.is_available():
        print(f"Classifier模型显存占用量: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        utilization = get_gpu_memory_utilization_rate()
        print(f"Classifier模型显存占用率: {utilization:.2f}%")

    for epoch in range(1, args.epochs2 + 1):
        train_model(supcon_model.encoder2, supcon_model.encoder3, classifier_model, criterion, optimizer,
                    data_train_loader,
                    device=device)
        train_accuracy, _ = evaluate(supcon_model.encoder2, supcon_model.encoder3, classifier_model, x_train, y_train,
                                     device=device)
        test_accuracy, predicted_test = evaluate(supcon_model.encoder2, supcon_model.encoder3, classifier_model, x_test,
                                                 y_test,
                                                 device=device)
        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        print(
            f'Epoch: {epoch}/{args.epochs2}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(classifier_model.state_dict(), best_model_path)

        scheduler.step()

    classifier_model.load_state_dict(torch.load(best_model_path))
    accuracy, predicted = evaluate(supcon_model.encoder2, supcon_model.encoder3, classifier_model, x_test, y_test,
                                   device=device)

    print("\n=== 推理速度评估 ===")
    sample_batch = next(iter(data_train_loader))
    processed_sample = [
        sample_batch[0][0].to(device),
        [x.to(device) for x in sample_batch[0][1]],
        [x.to(device) for x in sample_batch[0][2]]
    ]

    avg_time, std_time, throughput = benchmark_inference_speed(
        supcon_model, processed_sample, device
    )
    print(f"平均推理时间: {avg_time * 1000:.2f} ms")
    print(f"时间标准差: {std_time * 1000:.2f} ms")
    print(f"吞吐量: {throughput:.2f} 样本/秒")

    path = save_path + 'y_pre/DPA_BCL.txt'
    write_file(path, predicted)

    precision = precision_score(y_test.cpu(), predicted.cpu(), average='macro')
    recall = recall_score(y_test.cpu(), predicted.cpu(), average='macro')
    f1 = f1_score(y_test.cpu(), predicted.cpu(), average='macro')
    conf_matrix = get_confusion_matrix(y_test.cpu(), predicted.cpu())

    eval_save_path = save_path + f'model_evaluation/'
    save_metrics_plot(args, accuracy, precision, recall, f1, eval_save_path, "DPA_BCL")

    plt.figure(figsize=(10, 7))

    lithology_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=lithology_labels,
        yticklabels=lithology_labels
    )
    plt.title(f'Confusion Matrix for DPA_BCL on {args.dataset}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    confusion_matrix_save_path = save_path + f'confusion_matrix/'
    plt.savefig(confusion_matrix_save_path + f'DPA_BCL.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title(f'Accuracy Curves for DPA_BCL on {args.dataset}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_curve_save_path = save_path + f'accuracy_curve/'
    plt.savefig(accuracy_curve_save_path + f'DPA_BCL.png')
    plt.show()

def get_gpu_memory_utilization_rate():
    if torch.cuda.is_available():
        used_memory = torch.cuda.memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        utilization_rate = (used_memory / total_memory) * 100
        return utilization_rate
    return 0


def benchmark_inference_speed(model, sample_input, device, num_runs=100):
    model = model.to(device)
    sample_input = [x.to(device) if isinstance(x, torch.Tensor) else
                    [y.to(device) for y in x] for x in sample_input]

    model.eval()

    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            output = model(sample_input)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = 1.0 / avg_time

    return avg_time, std_time, throughput


if __name__ == '__main__':
    main()
