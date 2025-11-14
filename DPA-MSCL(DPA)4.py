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


from sklearn.metrics import precision_score, recall_score, f1_score

from utils.load_data import get_blind_daqing_MultiViewData
from utils.utils import get_cls_num_list, write_file, get_confusion_matrix, save_metrics_plot, warmup_lr_scheduler
from model.DPA_MSCL_DPA import BCLModel,Classifier
from losses.BCLLoss_DPA import BalSCL
from losses.logitadjustLoss import LogitAdjust
from train.DPA_MSCL_train_DPA import BCL_train, train_model, evaluate



def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="blind_daqing",
                        help='Dataset to use: Hugoton_Panoma or blind_HP or daqing or blind_daqing.')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=890,
                        help='Random seed.')
    parser.add_argument('--random_state', type=int, default=124,
                        help='random_state for Dataset split.')
    parser.add_argument('--epochs1', type=int, default=16,
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
    parser.add_argument('--beta', default=0.20, type=float,
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
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    if args.dataset == "blind_daqing":
        x_train, x_test, data_train_loader, y_train, y_test, test_loader = get_blind_daqing_MultiViewData(args.data_path3, args.data_path4, args)
        args.features = args.features2
        args.num_classes = args.num_classes2
        cls_num_list = get_cls_num_list(y_train, args.num_classes)
        save_path = 'datasave/blind_daqing/'
    else:
        raise ValueError("Invalid dataset name")




    train_loader = data_train_loader
    supcon_model = BCLModel(args.features, args.num_classes)
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
        time1 = time.time()
        loss = BCL_train(train_loader, supcon_model, criterion_ce, criterion_scl, optimizer, args, epoch)
        time2 = time.time()
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/train', loss, epoch)
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
    writer.close()




    times = 1

    train_accs = []
    test_accs = []

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    classifier_model = Classifier(args.num_classes)
    optimizer = optim.Adam(classifier_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
    best_model_path = save_path + f'{args.dataset}_best.pth'
    device = torch.device("cuda:0" if args.cuda else "cpu")
    classifier_model.to(device)

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
    path = save_path + 'y_pre/DPA-BCL(DPA).txt'
    write_file(path, predicted)

    precision = precision_score(y_test.cpu(), predicted.cpu(), average='macro')
    recall = recall_score(y_test.cpu(), predicted.cpu(), average='macro')
    f1 = f1_score(y_test.cpu(), predicted.cpu(), average='macro')
    conf_matrix = get_confusion_matrix(y_test.cpu(), predicted.cpu())

    eval_save_path = save_path + f'model_evaluation/'
    save_metrics_plot(args, accuracy, precision, recall, f1, eval_save_path, "DPA-BCL(DPA)")

    plt.figure(figsize=(10, 7))

    lithology_labels = ['SM', 'DM', 'HS', 'Hgs', 'BS', 'T']

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='.2f',
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

    plt.figure(figsize=(10, 7))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title(f'Accuracy Curves for DPA-BLC(DPA) on {args.dataset}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_curve_save_path = save_path + f'accuracy_curve/'
    plt.savefig(accuracy_curve_save_path + f'DPA-BCL(DPA).png')
    plt.show()


if __name__ == '__main__':
    main()
