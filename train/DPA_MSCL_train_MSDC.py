from __future__ import print_function

import sys
import time
import torch


from utils.utils import AverageMeter



def BCL_train(train_loader, model, criterion_ce, criterion_scl,optimizer, args, epoch ):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (input_list, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_list[0] = torch.cat([input_list[0], input_list[0], input_list[0]], dim=0).float()
        input_list[1] = torch.cat([input_list[1], input_list[1], input_list[1]], dim=0).float()
        input_list[2] = torch.cat([input_list[2], input_list[2], input_list[2]], dim=0).float()
        if torch.cuda.is_available():
            input_list = [x.cuda(non_blocking=True) for x in input_list]
            labels = labels.cuda(non_blocking=True)

        batch_size = labels.shape[0]
        # print(f"Batch size: {batch_size}")

        # compute loss
        feat_mlp, logits, centers = model(input_list)
        centers = centers[:args.num_classes]
        _, z2, z3 = torch.split(feat_mlp, [batch_size,batch_size, batch_size], dim=0)
        features = torch.cat([z2.unsqueeze(1), z3.unsqueeze(1)], dim=1)
        logits, _, _ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
        # print(f'feat_mlp.shape: {feat_mlp.shape}')
        # print(f'centers.shape: {centers.shape}')
        # print(f'features.shape: {features.shape}')
        # print(f'labels.shape: {labels.shape}')

        scl_loss = criterion_scl(centers, features, labels)
        ce_loss = criterion_ce(logits, labels)
        loss = args.alpha * ce_loss + args.beta * scl_loss


        # update metric
        losses.update(loss.item(), batch_size)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def train_model(encoder, classifier, criterion, optimizer, data_loader, device):
    encoder.eval()
    classifier.train()
    for input_data, target in data_loader:
        # print("input_data", input_data)
        # print("target", target)
        # assert target.min() >= 0 and target.max() < 6, \
        #     f"标签越界：需范围[0, 5]，实际范围[{target.min()}, {target.max()}]"
        input_data, target = [x.to(device).float() for x in input_data], target.to(device)
        with torch.no_grad():
            features = encoder(input_data)  # 只获取特征
        output = classifier(features)
        optimizer.zero_grad()
        loss = criterion(output, target)
        #查看output的维度
        loss.backward()
        optimizer.step()

def evaluate(encoder, classifier, x_data, y_data, device):
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        if device:
            x_data, y_data = [x.to(device).float() for x in x_data], y_data.to(device)
        features = encoder(x_data)
        features = features.float()
        output = classifier(features)
        #print(f'outp.shape: {output.shape}')
        _, predicted = torch.max(output, 1)
        #print(f'predicted.shape: {predicted}')
        #print(f'y_data.shape: {y_data}')
        correct = (predicted == y_data).sum().item()
        total = y_data.size(0)
        accuracy = correct / total
    return accuracy, predicted

def save_predictions(predictions, file_path):
    with open(file_path, "w") as file:
        for prediction in predictions:
            file.write(f"{prediction.item()}\n")
