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


    for idx, (inputs, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            inputs[0] = inputs[0].cuda()
            inputs[1] = [x.cuda(non_blocking=True) for x in inputs[1]]
            inputs[2] = [x.cuda(non_blocking=True) for x in inputs[2]]
            labels = labels.cuda(non_blocking=True)

        batch_size = labels.shape[0]
        if batch_size == 1:
            continue  # 跳过batch size=1的数据，防止BatchNormalization层报错
        # print(f"Batch size: {batch_size}")

        # compute loss
        z, logits = model(inputs)
        # centers = centers[:args.num_classes]



        scl_loss = criterion_scl(z, labels)
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


def train_model(encoder1, encoder2,  classifier, criterion, optimizer, data_loader, device):
    encoder1.eval()
    encoder2.eval()
    classifier.train()
    for input_data, target in data_loader:
        # print("input_data", input_data)
        # print("target", target)
        batch_size = target.shape[0]
        if batch_size == 1:
            continue  # 跳过batch size=1的数据，防止BatchNormalization层报错
        if device:
            input_data[0] = input_data[0].to(device)
            input_data[1] = [x.to(device) for x in input_data[1]]
            input_data[2] = [x.to(device) for x in input_data[2]]
            target = target.to(device)
        with torch.no_grad():
            features1 = encoder1(input_data[1])
            features2 = encoder2(input_data[2])
            features = torch.cat([features1, features2], dim=1)
        output = classifier(features)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(encoder1, encoder2, classifier, x_data, y_data, device):
    encoder1.eval()
    encoder2.eval()
    classifier.eval()
    with torch.no_grad():
        if device:
            x_data[0] = x_data[0].to(device)
            x_data[1] = [x.to(device) for x in x_data[1]]
            x_data[2] = [x.to(device) for x in x_data[2]]
            y_data = y_data.to(device)

        features1 = encoder1(x_data[1])
        features2 = encoder2(x_data[2])
        features = torch.cat([features1, features2], dim=1)
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
