# @Time  :2019/3/22
# @Author:langyi

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from args import parse_args
from models.model import get_model
from data import data_loader
from utils import AverageMeter, calculate_accuracy, adjust_learning_rate


def train_model(
        args,
        epoch,
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        vis=None):
    print("Training...")
    model.train()

    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    batch_time = AverageMeter()

    end_time = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        # Compute output and loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Record loss and accuracy
        accuracy = calculate_accuracy(outputs, labels, topk=(1,))
        train_loss.update(loss.item(), inputs.size(0))
        train_accuracy.update(accuracy[0].item(), inputs.size(0))

        # Compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % args.log_interval == 0:
            print('Train Epoch: [{}/{}]([{}/{}])\t'
                  'Loss: {:.4f}({:.4f})\t'
                  'Accuracy: {:.4f}({:.4f})\t'
                  'LR: {}\t'
                  'Batch Time: {:.3f}({:.3f})'.format(
                      epoch, args.epochs, i + 1, len(train_loader),
                      train_loss.val, train_loss.avg,
                      train_accuracy.val, train_accuracy.avg,
                      scheduler.get_lr(),
                      batch_time.val, batch_time.avg
                  ))

        if vis is not None:
            vis.plot('Train Loss', train_loss.avg)
            vis.plot('Train Accuracy', train_accuracy.avg)

    return train_accuracy.avg, train_loss.avg


def val_model(args, epoch, best_result, model, val_loader, criterion, vis):
    print('Waiting Validation...')
    model.eval()
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # # FiveCrop
            # if len(inputs.size()) == 4:
            #     inputs = inputs.unsqueeze(1)
            # bs, ncrops, c, h, w = inputs.size()
            # result = model(inputs.view(-1, c, h, w))  # fuse batch size and ncrops
            # outputs = result.view(bs, ncrops, -1).mean(1)
            # # FiveCrop

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Record loss and accuracy
            accuracy = calculate_accuracy(outputs, labels, topk=(1,))
            val_loss.update(loss.item(), inputs.size(0))
            val_accuracy.update(accuracy[0].item(), inputs.size(0))
    print('Val Epoch: [{}/{}]\t'
          'Loss: {:.4f}\t'
          'Accuracy: {:.4f}'.format(
              epoch, args.epochs,
              val_loss.avg,
              val_accuracy.avg
          ))

    # If val accuracy is better than current best one, instead and save it.
    if val_accuracy.avg > best_result['accuracy']:
        print('=> Saving current best model...\n')
        best_result['accuracy'] = val_accuracy.avg
        best_result['epoch'] = epoch
        save_file_path = os.path.join(
            args.result_path, '{}_{}.pth'.format(
                args.model, epoch))
        torch.save(model.state_dict(), save_file_path)

    return val_accuracy.avg, val_loss.avg


def save_checkpoint(args, epoch, best_result, model, optimizer, scheduler):
    checkpoint_path = os.path.join(
        args.checkpoint_path,
        'checkpoint_{}_{}.pth'.format(
            args.model,
            epoch))
    checkpoint = {
        'epoch': epoch,
        'best_result': best_result,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
    # Load having trained model parameters and then go training
    # Aim to train last epoch for test better result
    # Usually set epochs equals 1

    args = parse_args()
    # Load model without pretrained parameters
    args.scratch = True

    model = get_model(args)
    model.to(args.device)
    # print(model)

    if args.model_path != '':
        if os.path.isfile(args.model_path):
            print('loading model parameters from {}'.format(args.model_path))
            model.load_state_dict(torch.load(args.model_path))
        else:
            print('no found {}'.format(args.model_path))

    # Define loss function 、 optimizer and scheduler to adjust lr
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma)

    best_result = {
        'epoch': 1,
        'accuracy': 0.
    }
    # Load train and val dataset for training
    train_loader = data_loader(args, train=True, val=True)
    for epoch in range(args.begin_epoch, args.epochs + 1):
        adjust_learning_rate(scheduler)
        train_accuracy, train_loss = train_model(
            args, epoch, model, train_loader, criterion, optimizer, scheduler, None)

        print('=> Saving model...\n')
        file_path = os.path.join(
            args.result_path,
            'save_{}_{}.pth'.format(
                args.model, epoch))
        torch.save(model.state_dict(), file_path)
