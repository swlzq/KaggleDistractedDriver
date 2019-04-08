# @Author:langyi
# @Time  :2019/3/27

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from args import parse_args
from models.model import get_model
from data import data_loader
from train import train_model, val_model, save_checkpoint
from test import test_model
from utils import Visualize, adjust_learning_rate


# Train, verify and test
def main():
    # Record the best epoch and accuracy
    best_result = {
        'epoch': 1,
        'accuracy': 0.
    }

    args = parse_args()
    # Use model name to name env's
    args.env = args.model
    vis = Visualize(env=args.env) if not args.close_visdom else None

    # Create file to storage result and checkpoint
    if args.root_path != '':
        args.result_path = os.path.join(args.root_path, args.result_path)
        args.checkpoint_path = os.path.join(args.root_path, args.checkpoint_path)
        args.pretrained_models_path = os.path.join(args.root_path, args.pretrained_models_path)
        if not os.path.exists(args.result_path):
            os.mkdir(args.result_path)
        if not os.path.exists(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)
        if not os.path.exists(args.pretrained_models_path):
            os.mkdir(args.pretrained_models_path)
        if args.resume_path:
            args.resume_path = os.path.join(args.checkpoint_path, args.resume_path)

    # Set manual seed to reproduce random value
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create model
    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(args.device)
    model = get_model(args)
    model.to(args.device)
    print(model)

    # Define loss function 、 optimizer and scheduler to adjust lr
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Continue training from checkpoint epoch with checkpoint parameters
    if args.resume_path:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'...".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            args.begin_epoch = checkpoint['epoch'] + 1
            best_result = checkpoint['best_result']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_path))

    # Load dataset
    train_loader = data_loader(args, train=True)
    val_loader = data_loader(args, val=True)
    test_loader = data_loader(args, test=True)
    # Begin to train
    since = time.time()
    for epoch in range(args.begin_epoch, args.epochs + 1):
        adjust_learning_rate(scheduler)
        train_accuracy, train_loss = train_model(args, epoch, model, train_loader, criterion, optimizer, scheduler, vis)
        # Verify accuracy and loss after training
        val_accuracy, val_loss = val_model(args, epoch, best_result, model, val_loader, criterion, vis)

        # Plot train and val's accuracy and loss each epoch
        accuracy = [[train_accuracy], [val_accuracy]]
        loss = [[train_loss], [val_loss]]
        vis.plot2('accuracy', accuracy, ['train', 'val'])
        vis.plot2('loss', loss, ['train', 'val'])

        # Save checkpoint model each checkpoint interval and keep the last one
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            save_checkpoint(args, epoch, best_result, model, optimizer, scheduler)
    # Total time to train
    time_elapsed = time.time() - since
    print('Training complete in {}m {}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Test model with the best val model parameters
    best_model_path = os.path.join(args.result_path, '{}_{}.pth'.format(
        args.model, best_result['epoch']
    ))
    print("Using '{}' for test...".format(best_model_path))
    model.load_state_dict(torch.load(best_model_path))
    test_model(args, model, test_loader)




if __name__ == '__main__':
    main()
