# @Time  :2019/3/22
# @Author:langyi

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Distracted Driver Detection')

    # Path
    parser.add_argument('--root_path', default='/userhome/project/KaggleDistractedDriver',
                        type=str, help='Project root directory path')
    parser.add_argument('--data_path', default='/userhome/data/state-farm-distracted-driver-detection',
                        type=str, help='Dataset root directory path')
    parser.add_argument('--result_path', default='results',
                        type=str, help='Result Path to be saved')
    parser.add_argument('--checkpoint_path', default='checkpoints',
                        type=str, help='Checkpoint directory path')
    parser.add_argument('--resume_path', default='',
                        type=str, help='Saved model (checkpoint) path of previous training')
    parser.add_argument('--pretrained_models_path', default='pretrained_models',
                        type=str, help='Pretrained models directory path')
    parser.add_argument('--model_path', default='',
                        type=str, help='Load already trained model parameters')


    # I/O
    parser.add_argument('--input_size', default=224,
                        type=int, help='Input size of image')
    parser.add_argument('--num_classes', default=10,
                        type=int, help='Number of classes')

    # batch size and epoch
    parser.add_argument('--batch_size', default=128,
                        type=int, help='Batch size')
    parser.add_argument('--test_batch_size', default=128,
                        type=int, help='Test batch size')
    parser.add_argument('--epochs', default=50,
                        type=int, help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', default=1,
                        type=int, help='Training begins at this epoch')

    # model
    parser.add_argument('--model', default='resnet101',
                        type=str, help='Model')

    # optimizer
    parser.add_argument('--lr', default=0.001,
                        type=float, help='Initial learning rate')
    parser.add_argument('--step_size', default=20,
                        type=int, help='Step size to adjust learning rate')
    parser.add_argument('--gamma', default=0.1,
                        type=float, help='Gamma')
    parser.add_argument('--momentum', default=0.9,
                        type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='Weight decay')

    # scratch
    parser.add_argument('--scratch', default=False,
                        type=bool, help='If True, train dataset from scratch.')

    # device
    parser.add_argument('--device', default='cuda',
                        help='Default use cuda to train')
    parser.add_argument('--num_workers', default=4,
                        type=int, help='Number of threads for multiprocessing')

    # random number seed
    parser.add_argument('--seed', default=1, type=int,
                        help='Set random seed manually')

    # log
    parser.add_argument('--log_interval', default=20,
                        type=int, help='Interval to log training status')
    parser.add_argument('--checkpoint_interval', default=20,
                        type=int, help='Checkpoint interval to save model for keep on training')

    # visdom environment
    parser.add_argument('--close_visdom', action='store_true',
                        help='Not use visdom')
    parser.add_argument('--env', default='default',
                        type=str, help='Default environment for visdom')

    args = parser.parse_args()

    return args
