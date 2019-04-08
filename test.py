# @Time  :2019/3/23
# @Author:langyi

import os
import torch
from args import parse_args
from models.model import get_model
from data import data_loader
from utils import AverageMeter, calculate_accuracy


def test_model(args, model, test_loader):
    print('Wating Test...')
    model.eval()
    test_accuracy = AverageMeter()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # outputs = model(inputs)

            # FiveCrop
            if len(inputs.size()) == 4:
                inputs = inputs.unsqueeze(1)
            bs, ncrops, c, h, w = inputs.size()
            result = model(inputs.view(-1, c, h, w))  # fuse batch size and ncrops
            outputs = result.view(bs, ncrops, -1).mean(1)
            # FiveCrop

            # Record loss and accuracy
            accuracy = calculate_accuracy(outputs, labels, topk=(1,))
            test_accuracy.update(accuracy[0].item(), inputs.size(0))

        print(
            'Test result accuracy is: {:.4f}%'.format(
                test_accuracy.avg
            ))


if __name__ == '__main__':
    # Test model with a particular trained parameters
    args = parse_args()
    args.scratch = True
    assert args.model_path != '', 'Test model path must be not empty'
    if os.path.isfile(args.model_path):
        test_loader = data_loader(args, test=True)
        model = get_model(args)
        model.to(args.device)
        print(model)
        print('loading model parameters from {}'.format(args.model_path))
        model.load_state_dict(torch.load(args.model_path))
        test_model(args, model, test_loader)
    else:
        print('no found {}'.format(args.model_path))
