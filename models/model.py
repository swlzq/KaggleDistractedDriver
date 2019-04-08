# @Time  :2019/3/22
# @Author:langyi

import os
import torch
import torch.nn as nn
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.alexnet import alexnet


def get_model(args, model_path=None):
    """

    :param args: super arguments
    :param model_path: if not None, load already trained model parameters.
    :return: model
    """
    if args.scratch:  # train model from scratch
        pretrained = False
        model_dir = None
        print("=> Loading model '{}' from scratch...".format(args.model))
    else:  # train model with pretrained model
        pretrained = True
        model_dir = os.path.join(args.root_path, args.pretrained_models_path)
        print("=> Loading pretrained model '{}'...".format(args.model))

    if args.model.startswith('resnet'):

        if args.model == 'resnet18':
            model = resnet18(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'resnet34':
            model = resnet34(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'resnet50':
            model = resnet50(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'resnet101':
            model = resnet101(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'resnet152':
            model = resnet152(pretrained=pretrained, model_dir=model_dir)

        model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    elif args.model.startswith('vgg'):
        if args.model == 'vgg11':
            model = vgg11(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'vgg11_bn':
            model = vgg11_bn(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'vgg13':
            model = vgg13(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'vgg13_bn':
            model = vgg13_bn(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'vgg16':
            model = vgg16(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'vgg16_bn':
            model = vgg16_bn(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'vgg19':
            model = vgg19(pretrained=pretrained, model_dir=model_dir)
        elif args.model == 'vgg19_bn':
            model = vgg19_bn(pretrained=pretrained, model_dir=model_dir)

        model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.num_classes)

    elif args.model == 'alexnet':
        model = alexnet(pretrained=pretrained, model_dir=model_dir)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.num_classes)

    # Load already trained model parameters and go on training
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

    return model
