# @Time  :2019/3/22
# @Author:langyi
# Random divide raw dataset into 3 parts, train, val and test set

import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DriverDataset(Dataset):
    def __init__(self, root_path, data_path, transform=None, train=True, val=False, test=False):
        self.root_path = root_path
        self.data_path = data_path
        self.transform = transform
        self.train = train
        self.val = val
        self.test = test
        self.imgs = []
        self.target = []

        seed = 999
        random.seed(seed)

        csv_path = os.path.join(data_path, 'driver_imgs_list.csv')
        data_frame = pd.read_csv(csv_path)
        drivers_labels = data_frame['subject'].drop_duplicates().values
        drivers_labels = random.sample(list(drivers_labels), len(drivers_labels))

        # train : val : test = 7 : 1 : 2
        # train : val : test = 6 : 2 : 2
        len_test = int(0.2 * len(drivers_labels))
        len_val = int(0.1 * len(drivers_labels))
        len_train = len(drivers_labels) - len_test - len_val

        train_labels = drivers_labels[:len_train]
        val_labels = drivers_labels[len_train:(len_train + len_val)]
        test_labels = drivers_labels[-len_test:]

        # record dataset split and its seed
        data_split_path = os.path.join(root_path, 'data', 'data_split.txt')
        with open(data_split_path, 'a') as f:
            f.writelines('Dataset split by seed ' + str(seed) + '\n')
            f.writelines('train set drivers: ' + ','.join(str(x) for x in train_labels) + '\n')
            f.writelines('val set drivers: ' + ','.join(str(x) for x in val_labels) + '\n')
            f.writelines('test set drivers: ' + ','.join(str(x) for x in test_labels) + '\n\n')

        if self.val:  # val set
            labels = val_labels
        elif self.test:  # test set
            labels = test_labels
        else:  # train set
            labels = train_labels

        for label in labels:
            df = data_frame[(data_frame['subject'] == label)]
            for _, row in df.iterrows():
                self.imgs.append(row['img'])
                self.target.append(row['classname'])

        if self.transform is None:
            normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
            if self.val:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
                    # transforms.FiveCrop(224),
                    # transforms.Lambda(
                    #     lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))
                ])
            elif self.test:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    # transforms.ToTensor(),
                    # transforms.Normalize(
                    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    # )
                    transforms.FiveCrop(224),
                    transforms.Lambda(
                        lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.25, 1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        target = self.target[index]
        img_path = os.path.join(self.data_path, 'imgs', 'train', target, img_name)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        target = int(target[-1:])
        return img, target


if __name__ == '__main__':

    driver = DriverDataset(root_path='/userhome/project/DistractedDriverDetection',
                           data_path='/userhome/data/state-farm-distracted-driver-detection')
    print(driver.__len__())
    driver = DriverDataset(root_path='/userhome/project/DistractedDriverDetection',
                           data_path='/userhome/data/state-farm-distracted-driver-detection', val=True)
    print(driver.__len__())
    driver = DriverDataset(root_path='/userhome/project/DistractedDriverDetection',
                           data_path='/userhome/data/state-farm-distracted-driver-detection', test=True)
    print(driver.__len__())
