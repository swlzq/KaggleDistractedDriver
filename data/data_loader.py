# @Author:langyi
# @Time  :2019/4/7

from .dataset import DriverDataset
from torch.utils.data import DataLoader


# This function is to load different dataset, i.e. train, val and test set
def data_loader(args, train=False, val=False, test=False):
    if train and val:  # Return train and val set for training last epoch to test
        dataset1 = DriverDataset(root_path=args.root_path, data_path=args.data_path, train=True)
        dataset2 = DriverDataset(root_path=args.root_path, data_path=args.data_path, val=True)
        dataset = dataset1 + dataset2
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
    elif val:
        dataset = DriverDataset(root_path=args.root_path, data_path=args.data_path, val=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    elif test:
        dataset = DriverDataset(root_path=args.root_path, data_path=args.data_path, test=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        dataset = DriverDataset(root_path=args.root_path, data_path=args.data_path, train=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
    return dataloader
