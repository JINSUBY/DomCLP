import torch
from torch.utils.data.distributed import DistributedSampler
from .dataset import set_train_dataset, set_val_dataset


def set_train_loader_ddp(args, transform, transform_call=None, drop_last=True):
    # construct data loader
    train_dataset = set_train_dataset(args.data_folder, args.dataset, args.fold, transform, transform_call=transform_call, domain=args.train_domain, ft_ratio=args.ft_ratio, subclass=args.subclass)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=drop_last)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=drop_last)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    return train_loader

def set_train_val_loader_finetune(args, train_transform, valid_transform, transform_call=None):
    # construct data loader
    train_dataset = set_train_dataset(args.data_folder, args.dataset, args.fold, train_transform, transform_call=transform_call, domain=args.train_domain, ft_ratio=args.ft_ratio, subclass=args.subclass)
    train_sampler = None

    ################# weighted Sampler #####################
    new_index = torch.zeros((len(train_dataset.data)))
    for i in range(len(train_dataset.data)):
        new_index[i] = train_dataset.domains[i] * 7 + train_dataset.targets[i]
    train_sampler = create_weighted_sampler_from_new_index(new_index)
    ################# weighted Sampler #####################

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.num_workers,
                                               pin_memory=True, sampler=train_sampler, drop_last=True) # #drop_last

    val_datasets = []
    for domain in args.val_domain:
        if domain in args.train_domain:
            mean_std = True
        else:
            mean_std = False
        val_dataset = set_val_dataset(args.data_folder, args.dataset, args.fold, valid_transform, transform_call=transform_call, mean_std=mean_std, domain=domain, ft_ratio=100, subclass=args.subclass)
        val_datasets.append(val_dataset)

    val_loaders = []
    for val_dataset in val_datasets:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)
            val_loaders.append(val_loader)

    return train_loader, val_loaders