from .pacs import PACS
from .domainnet import DomainNet
import torchvision.transforms as transforms

def set_train_dataset(data_folder, dataset, fold, transform, transform_call=None, transform_2=None, mean_std=True, domain=None, ft_ratio=0, subclass=None):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    if transform_call != None:
        if transform_2 == None:
            transform = transform_call(transforms.Compose([transform]+[normalize]))
        else:
            transform = transform_call(transforms.Compose([transform]+[normalize]), transform_2)
    else:
        transform = transforms.Compose([transform]+[normalize])


    if 'pacs' in dataset:
        train_dataset = PACS(root=data_folder,
                             dataset=dataset,
                             ft_ratio=ft_ratio,
                             mode='train',
                             transform=transform,
                             domain=domain,
                             fold=fold)
    elif 'domainnet' in dataset:
        train_dataset = DomainNet(root=data_folder,
                                  dataset=dataset,
                                  ft_ratio=ft_ratio,
                                  mode='train',
                                  transform=transform,
                                  domain=domain,
                                  fold=fold)
    return train_dataset


def set_val_dataset(data_folder, dataset, fold, transform, transform_call=None, mean_std=True, domain=None, ft_ratio=0, subclass=None):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    if transform_call != None:
        transform = transform_call(transforms.Compose([transform]+[normalize]))
    else:
        transform = transforms.Compose([transform]+[normalize])

    if 'pacs' in dataset:
        val_dataset = PACS(root=data_folder,
                           dataset=dataset,
                           ft_ratio=ft_ratio,
                           mode='test',
                           domain=domain,
                           transform=transform,
                           fold=fold)
    elif 'domainnet' in  dataset:
        val_dataset = DomainNet(root=data_folder,
                                dataset=dataset,
                                ft_ratio=ft_ratio,
                                mode='test',
                                transform=transform,
                                domain=domain,
                                fold=fold)
    return val_dataset


def set_traintest_dataset(data_folder, dataset, fold, transform, transform_call=None, mean_std=True, domain=None, ft_ratio=0, subclass=None):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    if transform_call != None:
        transform = transform_call(transforms.Compose([transform]+[normalize]))
    else:
        transform = transforms.Compose([transform]+[normalize])

    if 'pacs' in dataset:
        val_dataset = PACS(root=data_folder,
                           dataset=dataset,
                           ft_ratio=ft_ratio,
                           mode='traintest',
                           domain=domain,
                           transform=transform,
                           fold=fold)
    elif 'domainnet' in dataset:
        val_dataset = DomainNet(root=data_folder,
                                dataset=dataset,
                                ft_ratio=ft_ratio,
                                mode='traintest',
                                transform=transform,
                                domain=domain,
                                fold=fold)
    else:
        raise ValueError('dataset not supported: {}'.format(dataset))
    return val_dataset

def default_setting(args):
    if 'pacs' in args.dataset:
        args.size = 224
        args.scale = (0.2, 1.)
        args.num_classes = 7
    elif 'domainnet' in args.dataset:
        args.size = 224
        args.scale = (0.2, 1.)
        args.num_classes = 20
    return args
