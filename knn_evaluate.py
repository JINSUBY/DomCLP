from __future__ import print_function
import argparse
import wandb
import os
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from util import set_random_seeds
from datasets.dataset import default_setting, set_train_dataset, set_val_dataset, set_traintest_dataset
from models.resnet_torchvision import DomCLP

wandb.init(mode='offline')

def fix_seed():
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--val_freq', type=int, default=10, help='val frequency')
    parser.add_argument('--save_freq', type=int, default=30, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer', type=str, default='SGD', help='learning rate')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,  help='weight decay')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='pacs',
                        help='pacs')
    parser.add_argument('--train_domain', type=str, default='photo,sketch',
                        help='art_painting, cartoon, photo, sketch')
    parser.add_argument('--val_domain', type=str, default='art_painting,cartoon,photo,sketch',
                        help='all, art_painting, cartoon, photo, sketch')
    parser.add_argument('--data_folder', type=str, default='./data/', help='path to custom dataset')
    parser.add_argument('--method', type=str, default='DomCLP', help='choose method')
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')

    # other setting
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument('--fold', type=str, default='0', help='fold')
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument('--model_pth', type=str, default='last.pth')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--ft_ratio', type=int, default=100, help='')
    parser.add_argument('--subclass', type=int, default=0, help='')
    parser.add_argument('--only_classifier', action='store_true')
    parser.add_argument('--min_scale', type=float, default=0.2)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--augN', type=int, default=5)
    parser.add_argument('--augM', type=int, default=9)
    parser.add_argument('--low-dim', default=128, type=int)
    parser.add_argument('--num_cluster', default='7,14,28', type=str, help='number of clusters')

    args = parser.parse_args()
    args = default_setting(args)

    args.model_path = './results/{}/{}_{}_models'.format(args.dataset, args.method, args.train_domain)
    args.tb_path = './results/{}/{}_{}_tensorboard'.format(args.dataset, args.method, args.train_domain)
    args.model_name = '{}_{}_{}_numclu{}_fold{}_trial{}'. format(args.dataset, args.train_domain, args.method, args.num_cluster, args.fold, args.trial)

    args.save_path = os.path.join(args.model_path, args.model_name)
    os.makedirs(args.save_path, exist_ok=True)
    return args

def extract_features(loader, model):
    model.eval()

    total_features = torch.tensor([])
    total_targets = torch.tensor([])
    total_domains = torch.tensor([])
    total_indexes = torch.tensor([])
    total_indexes2 = torch.tensor([])

    with torch.no_grad():
        for idx, (index, images, targets, domains) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            features, out, h, z = model(images)
            z = z.cpu()
            h = h.cpu()
            total_features = torch.cat((total_features, h))
            total_targets = torch.cat((total_targets, targets))
            total_domains = torch.cat((total_domains, domains))
            total_indexes = torch.cat((total_indexes, index))
            total_indexes2 = torch.cat((total_indexes2, domains*7+targets))

    return total_features, total_targets, total_domains, total_indexes, total_indexes2

def main():
    args = parse_option()
    set_random_seeds(args.seed)

    if args.ckpt == '':
        args.ckpt = os.path.join(args.model_path, args.model_name, args.model_pth)
    else:
        args.ckpt = os.path.join(args.model_path, args.ckpt, args.model_pth)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    args.pretrained = True
    if args.dataset == 'pacs':
        args.pretrained = False
    model = DomCLP(name=args.model, feat_dim=args.low_dim, pretrained=args.pretrained).cuda()

    # build data loader
    args.batch_size = 32
    train_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=args.size),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=args.size),
        transforms.ToTensor(),
    ])
    train_dataset = set_train_dataset(args.data_folder, args.dataset, args.fold, train_transform, transform_call=None, domain=args.train_domain, ft_ratio=args.ft_ratio, subclass=args.subclass)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_workers,
                                               pin_memory=True)

    val_datasets = []
    for domain in args.val_domain:
        if domain in args.train_domain:
            val_dataset = set_val_dataset(args.data_folder, args.dataset, args.fold, val_transform, transform_call=None, domain=domain, ft_ratio=100, subclass=args.subclass)
        else:
            val_dataset = set_traintest_dataset(args.data_folder, args.dataset, args.fold, val_transform, transform_call=None, domain=domain, ft_ratio=100, subclass=args.subclass)
        val_datasets.append(val_dataset)

    val_loaders = []
    for val_dataset in val_datasets:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        val_loaders.append(val_loader)

    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)

    # build model and criterion
    wandb.config.update(args)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train_features, train_targets, train_domains, train_indexes, train_indexes2 = extract_features(train_loader, model)
    for val_loader in val_loaders:
        val_features, val_targets, val_domains, val_indexes, val_indexes2 = extract_features(val_loader, model)
        sim_matrix = torch.matmul(F.normalize(val_features, dim=1), F.normalize(train_features, dim=1).T)
        NN_idx = sim_matrix.argmax(dim=1)
        NN_targets = train_targets[NN_idx]


        correct = (NN_targets == val_targets).sum()
        total = len(NN_targets)
        acc = correct / total
        val_domain = val_loader.dataset.domain
        print(f'{args.train_domain}_{val_domain}_{args.ft_ratio}_acc : {acc:.4f}({correct}/{total})')


if __name__ == '__main__':
    main()
