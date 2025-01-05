from __future__ import print_function

import os
import sys
import time
import argparse
import wandb
from tqdm import tqdm

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from models.model_utils import save_model

from util import DistributedAverageMeter, fix_random_seeds
from datasets.transforms import TwoCropTransform
from datasets.randaugment import RandAugment_autoaug
from datasets.dataset import default_setting, set_train_dataset
from models.resnet_torchvision import DomCLP
from optimizers.optimizer_utils import set_optimizer
from criterions.supcon import DomCLPLoss

import torch.distributed as dist
import faiss

def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda()
    domains = torch.zeros(len(eval_loader.dataset), dtype=torch.int64).cuda()
    labels = torch.zeros(len(eval_loader.dataset), dtype=torch.int64).cuda()
    indices_count = torch.zeros(len(eval_loader.dataset), dtype=torch.int64).cuda()
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        with torch.no_grad():
            for i, (index, images, labels_, domains_) in enumerate(tqdm(eval_loader)):
                index = index.cuda()
                labels_ = labels_.cuda()
                domains_ = domains_.cuda()
                images = images.cuda(non_blocking=True)
                feat, out, rep, proj = model(images)
                features[index] = proj
                labels[index] = labels_
                domains[index] = domains_
                indices_count[index] += 1
        dist.barrier()
        dist.all_reduce(features, op=dist.ReduceOp.SUM)
        dist.all_reduce(labels, op=dist.ReduceOp.SUM)
        dist.all_reduce(domains, op=dist.ReduceOp.SUM)
        dist.all_reduce(indices_count, op=dist.ReduceOp.SUM)
        bool_idx = torch.where(indices_count != 1)[0]
        if len(bool_idx) > 0:
            features[bool_idx] /= 2
            labels[bool_idx] = (labels[bool_idx] / 2).long()
            domains[bool_idx] = (domains[bool_idx] / 2).long()
            indices_count[bool_idx] -= 1

    return features.cpu(), domains.cpu()

def run_kmeans(x, domains, args):
    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': []}
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        for seed, num_cluster in enumerate(args.num_cluster):
            k = int(num_cluster)
            temp_im2cluster = torch.zeros(len(x), dtype=torch.long).cuda()
            temp_centroids = torch.zeros(len(args.train_domain) * k, args.low_dim).cuda()
            temp_density = torch.zeros(len(args.train_domain) * k).cuda()

            for idx_dom, dom in enumerate(args.train_domain):
                # intialize faiss clustering parameters
                idx = torch.where(domains == idx_dom)[0]
                x_dom = x[idx]
                d = x_dom.shape[1]
                clus = faiss.Clustering(d, k)
                clus.verbose = True
                clus.niter = 20
                clus.nredo = 5
                clus.seed = seed
                clus.max_points_per_centroid = 1000
                clus.min_points_per_centroid = 10

                res = faiss.StandardGpuResources()
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                cfg.device = args.gpu
                index = faiss.GpuIndexFlatL2(res, d, cfg)

                clus.train(x_dom, index)

                D, I = index.search(x_dom, 1)  # for each sample, find cluster distance and assignments
                im2cluster = [int(n[0]) for n in I]

                # get cluster centroids
                centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

                # sample-to-centroid distances for each cluster
                Dcluster = [[] for c in range(k)]
                for im, i in enumerate(im2cluster):
                    Dcluster[i].append(D[im][0])

                # concentration estimation (phi)
                density = np.zeros(k)
                for i, dist in enumerate(Dcluster):
                    if len(dist) > 1:
                        d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                        density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
                dmax = density.max()
                for i, dist in enumerate(Dcluster):
                    if len(dist) <= 1:
                        density[i] = dmax

                density = density.clip(np.percentile(density, 10),
                                       np.percentile(density, 90))  # clamp extreme values for stability
                density = args.temperature * density / density.mean()  # scale the mean to temperature

                # convert to cuda Tensors for broadcast
                centroids = torch.Tensor(centroids).cuda()
                centroids = nn.functional.normalize(centroids, p=2, dim=1)

                im2cluster = [k * idx_dom + x for x in im2cluster]
                im2cluster = torch.LongTensor(im2cluster).cuda()
                density = torch.Tensor(density).cuda()

                temp_im2cluster[idx] = im2cluster
                temp_centroids[torch.arange(k * idx_dom, k * (idx_dom + 1))] = centroids
                temp_density[torch.arange(k * idx_dom, k * (idx_dom + 1))] = density

            results['im2cluster'].append(temp_im2cluster)
            results['centroids'].append(temp_centroids)
            results['density'].append(temp_density)

    return results

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=200, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help="mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel",)
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer', type=str, default='Adam', help='learning rate')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18', help='resnet18')
    parser.add_argument('--dataset', type=str, default='pacs', help='pacs, domainnet')
    parser.add_argument('--train_domain', type=str, default='ACS', help='ACS, PCS, PAS, PAC')
    parser.add_argument('--data_folder', type=str, default='./data/', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='DomCLP', help='choose method')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')

    # other setting
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument('--fold', type=str, default='0', help='fold')
    parser.add_argument('--ckpt', type=str, default='', help='')
    parser.add_argument('--ft_ratio', type=int, default=0, help='')
    parser.add_argument('--subclass', type=int, default=0, help='')
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument('--device', type=str)
    parser.add_argument('--min_scale', type=float, default=0.2)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'])
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--port', type=int, default=2022)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--dist_url', type=str)
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--augM', type=int, default=9)
    parser.add_argument('--augN', type=int, default=5)
    parser.add_argument('--temperature', default=0.2, type=float, help='softmax temperature')
    parser.add_argument('--num_cluster', default='7,14,28', type=str, help='number of clusters')
    parser.add_argument('--low-dim', default=128, type=int)
    parser.add_argument('--beta', default=4, type=float, help='beta scheduler')

    args = parser.parse_args()
    args = default_setting(args)

    if args.train_domain != '':
        args.train_domain = args.train_domain
        args.num_domain = len(args.train_domain)

    args.model_path = './results/{}/{}_{}_models'.format(args.dataset, args.method, args.train_domain)
    args.tb_path = './results/{}/{}_{}_tensorboard'.format(args.dataset, args.method, args.train_domain)
    args.model_name = '{}_{}_{}_numclu{}_fold{}_trial{}'. format(args.dataset, args.train_domain, args.method, args.num_cluster, args.fold, args.trial)

    args.num_cluster = args.num_cluster.split(',')

    args.tb_folder = os.path.join(args.tb_path, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder, exist_ok=True)
    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)
    return args

def train(train_loader, model, criterion, criterion_pcl, optimizer, epoch, args, cluster_result, domains_all):
    batch_time = DistributedAverageMeter()
    data_time = DistributedAverageMeter()
    losses_self = DistributedAverageMeter()
    losses_proto = DistributedAverageMeter()
    losses_mixup = DistributedAverageMeter()
    model.train()

    end = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    for idx, (index, images, labels, domains) in enumerate(train_loader):

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            data_time.update(time.time() - end)

            idx_mixup= torch.randperm(images[0].size()[0])
            l = np.random.beta(args.beta, args.beta)
            l = max(l, 1 - l)
            input_a, input_b = images[0], images[0][idx_mixup]
            mixed_input = l * input_a + (1 - l) * input_b

            images = torch.cat([images[0], images[1], mixed_input], dim=0)
            images = images.to(args.gpu)
            labels = labels.to(args.gpu)
            domains = domains.to(args.gpu)
            bsz = labels.shape[0]

            features, out, h, z = model(images)
            z1, z2, z_mixup = torch.split(z, [bsz, bsz, bsz], dim=0)
            z_self = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
            loss_self = criterion(z_self, domains)

            target_proto = []
            output_proto = []
            for n, (k, im2cluster, prototypes, density) in enumerate(zip(args.num_cluster, cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):
                k = int(k)
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]
                neg_proto_id = torch.tensor([list(set(range(k*dom, k*(dom+1))) - {idx.item()}) for idx, dom in zip(pos_proto_id, domains_all[index])]).cuda()
                neg_prototypes = prototypes[neg_proto_id]
                proto_selected = torch.cat([pos_prototypes.unsqueeze(1), neg_prototypes], dim=1)
                logits_proto = torch.einsum('ij,ikj->ik', z1, proto_selected)  ### [256, 14]
                labels_proto = torch.zeros(len(pos_proto_id)).long().cuda()
                temp_proto = density[torch.cat((pos_proto_id.unsqueeze(1), neg_proto_id), dim=1)]
                logits_proto /= temp_proto

                target_proto.append(labels_proto)
                output_proto.append(logits_proto)

            loss_proto = torch.tensor(0.).to(args.gpu)
            for proto_out, proto_target in zip(output_proto, target_proto):
                loss_proto += criterion_pcl(proto_out, proto_target)

            loss_proto /= len(args.num_cluster)

            loss_mixup = torch.tensor(0.).to(args.gpu)
            for n, (k, im2cluster, prototypes, density) in enumerate(zip(args.num_cluster, cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):
                k = int(k)
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]
                proto_a, proto_b = pos_prototypes, pos_prototypes[idx_mixup]
                mixed_target = l * proto_a + (1 - l) * proto_b
                loss_mixup_temp = - F.cosine_similarity(z_mixup, mixed_target).mean()
                loss_mixup += loss_mixup_temp

            loss_mixup /= len(args.num_cluster)
            loss = loss_self + loss_proto + loss_mixup

        losses_self.update(loss_self.item(), bsz)
        losses_proto.update(loss_proto.item(), bsz)
        losses_mixup.update(loss_mixup.item(), bsz)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        batch_time.synchronize_between_processes()
        data_time.synchronize_between_processes()
        losses_self.synchronize_between_processes()
        losses_proto.synchronize_between_processes()
        losses_mixup.synchronize_between_processes()

        if (idx + 1) % args.print_freq == 0 or (idx + 1) == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss_self {loss_self.val:.3f} ({loss_self.avg:.3f})\t'
                  'loss_proto {loss_proto.val:.3f} ({loss_proto.avg:.3f})\t'
                  'loss_mixup {loss_mixup.val:.3f} ({loss_mixup.avg:.3f})\t'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_self=losses_self, loss_proto=losses_proto, loss_mixup=losses_mixup))
            sys.stdout.flush()

    torch.cuda.empty_cache()
    return losses_self.avg, losses_proto.avg, losses_mixup.avg

def init_for_distributed(args):
    args.distributed = True
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, 'env://'), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)

    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main():
    args = parse_option()
    fix_random_seeds(args.seed)

    args.num_workers = len(args.gpu_ids) * 4
    init_for_distributed(args)
    local_gpu_id = args.gpu
    args.batch_size = args.batch_size // args.world_size

    # build model and criterion
    args.pretrained = True
    if args.dataset == 'pacs':
        args.pretrained = False
    model = DomCLP(name=args.model, feat_dim=args.low_dim, pretrained=args.pretrained).cuda(local_gpu_id)
    criterion =  DomCLPLoss(temperature=args.temp).to(local_gpu_id)
    criterion_pcl = nn.CrossEntropyLoss().to(local_gpu_id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_gpu_id])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.size, scale=args.scale),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        RandAugment_autoaug(args.augN, args.augM),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=args.size),
        transforms.ToTensor(),
    ])

    # build data loader
    train_dataset = set_train_dataset(args.data_folder, args.dataset, args.fold, train_transform,
                                      transform_call=TwoCropTransform, domain=args.train_domain, ft_ratio=args.ft_ratio,
                                      subclass=args.subclass)
    eval_dataset = set_train_dataset(args.data_folder, args.dataset, args.fold, eval_transform, transform_call=None,
                                      domain=args.train_domain, ft_ratio=args.ft_ratio, subclass=args.subclass)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=eval_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # build optimizer
    optimizer = set_optimizer(args, model)
    if args.optimizer == 'SGD':
        scheduler = None
    if args.optimizer == 'Adam' or args.optimizer == 'LARS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # wandb
    if args.no_wandb:
        wandb.init(mode='offline')
    else:
        if args.gpu == 0:
            wandb.init(project='MDSSL_2024CVPR', name=args.model_name, group=args.dataset)
            wandb.config.update(args)

    # training routine
    for epoch in range(1, args.epochs + 1):
        cluster_result = None
        time1 = time.time()
        features, domains = compute_features(eval_loader, model, args)

        # placeholder for clustering result
        cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
        for num_cluster in args.num_cluster:
            cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset), dtype=torch.long).cuda())
            cluster_result['centroids'].append(torch.zeros(int(num_cluster) * len(args.train_domain), args.low_dim).cuda())
            cluster_result['density'].append(torch.zeros(int(num_cluster) * len(args.train_domain)).cuda())

        if args.gpu == 0:
            features = features.numpy()
            cluster_result = run_kmeans(features, domains, args)  # run kmeans clustering on master node

        dist.barrier()
        for k, data_list in cluster_result.items():
            for data_tensor in data_list:
                dist.broadcast(data_tensor, 0, async_op=False)
        time2 = time.time()
        print('epoch {}, kmeans clustering time {:.2f}'.format(epoch, time2 - time1))
        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        time1 = time.time()
        loss_self, loss_pcl, loss_mixup = train(train_loader, model, criterion, criterion_pcl, optimizer, epoch, args, cluster_result, domains)
        if scheduler is not None and epoch > 100:
            scheduler.step()
        time2 = time.time()
        print('epoch {}, train time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss_self', loss_self, epoch)
        logger.log_value('loss_pcl', loss_pcl, epoch)
        logger.log_value('loss_mixup', loss_mixup, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # wandb log
        if args.gpu == 0:
            wandb.log({"loss_self" : loss_self})
            wandb.log({"loss_pcl" : loss_pcl})
            wandb.log({"loss_mixup" : loss_mixup})
            wandb.log({"learning_rate" : optimizer.param_groups[0]['lr']})
            # Optional
            wandb.watch(model)

        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

    # save the last model
    save_file = os.path.join(args.save_folder, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        correct = output.eq(target)
        acc = correct.sum() / correct.shape[0]
        return acc

def accuracy_and_correct(output, target):
    """Computes the accuracy and correct predictions count over the k top predictions"""
    with torch.no_grad():
        correct = output.eq(target)
        acc = correct.sum() / correct.shape[0]
        return acc, correct.sum()

if __name__ == '__main__':
    main()
