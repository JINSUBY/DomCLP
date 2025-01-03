# DomCLP: Domain-wise Contrastive Learning with Prototype Mixup for Unsupervised Domain Generalization (AAAI 2025)
### Jin-Seop Lee, Noo-ri Kim, and Jee-Hyong Lee

### [[Project Page](https://jinsuby.github.io/DomCLP/)] [[Paper(arxiv)](https://arxiv.org/abs/2412.09074)]
This repository is the official implementation of **DomCLP: Domain-wise Contrastive Learning with Prototype Mixup for Unsupervised Domain Generalization**.


## Method Overview

In this paper, we propose a novel approach, DomCLP, Domain-wise Contrastive Learning with Prototype Mixup for unsupervised domain generalization. First, we theoretically and experimentally demonstrate that some negative terms in InfoNCE can suppress domainirrelevant common features and amplifies domain-relevant features. Building on this insight, we introduce the Domainwise Contrastive Learning (DCon) to enhance domainirrelevant common features while representation learning. Second, to effectively generalize diverse domain-irrelevant common features across multi-domain, we propose the
Prototype Mixup Learning (PMix). In PMix, to generalize common features from multi-domain, we interpolate common features in each domain utilizing mixup  We extract prototypes of features by k-means clustering, and train the model with mixed prototypes by mixup. It allows the model to effectively learn feature representations for unseen inter-manifold spaces while retaining diverse common feature information. Through our proposed method, DomCLP, the model effectively enhances and generalizes diverse common features. 

## Setup

```shell
git clone https://github.com/JINSUBY/DomCLP
conda env create --file environment.yml
conda activate domclp
```

## Training

### PACS dataset
```shell
# Source Domain : Art painting, Cartoon, Sketch
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=30000 train.py --dataset pacs --method domclp  --fold 0 --train_domain ACS --use_amp
# Source Domain : Photo, Cartoon, Sketch
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=30000 train.py --dataset pacs --method domclp  --fold 0 --train_domain ACS --use_amp
# Source Domain : Photo, Art painting, Sketch
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=30000 train.py --dataset pacs --method domclp  --fold 0 --train_domain ACS --use_amp
# Source Domain : Photo, Art painting, Cartoon
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=30000 train.py --dataset pacs --method domclp  --fold 0 --train_domain ACS --use_amp
```


## Evaluation

### PACS dataset
```shell
# Source Domain : Art painting, Cartoon, Sketch | Target Domain : Photo
## Label Fration 1%
CUDA_VISIBLE_DEVICES=0 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 1 --train_domain ACS --val_domain P
## Label Fration 5%
CUDA_VISIBLE_DEVICES=0 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 5 --train_domain ACS --val_domain P
## Label Fration 10%
CUDA_VISIBLE_DEVICES=0 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 10 --train_domain ACS --val_domain P

# Source Domain : Photo, Cartoon, Sketch | Target Domain : Art painting
## Label Fration 1%
CUDA_VISIBLE_DEVICES=1 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 1 --train_domain PCS --val_domain A
## Label Fration 5%
CUDA_VISIBLE_DEVICES=1 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 5 --train_domain PCS --val_domain A
## Label Fration 10%
CUDA_VISIBLE_DEVICES=1 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 10 --train_domain PCS --val_domain A

# Source Domain : Photo, Art painting, Sketch | Target Domain : Cartoon
## Label Fration 1%
CUDA_VISIBLE_DEVICES=2 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 1 --train_domain PAS --val_domain C
## Label Fration 5%
CUDA_VISIBLE_DEVICES=2 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 5 --train_domain PAS --val_domain C
## Label Fration 10%
CUDA_VISIBLE_DEVICES=2 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 10 --train_domain PAS --val_domain C

# Source Domain : Photo, Art painting, Cartoon | Target Domain : Sketch
## Label Fration 1%
CUDA_VISIBLE_DEVICES=3 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 1 --train_domain PAC --val_domain S
## Label Fration 5%
CUDA_VISIBLE_DEVICES=3 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 5 --train_domain PAC --val_domain S
## Label Fration 10%
CUDA_VISIBLE_DEVICES=3 python knn_evaluate.py --dataset pacs --method domclp --ft_ratio 10 --train_domain PAC --val_domain S
```

## BibTeX
```
@misc{lee2024domclpdomainwisecontrastivelearning,
      title={DomCLP: Domain-wise Contrastive Learning with Prototype Mixup for Unsupervised Domain Generalization}, 
      author={Jin-Seop Lee and Noo-ri Kim and Jee-Hyong Lee},
      year={2024},
      eprint={2412.09074},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.09074}, 
}