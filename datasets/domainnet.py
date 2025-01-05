import os
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torch

class DomainNet(VisionDataset):
    """DomainNet.

    Statistics:
        - 6 domains: clipart (48,833), infograph (53,201), painting (75,759), quickdraw (172,500), real (175,327), sketch (70,386)
        - 345 categories:

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
    """

    classes_20_list = ['The_Eiffel_Tower', 'bee', 'bird', 'blueberry', 'broccoli', 'fish', 'flower', 'giraffe', 'grass', 'hamburger', 'hexagon',
                       'horse', 'sun', 'tiger', 'toaster', 'tornado', 'train', 'violin', 'watermelon', 'zigzag']
    classes_40_list = ['arm', 'backpack', 'basket', 'bear', 'beard', 'belt', 'bird', 'book', 'bridge', 'cat',
                       'cookie', 'couch', 'donut', 'drill', 'face', 'fan', 'finger', 'golf club', 'grass', 'helicopter', 'jacket',
                       'key', 'keyboard', 'lighthouse', 'mailbox', 'marker', 'mug', 'pencil', 'pizza', 'potato', 'shoe', 'shovel',
                       'sink', 'skyscraper', 'spoon', 'squirrel', 'sweater', 'telephone', 'tiger', 'train']
    classes_100_list = ['The_Great_Wall_of_China', 'The_Mona_Lisa', 'apple', 'arm', 'asparagus', 'baseball_bat', 'bathtub', 'belt', 'bicycle',
                        'binoculars', 'boomerang', 'bracelet', 'brain', 'bread', 'bucket', 'cake', 'calendar', 'candle', 'cannon', 'carrot', 'cat', 'ceiling_fan',
                        'chair', 'circle', 'crayon', 'crocodile', 'cruise_ship', 'cup', 'dolphin', 'door', 'dragon', 'dresser',
                        'elbow', 'elephant', 'eye', 'fire_hydrant', 'flamingo', 'flashlight', 'flip_flops', 'garden_hose', 'hammer',
                        'harp', 'helmet', 'hot_air_balloon', 'hourglass', 'house', 'kangaroo', 'knife', 'leg', 'light_bulb', 'lighter',
                        'line', 'mailbox', 'marker', 'microwave', 'monkey', 'moon', 'ocean', 'paintbrush', 'piano', 'pickup_truck',
                        'pliers', 'pond', 'pool', 'potato', 'rabbit', 'rain', 'rake', 'rifle', 'river', 'rollerskates', 'sailboat', 'sandwich',
                        'saxophone', 'school_bus', 'see_saw', 'shoe', 'sink', 'snail', 'snorkel', 'snowflake', 'soccer_ball',
                        'sock', 'spoon', 'square', 'string_bean', 'swing_set', 'table', 'telephone', 'tennis_racquet', 'tent', 'toe', 'toothbrush', 'toothpaste',
                        'tree', 'trombone', 'umbrella', 'vase', 'waterslide', 'windmill']

    full_domain = ['photo', 'real', 'sketch', 'cartoon', 'infograph', 'quickdraw']
    domain_list = ["P", "R", "S", "C", "I", "Q"]
    domain2label_dict = {"P":0, "R":1, "S":2 , "C":3, "I":4, "Q":5}
    label2domain_dict =  {0:"P", 1:"R", 2:"S" , 3:"C", 4:"I", 5:"Q"}

    def __init__(self,
                 root: str,
                 dataset: str,
                 ft_ratio: int = 0,
                 mode: str = 'train',
                 domain: Optional[Callable] = None,
                 fold: str = 0,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform = transform, target_transform = target_transform)
        self.dataset = dataset
        self.ft_ratio = ft_ratio
        self.domain = domain
        self.fold = fold
        self.mode = mode # train | test | traintest

        self.data = []
        self.domains = []
        self.targets = []

        if self.ft_ratio == 0:
            txt = open(os.path.join(root, self.dataset, "train_test_list", f"20_{self.domain}_{self.mode}_fold_{self.fold}.txt"), "r")
        else:
            txt = open(os.path.join(root, self.dataset, "train_test_list", f"20_{self.domain}_{self.mode}_fold_{self.fold}_ft{self.ft_ratio}.txt"),"r")
        for line in txt.readlines():
            line = line.strip()
            data, target, domain = line.split(" ")
            self.data.append(os.path.join(root, self.dataset, "images", data))
            self.domains.append(int(domain))
            self.targets.append(int(target))
        txt.close()

        self.domains = torch.tensor(self.domains)
        self.targets = torch.tensor(self.targets)

        unique_labels = torch.unique(self.domains, sorted=True)
        mapping = {label.item(): index for index, label in enumerate(unique_labels)}

        self.domains = torch.tensor([mapping[label.item()] for label in self.domains])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.data[index]).convert("RGB")
        target, domain = self.targets[index], self.domains[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target, domain

    def __len__(self) -> int:
        return len(self.data)