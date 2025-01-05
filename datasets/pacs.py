import os
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torch

class PACS(VisionDataset):
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
    """
    # dataset = "pacs"
    domain_list = ["P", "A", "C", "S"]
    domain2label_dict = {"P":2, "A":0, "C":1 , "S":3}
    label2domain_dict =  {2:"P", 0:"A", 1:"C" , 3:"S"}
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
            txt = open(os.path.join(root, self.dataset, "train_test_list", f"{self.domain}_{self.mode}_fold_{self.fold}.txt"), "r")
        else:
            txt = open(os.path.join(root, self.dataset, "train_test_list", f"{self.domain}_{self.mode}_fold_{self.fold}_ft{self.ft_ratio}.txt"),"r")
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