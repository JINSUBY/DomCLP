from torchvision import transforms, datasets
from datasets.randaugment_fixmatch import RandAugmentMC

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TwoTypeMixTransform:
    def __init__(self, strong_transform, weak_transform, K1=2, K2=1):
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform
        self.K1 = K1
        self.K2 = K2

    def __call__(self, x):
        res = [self.strong_transform(x) for i in range(self.K1)] + [self.weak_transform(x) for i in range(self.K2)]
        return res

class TwoCropTransform_2:
    """Create two crops of the same image"""
    def __init__(self, transform_1, transform_2):
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __call__(self, x):
        return [self.transform_1(x), self.transform_2(x)]

class CropTransform_21:
    """Create two crops of the same image"""
    def __init__(self, transform_1, transform_2):
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __call__(self, x):
        return [self.transform_1(x), self.transform_1(x), self.transform_2(x)]

class TwoCropTransform_StyleCL:
    """Create two crops of the same image"""
    def __init__(self, transform, transform_raw):
        self.transform = transform
        self.transform_raw = transform_raw

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.transform_raw(x)]

class TwoCropTransform_Fourier:
    def __init__(self, transform, only_randomresize):
        self.transform = transform
        self.only_randomresize = only_randomresize

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.only_randomresize(x), self.only_randomresize(x)]

def return_train_val_transform(mean, std, size, scale, method):
    normalize = transforms.Normalize(mean=mean, std=std)
    if '_fp' in method:
        method = method.replace('_fp', '')
    if 'ce' == method or 'Finetune' == method or 'tsne' == method or 'knn' == method:
        train_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])
    elif method == 'SimCLR_randaug':
        print("randaug")
        train_transform = TwoCropTransform(transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=scale),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=5, m=10),
            # transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
        ]))
        val_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])

    elif 'SimCLR_StyleCL_PN_lam' == method:
        train_transform = TwoCropTransform_StyleCL(transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        , transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ]))
        val_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])
    elif ('StyleCL_PN' in method and 'SimCLR' not in method) or ('StyleCL_PN_memory' in method and 'SimCLR' not in method):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])
    elif 'SimCLR' in method or 'SupCon' == method or 'ByoL' in method or '230125_a' in method :
        train_transform = TwoCropTransform(transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
        ]))
        val_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])
    elif 'raw' == method:
        train_transform = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                normalize,
            ])
        val_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])
    elif '230124_a' in method or '230124_b' in method or '230124_c' in method or '230125_b' in method:
        train_transform = TwoCropTransform(transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
        val_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])
    elif '221226' in method or '2301' in method:
        train_transform = TwoCropTransform_Fourier(transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
        ]),
        transforms.Compose([transforms.RandomResizedCrop(size=size, scale=scale),
                            transforms.ToTensor()]))
        val_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])
    elif '221226' in method or '2301' in method or '2303' in method:
        train_transform = TwoCropTransform_Fourier(transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
        ]),
        # transforms.Compose([transforms.RandomResizedCrop(size=size, scale=scale),
        transforms.Compose([transforms.Resize(size=size),
                            transforms.ToTensor()]))
        val_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])
    elif '2212' in method:
        """"""
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
        ])
    return train_transform, val_transform