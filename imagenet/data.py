import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import warnings
import math
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .utils import str_to_interp_mode

#############################################
# Set imagenet directory!
IMGNET_DIR = '/ssd_data/imagenet'
OOD_DIR = '/data_large/readonly/ood_data'
#############################################

warnings.filterwarnings("ignore")


def transforms_imagenet_eval(input_size=224,
                             crop_pct=None,
                             interpolation='bilinear',
                             mean=IMAGENET_DEFAULT_MEAN,
                             std=IMAGENET_DEFAULT_STD,
                             **kwargs):
    DEFAULT_CROP_PCT = 0.875
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    print(f"ImageNet with {input_size}, {crop_pct}, {mean}")

    img_size = input_size
    if isinstance(img_size, (tuple, list)):
        if len(img_size) > 2:
            img_size = img_size[-2:]
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    tfl += [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ]

    return transforms.Compose(tfl)


def load_data(name, model, transform=None):
    traindir = os.path.join(IMGNET_DIR, 'train')
    if name == 'imagenet':
        valdir = os.path.join(IMGNET_DIR, 'val')
    elif name == 'dtd':
        valdir = os.path.join(OOD_DIR, 'dtd/images')
    elif name == 'places':
        valdir = os.path.join(OOD_DIR, 'Places')
    elif name == 'sun':
        valdir = os.path.join(OOD_DIR, 'SUN')
    elif name == 'inat':
        valdir = os.path.join(OOD_DIR, 'iNaturalist')

    if transform == None:
        test_transform = transforms_imagenet_eval(**model.default_cfg)
    else:
        test_transform = transform
    train_transform = test_transform

    trainset = ImageFolder(traindir, train_transform)
    valset = ImageFolder(valdir, test_transform)

    nclass = len(valset.classes)
    print(" # class: ", nclass)
    print(" # train: ", len(trainset.targets))
    print(" # valid: ", len(valset.targets))
    print()

    return trainset, valset


if __name__ == '__main__':
    from models.load import load_model

    model, transform = load_model('mae_vit_large')
    trainset, valset = load_data('imagenet', model, transform)

    print(len(valset))