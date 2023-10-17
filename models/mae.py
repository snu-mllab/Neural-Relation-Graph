# assert timm.__version__ == "0.3.2"  # version check
import torch
import os
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import PIL
from .mae_git import models_vit


def build_transform(is_train, input_size, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=PIL.Image.BICUBIC))
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def load_mae(args, nclass=1000, input_size=224, global_pool=True):
    if "mae_base" in args.name:
        arch = "vit_base_patch16"
    elif "mae_large" in args.name:
        arch = "vit_large_patch16"
    else:
        raise AssertionError("Check mae model args.name!")

    model = models_vit.__dict__[arch](num_classes=nclass, global_pool=global_pool)

    epoch = args.name.split('_')[-1]
    ckpt_path = os.path.join(args.cache_dir, f"checkpoint-{epoch}.pth")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("load ckpt from", ckpt_path)

    transform = build_transform(False, input_size, args=None)

    return model, transform


if __name__ == '__main__':
    name = "mae_vit_base"
    model = load_mae(name)
    print(model)