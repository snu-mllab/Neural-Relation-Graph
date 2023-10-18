import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='')
parser.add_argument('--cache_dir', type=str, default='./results')
# Setting
parser.add_argument('-n', '--name', type=str, default='mae_large_noise0.08_49', help='model name')
parser.add_argument('-d', '--dataset', type=str, default='imagenet')
parser.add_argument('--hop', type=int, default=1, help='subsample hop size of target dataset')
# Hyperparameters
parser.add_argument('--kernel',
                    type=str,
                    default='cos_p',
                    choices=['cos_p', 'cos'],
                    help='kernel function type (cos_p: cosine similarity with compatibility term)')
parser.add_argument('--pow', type=int, default=4, help='temperature t')
parser.add_argument('--reg', type=float, default=0.05, help='lambda for noisy set estimation')
# Feature extraction
parser.add_argument('--batch_size', type=int, default=128, help='batch size for feature extraction')
parser.add_argument('--workers', type=int, default=16, help='number of data loader workers')
parser.add_argument('--print_freq',
                    type=int,
                    default=10,
                    help='step size for printing loss, acc, etc.')
# Misc
parser.add_argument('--chunk',
                    type=int,
                    default=250,
                    help='batch size for kernel value calculation (trade-off memory and speed)')
parser.add_argument('--dtype',
                    type=str,
                    default='float32',
                    help='data type for feature extraction',
                    choices=['float32', 'float16'])
parser.add_argument('--verbose', type=str2bool, default=True)
args = parser.parse_args()

if args.name.startswith("mae"):
    args.folder = '_'.join(args.name.split('_')[:-1])
    args.epoch = args.name.split('_')[-1]
else:
    args.folder = args.name
    args.epoch = None

args.cache_dir = os.path.join(args.cache_dir, args.folder)
os.makedirs(args.cache_dir, exist_ok=True)
print(f"Results will be saved at {args.cache_dir}\n")

args.print_freq = int(128 / args.batch_size * 10)
if 'resnet' in args.name:
    args.chunk = min(args.chunk,
                     100)  # smaller chunk for resnet models which have 2048 feature dimension
