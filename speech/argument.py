import argparse

parser = argparse.ArgumentParser(description='')
# Directory
parser.add_argument('--cache_dir', type=str, default='/storage/janghyun/results/relation/speech')
# Setting
parser.add_argument('-t', '--task_name', type=str, default='esc50', help='Target task name')
parser.add_argument('-n', '--model_name', type=str, default='ast', help='Target model name')
parser.add_argument('-e', '--epoch', type=int, default=25, help='Trained epoch of target model')
parser.add_argument('--hop', type=int, default=1, help='Subsample hop size of target dataset')
# Hyperparameters
parser.add_argument('--kernel',
                    type=str,
                    default='cos_p',
                    choices=['cos_p', 'cos'],
                    help='kernel function type (cos_p: cosine similarity with compatibility term)')
parser.add_argument('--pow', type=int, default=8, help='temperature of kernel function')
parser.add_argument('--reg', type=float, default=0.05, help='lambda for noisy set estimation')
# Misc
parser.add_argument('--chunk',
                    type=int,
                    default=200,
                    help='batch size for kernel value calculation (trade-off memory and speed)')
parser.add_argument('--dtype',
                    type=str,
                    default='float32',
                    help='data type for feature extraction',
                    choices=['float32', 'float16'])
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()
