import timm
import warnings
from .mae import load_mae

warnings.filterwarnings("ignore", category=UserWarning)


def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(args, verbose=False):
    transform = None
    if args.name[:3] == 'mae':
        model, transform = load_mae(args, nclass=1000, input_size=224)
    else:
        model = timm.create_model(args.name, pretrained=True)

    if verbose:
        try:
            print(model.default_cfg)
        except:
            pass

        print(f" # Params: {count_param(model)/10**6:.1f}M")

    model.name = args.name
    return model, transform


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--name', type=str, default='mae_large_49')
    parser.add_argument('--cache_dir', type=str, default='./')
    args = parser.parse_args()

    model, transform = load_model(args)

    print(model)
