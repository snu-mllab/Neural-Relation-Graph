import gdown
import os

if __name__ == "__main__":
    from argument import args

    print(f"Download model, features, noisy labels at {args.cache_dir}\n")

    if args.name == 'mae_large_noise0.08_49':
        path_model = 'https://drive.google.com/drive/folders/1hkroHQ4DZwAQ6OBmevljMB1DO3k0cmxv?usp=sharing'
    elif args.name == 'mae_large_49':
        path_model = 'https://drive.google.com/drive/folders/1v-2w2dGRXcldGsMoOK442xBR0-AysKKJ?usp=sharing'
    elif args.name == 'resnet50':
        path_model = 'https://drive.google.com/drive/folders/13LuABuXfWBhFFgIM7wyWQB7jQBtTGiKV?usp=sharing'
    else:
        raise NotImplementedError("Not supported model name")

    gdown.download_folder(path_model, output=args.cache_dir)
