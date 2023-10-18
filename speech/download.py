import gdown
import os

if __name__ == "__main__":
    from argument import args

    print(f"Download model, features, noisy labels at {args.cache_dir}\n")
    os.makedirs(args.cache_dir, exist_ok=True)

    path_model = 'https://drive.google.com/drive/folders/1OkM5YV2zQmoR6ZyLX0tPN9Y0ah_COi2Z?usp=sharing'
    gdown.download_folder(path_model, output=args.cache_dir)
