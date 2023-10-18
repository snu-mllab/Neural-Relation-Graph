import gdown
import os

if __name__ == "__main__":
    from argument import args

    path = os.path.join(args.cache_dir, args.task_name)
    print(f"Download model, features, noisy labels at {path}\n")
    os.makedirs(path, exist_ok=True)

    if args.task_name == 'sst2':
        path_model = 'https://drive.google.com/drive/folders/1N-Do1tol7SSOng7Wv2PgCigZSmXLc1aY?usp=sharing'
    elif args.task_name == 'mnli':
        path_model = 'https://drive.google.com/drive/folders/12RIvatQbpzyuowpRuCANeRb-XhwPOcHw?usp=sharing'
    else:
        raise NotImplementedError("Not supported task name")

    gdown.download_folder(path_model, output=path)
