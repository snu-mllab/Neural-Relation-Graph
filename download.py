import gdown
import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--name', type=str, default='mae_large_49', help='model name')
    args = parser.parse_args()

    folder = '_'.join(args.name.split('_')[1:-1])
    dir_model = os.path.join('./models/mae_ckpt', folder)
    os.makedirs(dir_model, exist_ok=True)

    dir_result = os.path.join('./results', args.name)
    os.makedirs(dir_result, exist_ok=True)

    if args.name == 'mae_large_noise0.08_49':
        path_model = 'https://drive.google.com/drive/folders/1amBa-MJROBoc55xiGeA1-AeHfNIEjjJQ?usp=share_link'
        path_feat = 'https://drive.google.com/drive/folders/1QiA7lkscEOlp75_ZQKm9BM_0lJtPsDjG?usp=share_link'

        print("\nDownload trained model!")
        gdown.download_folder(path_model, output=dir_model)
        print("\nDownload features!")
        gdown.download_folder(path_feat, output=dir_result)

    elif args.name == 'mae_large_49':
        path_model = 'https://drive.google.com/drive/folders/1gCZqgS5UghaFFC9Dfb0-FRFugTMXb_RC?usp=share_link'
        path_feat = 'https://drive.google.com/drive/folders/1VGs_i0UOmTb0lFAwWVXS0IgrXzFWTFT7?usp=share_link'

        print("\nDownload trained model!")
        gdown.download_folder(path_model, output=dir_model)
        print("\nDownload features!")
        gdown.download_folder(path_feat, output=dir_result)
