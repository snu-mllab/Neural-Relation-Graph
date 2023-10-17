# Neural-Relation-Graph
Official PyTorch implementation of "[Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data](https://arxiv.org/abs/2301.12321)", published at NeurIPS'23


## Requirements
- PyTorch 1.11 and timm 0.5.4 (check requirements.txt)
- Install gdown
```pip uninstall --yes gdown (if you already installed gdown)
   pip install gdown -U --no-cache-dir
```
- set IMGNET_DIR in ```./imagenet/data.py``` that contains imagenet train and val directories


## Label error detection 
- IamgeNet with synthetic label error (8%) and MAE-Large
``` python download.py -n mae_large_noise0.08_49
    python detect.py -n mae_large_noise0.08_49 --pow 4
```

- ImageNet validation set
``` python download.py -n mae_large_49
    python detect_val.py -n mae_large_49 --pow 4
```


## OOD detection
- Download OOD datasets following https://github.com/deeplearning-wisc/knn-ood 
- Set OOD_DIR in ```./imagenet/data.py``` (a directory that contains dtd, iNaturalist, Places, SUN folders)
- Run  
``` python detect_ood.py -n mar_large_49 --pow 1
```
- You can also test with ```-n resnet50```


## Language and Speech datasets
- Check ```./language``` and ```./speech```


## Citation
```
@inproceedings{kim2023neural,
  title={Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data},
  author={Kim, Jang-Hyun and Yun, Sangdoo and Song, Hyun Oh},
  booktitle={Neural Information Processing Systems},
  year={2023}
}
```