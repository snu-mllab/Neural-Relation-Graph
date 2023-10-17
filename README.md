# Neural-Relation-Graph
Official PyTorch implementation of "[Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data](https://arxiv.org/abs/2301.12321)", published at NeurIPS'23

<p align="center">
   <img src="https://github.com/snu-mllab/Neural-Relation-Graph/blob/main/figure/method.png" align="center" width=70%>
</p>



## Requirements
- PyTorch 1.11 and timm 0.5.4 (check requirements.txt)
- Set **IMGNET_DIR** in [```./imagenet/data.py```](https://github.com/snu-mllab/Neural-Relation-Graph/blob/main/imagenet/data.py) that contains ImageNet train and val directories.

## Download checkpoints
- We provide main experiment checkpoints, including data features via [Google Drive](https://drive.google.com/drive/folders/1nFBRYFcEXFhTku0JKllu4S97V_NYAebe?usp=sharing).
- Install gdown
```
   pip uninstall --yes gdown (# if you already installed gdown)
   pip install gdown -U --no-cache-dir
```


## Label error detection 
- IamgeNet with synthetic label error (8%) and MAE-Large
```
python download.py -n mae_large_noise0.08_49
python detect.py -n mae_large_noise0.08_49 --pow 4
```

- ImageNet validation set
```
python download.py -n mae_large_49
python detect_val.py -n mae_large_49 --pow 4
```


## OOD detection
- Download OOD datasets following https://github.com/deeplearning-wisc/knn-ood 
- Set OOD_DIR in [```./imagenet/data.py```](https://github.com/snu-mllab/Neural-Relation-Graph/blob/main/imagenet/data.py) (to contain dtd, iNaturalist, Places, SUN folders)
- Run  
```
python detect_ood.py -n mar_large_49 --pow 1
```
- You can also test with ```-n resnet50```


## Language and Speech datasets
- Check ```./language``` and ```./speech```


## Citation
```
@article{kim2023neural,
  title={Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data},
  author={Kim, Jang-Hyun and Yun, Sangdoo and Song, Hyun Oh},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
