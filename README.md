# Neural-Relation-Graph
Official PyTorch implementation of "[Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data](https://arxiv.org/abs/2301.12321)", published at NeurIPS'23

<p align="center">
   <img src="https://github.com/snu-mllab/Neural-Relation-Graph/blob/main/figure/method.png" align="center" width=70%>
</p>



## Requirements
- PyTorch 1.11 and timm 0.5.4 (check [requirements.txt](https://github.com/snu-mllab/Neural-Relation-Graph/blob/main/requirements.txt))
- Set **IMGNET_DIR** in [```./imagenet/data.py```](https://github.com/snu-mllab/Neural-Relation-Graph/blob/main/imagenet/data.py) that contains ImageNet train and val directories.
- We provide main experiment checkpoints, including data features via [Google Drive](https://drive.google.com/drive/folders/1nFBRYFcEXFhTku0JKllu4S97V_NYAebe?usp=sharing).
- Install gdown
```
   pip uninstall --yes gdown
   pip install gdown -U --no-cache-dir
```

## Notes
- The default save directory is `./results`. You can specify a different directory by using the `--cache_dir` option (for both **download and detection**). 
- We train MAE-Large model for 50 epochs following [official codes](https://github.com/facebookresearch/mae). 
- For ResNet50, we use the model trained by [Timm](https://github.com/huggingface/pytorch-image-models). 
- For detection, you can reduce memory usage by half using half precision with `--dtype float16`, with a marginal performance drop.  Also, using smaller `--chunk` (e.g., 50) reduces the memory usage, while leads to increased computation time.  


## Label error detection 
### IamgeNet with synthetic label error (8%) and MAE-Large
- Download model, features, noisy labels (**6.3GB**):
```
python download.py -n mae_large_noise0.08_49
```
- To conduct detection, run
```
python detect.py -n mae_large_noise0.08_49 --pow 4
```
- The required GPU Memory is approximately **14GB**. You can reduce memory usage by half using half precision with `--dtype float16`, with a marginal performance drop. 

### ImageNet validation set cleaning
- Download model and features (**6.4GB**):
```
python download.py -n mae_large_49
```
- To conduct detection, run
```
python detect_val.py -n mae_large_49 --pow 4
```


## OOD detection
- Download OOD datasets following [this link](https://github.com/deeplearning-wisc/knn-ood). 
- Set **OOD_DIR** in [```./imagenet/data.py```](https://github.com/snu-mllab/Neural-Relation-Graph/blob/main/imagenet/data.py) (to contain dtd, iNaturalist, Places, SUN folders).
- Download model and features (**6.4GB** for MAE-Large / **11GB** for ResNet50): 
```
python download.py -n [mae_large_49/resnet50]
```
- Run,  
```
python detect_ood.py -n [mae_large_49/resnet50] --pow 1
```
- The required GPU Memory is approximately **14GB** for MAE-Large and **18GB** for ResNet50. You can reduce memory usage by half using half precision with `--dtype float16`, with a marginal performance drop. 


## Language and speech datasets
- Check [`./language`](https://github.com/snu-mllab/Neural-Relation-Graph/tree/main/language) and [`./speech`](https://github.com/snu-mllab/Neural-Relation-Graph/tree/main/speech)


## Citation
```
@article{kim2023neural,
  title={Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data},
  author={Kim, Jang-Hyun and Yun, Sangdoo and Song, Hyun Oh},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
