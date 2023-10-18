# Label error detection on speech classification

## Download model, features, noisy labels
- To download the necessary files, run
```
python download.py
```
- The default save directory is `./results`. You can specify a different directory by using the `--cache_dir` option.
- These files will be downloaded (Total **338MB**):
  - audio_model_epoch25.pth : Trained AST model on the noisy training set. 
  - feat_train_25.pt : Training data features extracted using the model above.
  - target_noisy0.1.pt : Noisy label for ESC-50.

## Run detection
- ESC-50 with synthetic label error (10%) and the AST model
```
python detect.py
```
- Use the identical `--cache_dir` as above.
- For speech data, the default kernel temperature is set to `--pow 8`.
- You can reduce GPU memory usage by half using half precision with `--dtype float16`, with a marginal performance drop.

## Training and feature extraction
- AST GitHub: https://github.com/YuanGongND/ast 
- You can download the ESC-50 dataset from the GitHub repository above.
- For training models, please refer to the GitHub above. We also provide our modified code in `./model`.
- For feature extraction, please refer to `./model/src/feat.py`
