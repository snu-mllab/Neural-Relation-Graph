# set -x
# comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=esc50
batch_size=48
fstride=10
tstride=10

dataset_mean=-6.6268077
dataset_std=5.358466
audio_length=512

metrics=acc
loss=CE

base_exp_dir=./exp/test-${dataset}-f$fstride-t$tstride-impTrue-aspTrue-b$batch_size-lr1e-5-noise

python ./prep_esc50.py

echo ''
echo 'now process fold'1

exp_dir=${base_exp_dir}/fold1

tr_data=./data/datafiles/esc_train_data_1.json
te_data=./data/datafiles/esc_eval_data_1.json

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/feat.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/esc_class_labels_indices.csv --n_class 50 \
--batch-size $batch_size \
--tstride $tstride --fstride $fstride \
--metrics ${metrics} --loss ${loss} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length}
