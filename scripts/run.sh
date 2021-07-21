export CUDA_VISIBLE_DEVICES=$1
export OMP_NUM_THREADS=1

config=yamls/coteaching.yml # vanilla_local.yml
#config=yamls/vanilla.yml
#backbone=Classifier_3L # Classifier_3L  # SimConv4  # Inception  # resnet
gpus_id=0
train_batch_size=$((2048 * 2))
loss_name=elr # elr gce
lr=5e-4
elr_alpha=0 # 1
epochs=100
backbone=inception

for min_inv in .7; do
  for seed in 150; do # {21..40}; do
    suffix=_sd${seed}_bs${train_batch_size}_lr${lr}$2_ep${epochs}_ap${elr_alpha}_${loss_name}_miv${min_inv}

    # shellcheck disable=SC2116
    cmd_tr=$(echo main.py --gpus-id ${gpus_id} --seed ${seed} --lr ${lr} --train-batch-size ${train_batch_size} --loss-name ${loss_name} --n-epochs ${epochs} --config ${config} --exp-suffix ${suffix} --elr_alpha ${elr_alpha} --min-inv ${min_inv} --backbone ${backbone})  #  --num-workers ${num_workers}  --backbone ${backbone}
    echo "${cmd_tr}"
    python ${cmd_tr}
  done
done

#     shellcheck disable=SC2116
#    cmd_t=$(echo main.py --gpus-id ${gpus_id} --seed ${seed} --eval --config ${config} --backbone ${backbone} --exp-suffix ${suffix})
#    echo "${cmd_t}"
#    python ${cmd_t}
