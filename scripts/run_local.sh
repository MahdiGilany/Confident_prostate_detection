export CUDA_VISIBLE_DEVICES=$1
export OMP_NUM_THREADS=1

config=yamls/coteaching_local.yml # vanilla_local.yml
#config=yamls/vanilla.yml
#backbone=Classifier_3L # Classifier_3L  # SimConv4  # Inception  # resnet
gpus_id=0
train_batch_size=$((2048 * 1))
loss_name=elr  # elr gce
lr=1e-4
elr_alpha=0
epochs=100
backbone=inception
min_inv=.4

for min_inv in .4; do #  1e-2; do  # nce_rce
  for seed in 50; do # {21..40}; do
    suffix=_sd${seed}_bs${train_batch_size}_lr${lr}$2_ep${epochs}_ap${elr_alpha}_${loss_name}

    # shellcheck disable=SC2116
    cmd_tr=$(echo main.py --gpus-id ${gpus_id} --seed ${seed} --lr ${lr} --train-batch-size ${train_batch_size} --loss-name ${loss_name} --n-epochs ${epochs} --config ${config} --exp-suffix ${suffix} --elr_alpha ${elr_alpha} --min-inv ${min_inv} --backbone ${backbone})  #  --num-workers ${num_workers}  --backbone ${backbone}
    echo "${cmd_tr}"
    python ${cmd_tr}

    # shellcheck disable=SC2116
    #    cmd_t=$(echo main.py --gpus-id ${gpus_id} --seed ${seed} --eval --config ${config} --backbone ${backbone} --exp-suffix ${suffix} --num-workers ${num_workers})
    #    echo "${cmd_t}"
    #    python ${cmd_t}
  done
done
