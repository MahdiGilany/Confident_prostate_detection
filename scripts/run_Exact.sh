export CUDA_VISIBLE_DEVICES=$1
export OMP_NUM_THREADS=1

config=yamls/coteaching_Exact1D.yml

gpus_id=0
epochs=80
seed=0

for threshold in .5; do
  suffix=s/lr1e-3_fr.1_adamw_str2_krnl15_normalized

  # shellcheck disable=SC2116
  cmd_tr=$(echo main.py --gpus-id ${gpus_id} --seed ${seed} --n-epochs ${epochs} --config ${config} --exp-suffix ${suffix} --core_th ${threshold})
  echo "${cmd_tr}"
  python ${cmd_tr}
done
