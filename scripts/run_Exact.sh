#export CUDA_VISIBLE_DEVICES=$1
#export OMP_NUM_THREADS=1

config=yamls/coteaching_Exact2D.yml

gpus_id=0

for seed in 0 5; do
  suffix=_mskedPatch/crrt-ds-splt--_res10_srs57_vs.25_ce-thr.1-drp.1-30_sd${seed}_invl1val15tst_ep200

  # shellcheck disable=SC2116
  cmd_tr=$(echo main.py --gpus-id ${gpus_id} --seed ${seed} --config ${config} --exp-suffix ${suffix})
#  cmd_tr=$(echo ./networks/resnet.py)
  echo "${cmd_tr}"
  python ${cmd_tr}
done