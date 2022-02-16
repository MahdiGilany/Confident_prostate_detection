#export CUDA_VISIBLE_DEVICES=$1
#export OMP_NUM_THREADS=1

config=yamls/coteaching_Exact2D_linux.yml

gpus_id=0

for seed in 0 5 10 15 20; do
  suffix=_mskedPatch_Thns/1e-4_fr.4ngrad6_crrt-ds-splt--_res10_srs13_vs.25_ce_sd${seed}_ep200

  # shellcheck disable=SC2116
  cmd_tr=$(echo main.py --gpus-id ${gpus_id} --seed ${seed} --config ${config} --exp-suffix ${suffix})
#  cmd_tr=$(echo ./networks/resnet.py)
  echo "${cmd_tr}"
  python ${cmd_tr}
done