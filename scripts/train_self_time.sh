#!/bin/bash

python train_self_time.py --dataset_name ProstateTeUS --model_name SelfTime --batch_size 512 --num_workers 15 --include_cancer --epochs 45
