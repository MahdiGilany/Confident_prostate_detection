#!/bin/bash

python test_self_time.py --dataset_name ProstateTeUS --model_name SelfTime --batch_size 512 --num_workers 4 --include_cancer  # --learning_rate_test 1e-3

