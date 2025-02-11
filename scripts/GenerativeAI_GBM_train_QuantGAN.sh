#!/bin/bash
python run.py \
    --task_name train \
    --exp_root_path ./datasets/Nasdaq/ \
    --model_type GAN \
    --model_name QuantGAN \
    --test_start_year 2022 \
    --total_test_months 36 \
    --sliding_test_months 12 \
    --train_months 36 \
    --noise_input_size 3 \
    --noise_output_size 1 \
    --model_optimizer Adam \
    --learning_rate 0.0002 \
    --seq_len 127 \
    --batch_size 64 \
    --num_workers 0 \
    --num_epochs 3000 \
    --min_epochs 500 \
    --check_interval 10 \
    --ks_threshold 0.05 \
    --pvalue_threshold 0.05 \
    --fake_sample 10 \
    --confidence 0.8 \
    --coverage_threshold 0.8 \
    --loss_tolerance 0.0001 \
    --early_stop_patience 10 \
    --num_simulations 10 \
    --num_noise_samples 1
