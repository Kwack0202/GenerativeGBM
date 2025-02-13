#!/bin/bash
python run.py \
    --task_name simulate \
    --exp_root_path ./datasets/Nasdaq/ \
    --model_type GAN \
    --model_name VanillaGAN \
    --test_start_year 2022 \
    --total_test_months 36 \
    --sliding_test_months 12 \
    --train_months 36 \
    --noise_input_size 3 \
    --noise_output_size 1 \
    --model_optimizer Adam \
    --learning_rate 0.0002 \
    --seq_len 127 \
    --batch_size 32 \
    --num_epochs 500 \
    --num_simulations 10 \
    --num_noise_samples 1
