#!/bin/bash
#SBATCH --job-name=llama3.1_8b_sft
#SBATCH --output=slurm_logs/%j.%x.%N.out
#SBATCH --error=slurm_logs/%j.%x.%N.err
#SBATCH --time=02-00:00:00
#SBATCH --partition=accelerated-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alielganzory@hotmail.com


# Load env vars from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Activate the env
source .venv/bin/activate

# modify the following `MACHINE_RANK`, `MAIN_PROCESS_IP`,
# `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
# `GRADIENT_ACCUMULATION_STEPS` according to your setup
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=1
NUM_PROCESSES=4
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32
srun accelerate launch \
    --mixed_precision bf16 \
    --num_machines $NUM_MACHINES \
    --num_processes $NUM_PROCESSES \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port 29400 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --use_flash_attn \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --use_lora True \
    --lora_rank 64 \
    --lora_alpha 16 \
    --output_dir output/llama3.1_8b_sft \
    --with_tracking \
    --report_to wandb \
    --wandb_project_name "openeurollm" \
    --wandb_entity ali-elganzory-university-of-freiburg \
    --logging_steps 1 \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --chat_template_name tulu \
    --checkpointing_steps 1000 \
    --keep_last_n_checkpoints -1 \
    --save_to_hub False \
    --dataset_mix_dir output/llama3.1_8b_sft \
    --exp_name llama3.1_8b_sft \
    --verbose True \
    --seed 123
    # --cache_dataset_only True \