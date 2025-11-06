#!/bin/bash
#SBATCH --job-name=llama3.1_8b_sft__123
#SBATCH --output=slurm_logs/llama3.1_8b_sft__123/%j.%x.%N.out
#SBATCH --error=slurm_logs/llama3.1_8b_sft__123/%j.%x.%N.err
#SBATCH --time=00-01:00:00
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

# modify the following variables according to your setup
SEED=123
TIMESTAMP=$(date +%s)
# EXP_NAME=llama3.1_8b_sft__$SEED__$TIMESTAMP
EXP_NAME=llama3.1_8b_sft__1760768879
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=1
NUM_PROCESSES=4
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32
CHECKPOINTING_STEPS=200
PUSH_TO_HUB=False
DO_NOT_RANDOMIZE_OUTPUT_DIR=True
ADD_SEED_AND_DATE_TO_EXP_NAME=False
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
    --output_dir output/$EXP_NAME \
    --do_not_randomize_output_dir $DO_NOT_RANDOMIZE_OUTPUT_DIR \
    --add_seed_and_date_to_exp_name $ADD_SEED_AND_DATE_TO_EXP_NAME \
    --with_tracking \
    --report_to wandb \
    --wandb_project_name "openeurollm" \
    --wandb_entity ali-elganzory-university-of-freiburg \
    --logging_steps 1 \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --chat_template_name tulu \
    --checkpointing_steps $CHECKPOINTING_STEPS \
    --keep_last_n_checkpoints 1 \
    --clean_checkpoints_at_end False \
    --save_to_hub False \
    --push_to_hub $PUSH_TO_HUB \
    --try_launch_beaker_eval_jobs False \
    --dataset_mix_dir output/$EXP_NAME \
    --exp_name $EXP_NAME \
    --verbose True \
    --seed $SEED
    # --use_lora True \
    # --lora_rank 64 \
    # --lora_alpha 16 \
    # --cache_dataset_only True \