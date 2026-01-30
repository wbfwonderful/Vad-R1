cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export WANDB_MODE="offline"

# Qwen/Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo.py \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --save_steps 1000 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_train_epochs 1 \
    --run_name Qwen2_5_VL_7B_RL \
    --output_dir "/XX/Qwen2_5_VL_7B_RL" \
    --model_name_or_path '/XX/Qwen2_5_VL_7B_SFT' \
    --dataset_name "/XX/Vad-Reasoning-RL.jsonl" \
    --deepspeed local_scripts/zero3.json \
    --anomaly_reward true \
    --temporal false \
    --len_control true \
    --num_generations 4  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
