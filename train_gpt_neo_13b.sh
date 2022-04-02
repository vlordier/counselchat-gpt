#! /bin/bash
./train.py \
    --output_dir $HOME/gpt-busi-1.3b \
    --model_name_or_path EleutherAI/gpt-neo-1.3B \
    --datafile_name busi_data.csv \
    --text_column="Abstract Text" \
    --do_train --do_eval \
    --block_size 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --dataloader_drop_last false \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type="cosine" \
    --warmup_ratio 0.1 \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --push_to_hub="False" \
    --gradient_accumulation_steps 64 \
    --gradient_checkpointing true \
    --fp16 true \
    --report_to="none" \
    --run_name="test_13b" \
    --save_total_limit 1 \
    --save_strategy no

