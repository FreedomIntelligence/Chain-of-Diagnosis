model_path="01-ai/Yi-34B"
data_path="CoD_en2_Yi-34B_4096_dataset"
accelerate launch --config_file sft.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train.py \
    --experiment_name DiagnosisGPT \
    --model_path  $model_path \
    --max_ckpts 2  \
    --max_seq_len 4096 \
    --gradient_accumulation_steps 2 \
    --data_dir $data_path \
    --output_dir ./ckpts \
    --log_dir ./train_logs \
    --n_epochs 2 \
    --warmup_rates 0.05 \
    --train_bsz_per_gpu 4 \
    --eval_bsz_per_gpu 32 \
    --learning_rate 2e-5 \
    --gradient_checkpointing