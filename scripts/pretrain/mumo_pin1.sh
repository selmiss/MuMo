: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}
filename=$(basename "${BASH_SOURCE[0]}" .sh)

# Base config
output_model=${DATA_DIR}/model/pretrain/${filename}

export WANDB_PROJECT="MuMo"
export WANDB_DIR="${output_model}/wandb"
DS_CONFIG=${BASE_DIR}/config/deepspeed/ds_config_zero2.json
MODEL_CONFIG=${BASE_DIR}/config/mumo/config_cls_low.json
BASE_MODEL=${DATA_DIR}/model/pretrain/mumo

# Keep this
SCRIPT_PATH="$(realpath "$0")"
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ${SCRIPT_PATH} ${output_model}
cp ${DS_CONFIG} ${output_model}
cp ${MODEL_CONFIG} ${output_model}/config.json
cp ${BASE_DIR}/train/pretrain.py ${output_model}

export CUDA_HOME=/usr/local/cuda/
# export NCCL_P2P_DISABLE=1

# Deepspeed settings
MASTER_PORT=29504
GPUs=4,5

# Runner
deepspeed --master_port ${MASTER_PORT} --include localhost:${GPUs} ${BASE_DIR}/train/pretrain.py \
    --run_name ${filename}\
    --config_name ${MODEL_CONFIG} \
    --model_name_or_path ${BASE_MODEL} \
    --tokenizer_name ${BASE_DIR}/smiles_tokenizer/mumo_tokenizer \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --model_class MuMoPretrain \
    --ddp_timeout 18000000 \
    --train_files   ${DATA_DIR}/dataset/dk/sft/training_whole1.jsonl \
                    ${DATA_DIR}/dataset/dk/targets/2D_sample_target_200k.jsonl \
    --preprocessing_num_workers 10 \
    --seed 42 \
    --ignore_data_skip true \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 2000 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --max_eval_samples 5000 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 20 \
    --disable_tqdm false \
    --ddp_find_unused_parameters true \
    --overwrite_output_dir \
    --report_to wandb \
    --do_train \
    --do_eval \
    --bf16 True \
    --torch_dtype float32 \
    | tee -a ${output_model}/train.log
    
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
    # --save_steps 1243 \
