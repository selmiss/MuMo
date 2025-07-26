: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"
export PYTHONPATH=${BASE_DIR}

filename=$(basename "${BASH_SOURCE[0]}" .sh)
MODEL_NAME=mumo
TASK_NAME=xx
DATATYPE=xx
MODEL_CLASS=MuMoFinetune
CONFIG_NAME=${BASE_DIR}/config/mumo/config_cls_reg.json

# Base config
output_model=xxx/model/sft/mumo/dk

export WANDB_PROJECT="NeurIPS_Rebuttal"
export WANDB_DIR="${output_model}/wandb"
BASE_MODEL=${DATA_DIR}/model/pretrain/${MODEL_NAME}
DS_CONFIG=${BASE_DIR}/config/deepspeed/ds_config_zero2.json

# Keep
SCRIPT_PATH="$(realpath "$0")"
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ${SCRIPT_PATH} ${output_model}
cp ${DS_CONFIG} ${output_model}

# Runner
deepspeed --master_port 29500 --include localhost:0 ${BASE_DIR}/train/finetune.py \
    --run_name ${filename} \
    --model_name_or_path ${BASE_MODEL} \
    --config_name ${CONFIG_NAME} \
    --train_files xxx/Nips/dataset/dk/training_10%.csv \
    --validation_files xxx/Nips/dataset/dk/validation_10%.csv \
    --test_files xxx/Nips/dataset/dk/testing_10%.csv \
    --data_column_name smiles \
    --label_column_name label \
    --normlization True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --train_on_inputs True \
    --model_class ${MODEL_CLASS} \
    --task_type regression \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --max_eval_samples 1000 \
    --frozen_layer -2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --warmup_steps 50 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 20 \
    --save_strategy no \
    --preprocessing_num_workers 10 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --seed 42 \
    --disable_tqdm false \
    --block_size 1024 \
    --report_to wandb \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --bf16 False \
    --torch_dtype float32 \
    --deepspeed ${DS_CONFIG} \
    | tee -a ${output_model}/train.log
