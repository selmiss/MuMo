: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"
export PYTHONPATH=${BASE_DIR}

filename=$(basename "${BASH_SOURCE[0]}" .sh)
MODEL_NAME=mumo_pin1
TASK_NAME=pin1
MODEL_CLASS=MuMoFinetune
DATATYPE=pe

CONFIG_NAME=${BASE_DIR}/config/mumo/config_cls_low_pe.json


# Base config
output_model=${DATA_DIR}/model/sft/${MODEL_NAME}/${MODEL_NAME}_${MODEL_CLASS}_${DATATYPE}-${TASK_NAME}

export WANDB_PROJECT="MuMo"
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
WANDB_PROJECT=mumo-sft \
deepspeed --master_port 29501 --include localhost:1,2 ${BASE_DIR}/train/finetune.py \
    --run_name ${filename}\
    --model_name_or_path ${BASE_MODEL} \
    --config_name ${CONFIG_NAME} \
    --train_files ${DATA_DIR}/dataset/nih/nih_p_e/train_pin1_pe_1.csv \
    --validation_files ${DATA_DIR}/dataset/nih/nih_p_e/valid_pin1_pe_1.csv \
    --test_files ${DATA_DIR}/dataset/nih/nih_p_e/test_pin1_pe_1.csv \
    --data_column_name smiles \
    --label_column_name P_E \
    --normlization True \
    --model_class ${MODEL_CLASS} \
    --task_type regression \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --train_on_inputs True \
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
