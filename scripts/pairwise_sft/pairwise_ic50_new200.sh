: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"
export PYTHONPATH=${BASE_DIR}

filename=$(basename "${BASH_SOURCE[0]}" .sh)
MODEL_NAME=mumo_pin1
TASK_NAME=pin1_ic50_new200
MODEL_CLASS=MuMoFinetunePairwise
DATATYPE=pairwise
CONFIG_NAME=${BASE_DIR}/config/mumo/config_cls_low_ic50.json


# Base config
output_model=${DATA_DIR}/model/sft/${MODEL_NAME}/${MODEL_NAME}_${MODEL_CLASS}_${DATATYPE}-${TASK_NAME}_new200

export WANDB_PROJECT="pin1"
export WANDB_DIR="${output_model}/wandb"


BASE_MODEL=${DATA_DIR}/model/sft/mumo_pin1/mumo_pin1_MuMoFinetunePairwise_pairwise-pin1_ic50_e15
DS_CONFIG=${BASE_DIR}/config/deepspeed/ds_config_zero2.json

# Keep
SCRIPT_PATH="$(realpath "$0")"
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ${SCRIPT_PATH} ${output_model}
cp ${DS_CONFIG} ${output_model}

# Runner
WANDB_PROJECT=mumo-pairwise-sft \
deepspeed --master_port 29502 --include localhost:6 ${BASE_DIR}/train/pairwise_sft.py \
    --model_name_or_path ${BASE_MODEL} \
    --config_name ${CONFIG_NAME} \
    --run_name ${filename}\
    --train_files ${DATA_DIR}/dataset/nih/new_ic50/split/train.csv \
    --validation_files ${DATA_DIR}/dataset/nih/new_ic50/split/valid.csv \
    --test_files ${DATA_DIR}/dataset/nih/new_ic50/split/test.csv \
    --data_column_name smiles \
    --label_column_name Log_IC50 \
    --normlization False \
    --model_class ${MODEL_CLASS} \
    --task_type regression \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --train_on_inputs True \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --max_eval_samples 2000 \
    --frozen_layer -2 \
    --learning_rate 5e-6 \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 15 \
    --warmup_steps 100 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
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
    | tee -a ${output_model}/train.log

    # --deepspeed ${DS_CONFIG}
