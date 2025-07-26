BASE_DIR=xxx # Change to your project dir
DATA_DIR=xxx # Change to your data file dir
export PYTHONPATH=${BASE_DIR}

filename=$(basename "${BASH_SOURCE[0]}" .sh)
MODEL_NAME=$1
TASK_NAME=lipo
DATATYPE=$5
MODEL_CLASS=$4
CONFIG_NAME=${BASE_DIR}/config/mumo/config_cls_reg.json
for i in {1..3}
do
    echo $i
    # Base config
    output_model=${DATA_DIR}/model/sft/${MODEL_NAME}/${MODEL_NAME}_${MODEL_CLASS}_${DATATYPE}-${TASK_NAME}_${i}

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
    deepspeed --master_port $2 --include localhost:$3 ${BASE_DIR}/train/finetune.py \
    --run_name ${filename} \
        --model_name_or_path ${BASE_MODEL} \
        --fp_tokenizer_path ${fp_tokenizer_path} \
        --config_name ${CONFIG_NAME} \
        --functional_group False \
        --train_files ${DATA_DIR}/dataset/${DATATYPE}/${TASK_NAME}_${i}/raw/train_${TASK_NAME}_${i}.csv \
        --validation_files ${DATA_DIR}/dataset/${DATATYPE}/${TASK_NAME}_${i}/raw/test_${TASK_NAME}_${i}.csv \
        --test_files ${DATA_DIR}/dataset/${DATATYPE}/${TASK_NAME}_${i}/raw/test_${TASK_NAME}_${i}.csv \
        --data_column_name smiles \
        --label_column_name lipo \
        --normlization True \
        --model_class ${MODEL_CLASS} \
        --task_type regression \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 10 \
        --train_on_inputs True \
        --do_train \
        --do_eval \
        --use_fast_tokenizer false \
        --output_dir ${output_model} \
        --max_eval_samples 1000 \
        --frozen_layer -2 \
        --learning_rate 3e-5 \
        --lr_scheduler_type linear \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 20 \
        --warmup_steps 10 \
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
done