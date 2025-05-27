BASE_DIR=xxx # Change to your project dir
DATA_DIR=xxx/model # Change to your data file dir
export PYTHONPATH=${BASE_DIR}
MODEL_NAME=mumo
MODEL_CLASS=MuMoFinetune
CONFIG_NAME=${BASE_DIR}/config/mamba/config_cls_reg.json

# Base config
output_model=xxx/model/sft/mumo/dk_test
BASE_MODEL=xxx/model/sft/mumo/dk
DS_CONFIG=${BASE_DIR}/config/deepspeed/ds_config_zero2.json

# Keep
SCRIPT_PATH="$(realpath "$0")"
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ${SCRIPT_PATH} ${output_model}
cp ${DS_CONFIG} ${output_model}

# Runner
# deepspeed --master_port 29500 --include localhost:0 ${BASE_DIR}/train/inference.py \
CUDA_VISIBLE_DEVICES=0 python ${BASE_DIR}/train/inference.py \
    --model_name_or_path ${BASE_MODEL} \
    --config_name ${CONFIG_NAME} \
    --test_files xxx/Nips/dataset/dk/final_5.csv \
    --data_column_name smiles \
    --label_column_name label \
    --normlization True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --train_on_inputs True \
    --model_class ${MODEL_CLASS} \
    --task_type regression \
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
    --evaluation_strategy no \
    --eval_steps 100 \
    --seed 42 \
    --disable_tqdm false \
    --block_size 1024 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --bf16 False \
    --torch_dtype float32 \
    | tee -a ${output_model}/train.log
