BASE_DIR=xxx # Change to your project dir
DATA_DIR=xxx/Nips # Change to your data file dir

export PYTHONPATH=${BASE_DIR}
MODEL_NAME=mumo
TASK_NAME=Pgp_Broccatelli
MODEL_CLASS=MuMoFinetune
DATATYPE=sft_tdc_geo

# Base config
output_model=${DATA_DIR}/model/sft/${MODEL_NAME}/${MODEL_NAME}_${MODEL_CLASS}_${DATATYPE}-${TASK_NAME}
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
deepspeed --master_port 29502 --include localhost:1 ${BASE_DIR}/train/finetune.py \
    --model_class ${MODEL_CLASS} \
    --task_type classification \
    --model_name_or_path ${BASE_MODEL} \
    --pool_method bipooler \
    --tokenizer_name ${BASE_MODEL} \
    --train_files ${DATA_DIR}/dataset/${DATATYPE}/${TASK_NAME}/train.csv \
    --validation_files ${DATA_DIR}/dataset/${DATATYPE}/${TASK_NAME}/valid.csv \
    --test_files ${DATA_DIR}/dataset/${DATATYPE}/${TASK_NAME}/test.csv \
    --data_column_name smiles \
    --label_column_name Y \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --class_weight True \
    --do_train \
    --do_eval \
    --train_on_inputs True \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --max_eval_samples 1000 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --warmup_steps 20 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 50 \
    --save_strategy no \
    --save_total_limit 5 \
    --preprocessing_num_workers 10 \
    --evaluation_strategy steps \
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

# --resume_from_checkpoint ${output_model}/checkpoint-20400 \
