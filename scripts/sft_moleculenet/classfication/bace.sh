BASE_DIR=xxx # Change to your project dir
DATA_DIR=xxx/Nips/dataset/ # Change to your data file dir

export PYTHONPATH=${BASE_DIR}
MODEL_NAME=$1
TASK_NAME=bace
MODEL_CLASS=$4
DATATYPE=$5

for i in {1..3}
do
    echo $i
    # Base config
    output_model=${DATA_DIR}/model/sft/${MODEL_NAME}/${MODEL_NAME}_${MODEL_CLASS}_${DATATYPE}-${TASK_NAME}_${i}
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
        --model_class ${MODEL_CLASS} \
        --task_type classification \
        --model_name_or_path ${BASE_MODEL} \
        --pool_method bipooler \
        --tokenizer_name ${BASE_MODEL} \
        --train_files ${DATA_DIR}/dataset/${DATATYPE}/${TASK_NAME}_${i}/raw/train_${TASK_NAME}_${i}.csv \
        --validation_files ${DATA_DIR}/dataset/${DATATYPE}/${TASK_NAME}_${i}/raw/test_${TASK_NAME}_${i}.csv \
        --test_files ${DATA_DIR}/dataset/${DATATYPE}/${TASK_NAME}_${i}/raw/test_${TASK_NAME}_${i}.csv \
        --data_column_name smiles \
        --label_column_name Class \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --do_train \
        --do_eval \
        --train_on_inputs True \
        --use_fast_tokenizer false \
        --output_dir ${output_model} \
        --max_eval_samples 1000 \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 5 \
        --warmup_steps 5 \
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
        --report_to tensorboard \
        --overwrite_output_dir \
        --ignore_data_skip true \
        --bf16 False \
        --torch_dtype float32 \
        | tee -a ${output_model}/train.log

done

# --resume_from_checkpoint ${output_model}/checkpoint-20400 \
