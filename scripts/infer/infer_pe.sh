: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"
export PYTHONPATH=${BASE_DIR}
MODEL_NAME=mumo_pin1
MODEL_CLASS=MuMoFinetune
CONFIG_NAME=${BASE_DIR}/config/mumo/config_cls_low.json

# Base config
output_dir=${DATA_DIR}/model/sft/mumo_pin1/mumo_pin1_MuMoFinetunePairwise_pairwise-pin1_pe
BASE_MODEL=${DATA_DIR}/model/sft/mumo_pin1/mumo_pin1_MuMoFinetunePairwise_pairwise-pin1_pe

# DeepSpeed config
DS_CONFIG=${BASE_DIR}/config/deepspeed/ds_config_zero2.json

# Keep
SCRIPT_PATH="$(realpath "$0")"
if [ ! -d ${output_dir} ];then  
    mkdir ${output_dir}
fi
cp ${SCRIPT_PATH} ${output_dir}

# Runner

CUDA_VISIBLE_DEVICES=0 python ${BASE_DIR}/train/inference.py \
    --model_name_or_path ${BASE_MODEL} \
    --model_class ${MODEL_CLASS} \
    --task_type regression \
    --cache_dir ${DATA_DIR}/cache \
    --model_revision main \
    --use_fast_tokenizer false \
    --torch_dtype float32 \
    --output_size 1 \
    --pool_method mean \
    --test_files ${DATA_DIR}/dataset/dk/targets/smiles_all_051.jsonl \
    --data_column_name smiles \
    --output_dir ${output_dir} \
    --batch_size 1024 \
    --max_length 1024 \
    | tee -a ${output_dir}/inference.log
