BASE_DIR=/home/UWO/zjing29/Mams/MuMo # Change to your project dir
DATA_DIR=/data/lab_ph/zihao/Nips # Change to your data file dir
export PYTHONPATH=${BASE_DIR}
MODEL_NAME=mumo
MODEL_CLASS=MuMoFinetune
CONFIG_NAME=${BASE_DIR}/config/mumo/config_cls_reg.json

# Base config
output_dir=/data/lab_ph/zihao/Nips/model/sft/mumo/dk_test
BASE_MODEL=/data/lab_ph/zihao/Nips/model/sft/ma-mamba/ma-mamba_MA_MambaFinetune_sft_tdc_geo-HydrationFreeEnergy_FreeSolv

# Keep
SCRIPT_PATH="$(realpath "$0")"
if [ ! -d ${output_dir} ];then  
    mkdir ${output_dir}
fi
cp ${SCRIPT_PATH} ${output_dir}

# Runner
CUDA_VISIBLE_DEVICES=0 python ${BASE_DIR}/train/inference.py \
    --model_name_or_path ${BASE_MODEL} \
    --model_class MuMoFinetune \
    --task_type regression \
    --cache_dir ${BASE_DIR}/cache \
    --model_revision main \
    --use_fast_tokenizer false \
    --torch_dtype float32 \
    --output_size 1 \
    --pool_method mean \
    --test_files /data/lab_ph/zihao/Nips/dataset/dk/final_5.csv \
    --data_column_name smiles \
    --output_dir ${output_dir} \
    --batch_size 2 \
    --max_length 1024 \
    --scaler_path /data/lab_ph/zihao/Nips/dataset/dk/scaler.pkl \
    | tee -a ${output_dir}/inference.log
