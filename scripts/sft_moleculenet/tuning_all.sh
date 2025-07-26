: "${BASE_DIR:?Environment variable BASE_DIR not set}"

export PYTHONPATH=${BASE_DIR}
MODEL_NAME=$1
TASKS=$2
MASTER_PORT=$master_port
GPUs=$gpus
MODEL_CLASS=$3
DATA_TYPE=$4


IFS=',' read -ra TASK_ARRAY <<< "$TASKS"

declare -A TASK_SCRIPTS=(
  ["bace"]="${BASE_DIR}/scripts/sft_moleculenet/classfication/bace.sh"
  ["bbbp"]="${BASE_DIR}/scripts/sft_moleculenet/classfication/bbbp.sh"
  ["clintox"]="${BASE_DIR}/scripts/sft_moleculenet/classfication/clintox.sh"
  ["hiv"]="${BASE_DIR}/scripts/sft_moleculenet/classfication/hiv.sh"
  ["lipo"]="${BASE_DIR}/scripts/sft_moleculenet/regression/lipo.sh"
  ["delaney"]="${BASE_DIR}/scripts/sft_moleculenet/regression/delaney.sh"
  ["freesolv"]="${BASE_DIR}/scripts/sft_moleculenet/regression/freesolv.sh"
  ["sider"]="${BASE_DIR}/scripts/sft_moleculenet/classfication/sider.sh"
  ["tox21"]="${BASE_DIR}/scripts/sft_moleculenet/classfication/tox21.sh"
)


for TASK in "${TASK_ARRAY[@]}"; do
  if [[ -n "${TASK_SCRIPTS[$TASK]}" ]]; then
    echo "Executing task: $TASK"
    bash "${TASK_SCRIPTS[$TASK]}" "${MODEL_NAME}" "${MASTER_PORT}" "${GPUs}" "${MODEL_CLASS}" "${DATA_TYPE}"
  else
    echo "Task $TASK is not recognized. Skipping."
  fi
done

python ${BASE_DIR}/tools/gen_results.py --model_name "${MODEL_NAME}" --model_class ${MODEL_CLASS} --data_type ${DATA_TYPE}  --tasks bace,bbbp,clintox,hiv,delaney,lipo,freesolv,sider,tox21




