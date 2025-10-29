: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"
export PYTHONPATH=${BASE_DIR}

bash ${BASE_DIR}/scripts/pretrain/mumo.sh;
master_port=29500 gpus=0,1 bash ${BASE_DIR}/scripts/sft_moleculenet/tuning_all.sh mumo bace,bbbp,clintox,tox21,sider,delaney,lipo,freesolv MuMoFinetune sft_geo_randomsplit;
master_port=29500 gpus=0,1 bash ${BASE_DIR}/scripts/sft_moleculenet/tuning_all.sh mumo bace,bbbp MuMoFinetune sft_geo_scaffoldsplit;