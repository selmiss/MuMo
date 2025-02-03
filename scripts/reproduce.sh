bash ./pretrain/ma-mamba.sh;
master_port=29500 gpus=0,1 bash ./sft/tuning_all.sh ma-mamba bace,bbbp,clintox,tox21,sider,delaney,lipo,freesolv MA_MambaFinetune sft_geo_randomsplit;
master_port=29500 gpus=0,1 bash ./sft/tuning_all.sh ma-mamba bace,bbbp MA_MambaFinetune sft_geo_scaffoldsplit;