: "${BASE_DIR:?Environment variable BASE_DIR not set}"

bash ${BASE_DIR}/scripts/sft_QM/qm9_alpha.sh;
bash ${BASE_DIR}/scripts/sft_QM/qm9_cv.sh;
bash ${BASE_DIR}/scripts/sft_QM/qm9_mu.sh;
bash ${BASE_DIR}/scripts/sft_QM/qm9_r2.sh;
bash ${BASE_DIR}/scripts/sft_QM/qm9_zpve.sh;
bash ${BASE_DIR}/scripts/sft_QM/qm9.sh;