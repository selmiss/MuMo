: "${BASE_DIR:?Environment variable BASE_DIR not set}"

python ${BASE_DIR}/preprocess/similarity/random_pair.py;
bash ${BASE_DIR}/scripts/eval/infer_embedding.sh;
python ${BASE_DIR}/preprocess/similarity/csv_tools.py
