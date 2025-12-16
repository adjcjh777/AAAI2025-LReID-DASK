export CUDA_VISIBLE_DEVICES=3,4

/DATA2025/cjh/envs/IRL/bin/python continual_train.py \
    --setting 1\
    --mobile \
    --batch-size 128\
    --logs-dir /DATA2025/cjh/AAAI2025-LReID-DASK/output \
    --data-dir /DATA2025/cjh/AAAI2025-LReID-DASK/PRID\
    --dropout 0 \
    --l2sp-weight 0.01\
    --middle_test\
    --fisher-freeze\
    --fisher-ratio 0.3\
    --fisher-sample-num 1000
    


