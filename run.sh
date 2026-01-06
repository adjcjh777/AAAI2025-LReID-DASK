export CUDA_VISIBLE_DEVICES=0

/DATA2025/cjh/envs/IRL/bin/python continual_train.py \
    --mobile \
    --mobilenet-type resnet50 \
    --with-attention \
    --batch-size 128\
    --logs-dir /DATA2025/cjh/AAAI2025-LReID-DASK/output \
    --data-dir /DATA2025/cjh/AAAI2025-LReID-DASK/PRID\
    --setting 1\
    #--dropout 0.2 \
    # --fisher-freeze\
    # --fisher-ratio 0.5\
    #--fisher-sample-num 500\
    #--l2sp-weight 0.01\


