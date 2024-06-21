#!/bin/bash
export MODEL=princeton-nlp/sup-simcse-roberta-large  #./simcse-base #./diffcse-roberta-base-sts #
export ENCODER_TYPE=bi_encoder
export OBJECTIVE=mse #triplet_mse # classification # # mse
export TRIENCODER_HEAD=None # hadamard
export NUM_LABELS=1

export TRANSFORM=False
export ROUT_START=21 # router开始的层数
export ROUT_END=24 # router结束的层数
export ROUTER_TYPE=3 
export MASK_TYPE=0
export MASK_TYPE_2=4
export SEED=45

export USE_SUPER=False # 使用gpt的监督信号
export LAYER_SUPER=-1 # 监督的层数
export MARGIN=0.1 # loss函数的margin

export POOLER_TYPE=cls
export SHOW_EXAMPLE=18
export NUM_EPOCHS=3

export EVAL_FILE=data/csts_validation.csv 
export TRAIN_FILE=data/csts_train.csv
export TEST_FILE=data/csts_test.csv
