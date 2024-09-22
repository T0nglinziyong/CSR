#!/bin/bash
export MODEL=princeton-nlp/sup-simcse-roberta-large  #./simcse-base #./diffcse-roberta-base-sts #
export ENCODER_TYPE=tri_encoder
export OBJECTIVE=mse #triplet_mse # classification # 
export TRIENCODER_HEAD=hadamard
export NUM_LABELS=1

export TRANSFORM=True
export ROUT_START=24 # router开始的层数
export ROUT_END=24 # router结束的层数
export ROUTER_TYPE=3 # 0-1: use additional router; 2-3 use attention score provided by self-attention
export MASK_TYPE=4
export MASK_TYPE_2=0
export SEED=45

export POOLER_TYPE=cls
export SHOW_EXAMPLE=18
export LAYER_SUPER=-1 
export NUM_EPOCHS=3

export EVAL_FILE=data/csts_validation.csv 
export TRAIN_FILE=data/csts_train.csv
export TEST_FILE=data/csts_test.csv
