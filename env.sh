#!/bin/bash
export MODEL=princeton-nlp/sup-simcse-roberta-large # openai/clip-vit-large-patch14 #
export ENCODER_TYPE=bi_encoder
export OBJECTIVE=mse #triplet_mse # classification # # mse
export NUM_LABELS=1

export TRANSFORM=False
export USE_CONDITION=True # 在使用router_type0时有用，可暂时忽略
export TEMPERATURE=1 # 在使用router_type0时有用，可暂时忽略
export ROUT_START=24 # router开始的层数
export ROUT_END=24 # router结束的层数
export ROUTER_TYPE=3 # 0-1: use additional router; 2-3 use attention score provided by self-attention
export USE_OUTPUT=False # whether the input of the router is hidden_states or self_output from self-attention
export USE_ATTN=True # 额外操作是否要包含self-attention
export MASK_TYPE=0
export MASK_TYPE_2=4
export SEED=43

export USE_SUPER=False # 使用gpt的监督信号
export LAYER_SUPER=23 # 监督的层数
export MARGIN=0.1 # loss函数的margin

export POOLER_TYPE=cls
export SHOW_EXAMPLE=18
export NUM_EPOCHS=3

export EVAL_FILE=data/csts_validation.csv 
export TRAIN_FILE=data/csts_train.csv
export TEST_FILE=data/csts_test.csv

#export EVAL_FILE=data/val_3_5.csv 
#export TRAIN_FILE=data/train.csv
#export TEST_FILE=data/val_3_5.csv 

#export EVAL_FILE=data/hard_example.csv
