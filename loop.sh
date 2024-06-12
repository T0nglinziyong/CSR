#!/bin/bash

function get_config()
{
    # 参数依次为
    # ${transform},${use_condition},${routing_start},${routing_end},${mask_type},${mask_type_2}
    # ${use_supervision},${layer_super},${margin}
    # ${lr},${wd},${seed}
    transform=$1
    use_condition=$2
    routing_start=$3
    routing_end=$4
    mask_type=$5
    mask_type_2=$6
    use_supervision=$7
    layer_super=$8
    margin=$9
    lr=${10}
    wd=${11}
    seed=${12}
    router_type=${13}
    use_attn=${14}

    if [ "$routing_start" == "24" ]; then
        config_=trans_${transform}__mask_${mask_type}__sup_${use_supervision}_${layer_super}_margin_${margin}_lr_${lr}__wd_${wd}__s_${seed}
    else
        config_=trans_${transform}__rout_${router_type}_from_${routing_start}_to_${routing_end}__mask_${mask_type}_${mask_type_2}__attn_${use_attn}__s_${seed}
    fi
    echo $config_
}

model=${MODEL:-princeton-nlp/sup-simcse-roberta-large}  # pre-trained model
encoding=${ENCODER_TYPE:-bi_encoder}  # cross_encoder, bi_encoder, tri_encoder
lr=${LR:-1e-5}  # learning rate
wd=${WD:-0.1}  # weight decay

transform=${TRANSFORM:-False}  # whether to use an additional linear layer after the encoder
routing_start=${ROUT_START:-20} # where to start using routing
routing_end=${ROUT_END:-666} # where to end using routing
router_type=${ROUTER_TYPE:-0}
use_attn=${USE_ATTN:-False}
use_condition=${USE_CONDITION:-True}
temperature=${TEMPERATURE:-1}
mask_type=${MASK_TYPE:-0} # mask type: 0-5
mask_type_2=${MASK_TYPE_2:-4}
use_supervision=${USE_SUPER:-False}
layer_super=${LAYER_SUPER:-23}
margin=${MARGIN:-0.1}
objective=${OBJECTIVE:-mse}  # mse, triplet, triplet_mse

triencoder_head=${TRIENCODER_HEAD:-None}  # hadamard, concat (set for tri_encoder)
num_train_epochs=${NUM_EPOCHS:-3}
output_dir=${OUTPUT_DIR:-output}
basic_config=${encoding}__obj_${objective}__version_0

train_file=${TRAIN_FILE:-data/csts_train.csv}
eval_file=${EVAL_FILE:-data/csts_validation.csv}
test_file=${TEST_FILE:-data/csts_test.csv}

random_seeds=(42 43 44)

basic_config=${encoding}__obj_${objective}__version_2
use_output=False
router_type=0
for routing_end in 24; do
for routing_start in 23 22 21 20 19 18; do
for use_attn in True False; do
for seed in "${random_seeds[@]}"; do
    config=$(get_config ${transform} ${use_condition} ${routing_start} ${routing_end} ${mask_type} ${mask_type_2} ${use_supervision} ${layer_super} ${margin} ${lr} ${wd} ${seed} ${router_type} ${use_attn})

    python run_sts.py \
    --output_dir "${output_dir}/${basic_config}/${config}" \
    --model_name_or_path ${model} \
    --objective ${objective} \
    --encoding_type ${encoding} \
    --pooler_type cls \
    --freeze_encoder False \
    --transform ${transform} \
    --triencoder_head ${triencoder_head} \
    --max_seq_length 512 \
    --train_file ${train_file} \
    --validation_file ${eval_file} \
    --test_file ${test_file} \
    --condition_only False \
    --sentences_only False \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate ${lr} \
    --weight_decay ${wd} \
    --max_grad_norm 0.0 \
    --num_train_epochs ${num_train_epochs} \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --log_level info \
    --disable_tqdm True \
    --save_strategy epoch \
    --save_total_limit 2 \
    --seed ${seed} \
    --data_seed ${seed} \
    --fp16 True \
    --log_time_interval 15 \
    --overwrite_output_dir False \
    --mask_type ${mask_type} \
    --mask_type_2 ${mask_type_2} \
    --routing_start ${routing_start} \
    --routing_end ${routing_end} \
    --router_type ${router_type} \
    --use_output ${use_output} \
    --use_attn ${use_attn} \
    --use_condition ${use_condition} \
    --temperature ${temperature} \
    --use_supervision ${use_supervision} \
    --layer_super ${layer_super} \
    --margin ${margin}
done
done
done
done