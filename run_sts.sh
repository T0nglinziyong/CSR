#!/bin/bash
model=${MODEL:-princeton-nlp/sup-simcse-roberta-large}  # pre-trained model
encoding=${ENCODER_TYPE:-bi_encoder}  # cross_encoder, bi_encoder, tri_encoder
lr=${LR:-1e-5}  # learning rate
wd=${WD:-0.1}  # weight decay

transform=${TRANSFORM:-False}  # whether to use an additional linear layer after the encoder
routing_start=${ROUT_START:-20} # where to start using routing
routing_end=${ROUT_END:-666} # where to end using routing
router_type=${ROUTER_TYPE:-0}
use_output=${USE_OUTPUT:-True}
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
seed=${SEED:-42}
output_dir=${OUTPUT_DIR:-output}
#basic_config=model_${model//\//__}__enc_${encoding}__obj_${objective}
basic_config=${encoding}__obj_${objective}
if [ "$routing_start" == "24" ]; then
    config=trans_${transform}__mask_${mask_type}__sup_${use_supervision}_${layer_super}_margin_${margin}_lr_${lr}__wd_${wd}__s_${seed}
else
    config=trans_${transform}__rout_${use_condition}_t_${temperature}_from_${routing_start}_to_${routing_end}__mask_${mask_type}_${mask_type_2}__sup_${use_supervision}_${layer_super}_margin_${margin}__lr_${lr}__wd_${wd}__s_${seed}
fi
#config=trans_${transform}__rout_${use_condition}_t_${temperature}_from_${routing_start}_to_${routing_end}__mask_${mask_type}_${mask_type_2}__sup_${use_supervision}__lr_${lr}__wd_${wd}__s_${seed}
train_file=${TRAIN_FILE:-data/csts_train.csv}
eval_file=${EVAL_FILE:-data/csts_validation.csv}
test_file=${TEST_FILE:-data/csts_test.csv}

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
  --overwrite_output_dir True \
  --show_example 8 \
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
  --margin ${margin} \

  
