#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.3

data_dir=work/
ckpt_dir=models/role

python sequence_label.py --num_epoch 7 \
    --learning_rate 3e-5 \
    --data_dir ${data_dir} \
    --schema_path ${data_dir}/event_schema/event_schema.json \
    --train_data ${data_dir}/train_data/train.json \
    --dev_data ${data_dir}/dev_data/dev.json \
    --test_data ${data_dir}/dev_data/dev.json \
    --predict_data ${data_dir}/test1_data/test1.json \
    --do_train True \
    --do_predict True \
    --do_model role \
    --max_seq_len 256 \
    --batch_size 8 \
    --model_save_step 100 \
    --eval_step 100 \
    --checkpoint_dir ${ckpt_dir}

