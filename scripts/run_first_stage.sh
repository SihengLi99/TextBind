export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

SAVE_MODULES="query_tokens qformer qformer_projection"
TRAIN_DATA_PATH="../MIM/data/minigpt4/cc_sbu/cc_sbu_dataset/{00000..01254}.tar"
CHECKPOINT=./checkpoint/blip2_vision.pt
VISION_MODEL=./checkpoint/blip2_vision_model
LANGUAGE_MODEL=../../CKPT/meta-llama/Llama-2-7b-chat-hf
PROCESSOR=../../CKPT/Salesforce/blip2-flan-t5-xxl

deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    main.py \
    --train \
    --stage first \
    --project_name first_stage_1e-4_256_12_12_768_3072_blip2_cosine8 \
    --deepspeed_config ./ds_config_first_stage.json \
    --with_epoch 10000000 \
    --num_epochs 8 \
    --warmup_steps 2000 \
    --with_num_works 4 \
    --save_modules $SAVE_MODULES \
    --train_data_path $TRAIN_DATA_PATH \
    --checkpoint $CHECKPOINT \
    --vision_model $VISION_MODEL \
    --language_model $LANGUAGE_MODEL \
    --processor $PROCESSOR \
    --max_input_length 256 \
    --num_query_tokens 32 \
    --num_qformer_hidden_layers 12 \
    --num_qformer_attention_heads 12 \
    --qformer_hidden_size 768 \
    --qformer_intermediate_size 3072 