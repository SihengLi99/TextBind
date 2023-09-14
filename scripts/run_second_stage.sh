export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

TRAIN_DATA_PATH=./data/platypus_24926_minigpt4_3439_shikra_7081_llava_157712_multi_instruct_105745_mim_21629.json
SAVE_MODULES="query_tokens qformer qformer_projection language_model"
CHECKPOINT=./checkpoint/first_stage_1e-4_256_12_12_768_3072_blip2_cosine2/checkpoint_epoch2_step78128/pytorch_model.pt
VISION_MODEL=./checkpoint/blip2_vision_model
LANGUAGE_MODEL=../../CKPT/meta-llama/Llama-2-7b-chat-hf
PROCESSOR=../../CKPT/Salesforce/blip2-flan-t5-xxl

deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    main.py \
    --train \
    --stage second \
    --training_lm \
    --deepspeed_config ./ds_config_second_stage.json \
    --project_name second_stage_1e-5_12_12_768_3072_blip2_cosine2_tlm_plat25k_mini3k_shik7k_llav158k_mult106k_mim22k_768 \
    --num_epochs 3 \
    --warmup_steps 100 \
    --train_data_path $TRAIN_DATA_PATH \
    --save_modules $SAVE_MODULES \
    --checkpoint $CHECKPOINT \
    --vision_model $VISION_MODEL \
    --language_model $LANGUAGE_MODEL \
    --processor $PROCESSOR \
    --max_input_length 768 \
    --num_query_tokens 32 \
    --num_qformer_hidden_layers 12 \
    --num_qformer_attention_heads 12 \
    --qformer_hidden_size 768 \
    --qformer_intermediate_size 3072 \
