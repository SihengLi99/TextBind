export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
export CUDA_VISIBLE_DEVICES=0

VAL_DATA_PATH=../MIM/data/minigpt4/cc_sbu/cc_sbu_dataset/01255.tar
CHECKPOINT=./checkpoint/first_stage_1e-4_256_12_12_768_3072_blip2_cosine2/checkpoint_epoch2_step78128/pytorch_model.pt
VISION_MODEL=./checkpoint/blip2_vision_model
LANGUAGE_MODEL=../../CKPT/meta-llama/Llama-2-7b-chat-hf
PROCESSOR=../../CKPT/Salesforce/blip2-flan-t5-xxl

python main.py \
    --use_causal_mask \
    --fp16 \
    --compute_loss \
    --stage first \
    --deepspeed_config ./ds_config_first_stage.json \
    --with_epoch 5000 \
    --with_num_works 4 \
    --val_data_path $VAL_DATA_PATH \
    --checkpoint $CHECKPOINT \
    --vision_model $VISION_MODEL \
    --language_model $LANGUAGE_MODEL \
    --processor $PROCESSOR \
    --max_input_length 768 \
    --num_query_tokens 32 \
    --num_qformer_hidden_layers 12 \
    --num_qformer_attention_heads 12 \
    --qformer_hidden_size 768 \
    --qformer_intermediate_size 3072 