export CUDA_VISIBLE_DEVICES=0


VAL_DATA_PATH=./good_cases/inference.json \
INFERENCE_DIR=./inference_results
CHECKPOINT=./checkpoint/second_stage_model.pt
VISION_MODEL=./checkpoint/blip2_vision_model
LANGUAGE_MODEL=../../CKPT/meta-llama/Llama-2-7b-chat-hf
PROCESSOR=../../CKPT/Salesforce/blip2-flan-t5-xxl
SD_BASE=../../CKPT/stabilityai/stable-diffusion-xl-base-1.0
SD_REFINER=../../CKPT/stabilityai/stable-diffusion-xl-refiner-1.0

python inference.py \
    --fp16 \
    --generate_image \
    --max_output_length 320 \
    --val_data_path $VAL_DATA_PATH \
    --inference_dir $INFERENCE_DIR \
    --checkpoint $CHECKPOINT \
    --vision_model $VISION_MODEL \
    --language_model $LANGUAGE_MODEL \
    --processor $PROCESSOR \
    --sd_base $SD_BASE \
    --sd_refiner $SD_REFINER \
    --num_query_tokens 32 \
    --num_qformer_hidden_layers 12 \
    --num_qformer_attention_heads 12 \
    --qformer_hidden_size 768 \
    --qformer_intermediate_size 3072 
