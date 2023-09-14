
export CUDA_VISIBLE_DEVICES=0

INFERENCE_DIR=./MMBench_Results/second_stage_1e-5_12_12_768_3072_blip2_cosine2_tlm_mim_21k
CHECKPOINT=./checkpoint/second_stage_1e-5_12_12_768_3072_blip2_cosine2_tlm_mim_21k/checkpoint_epoch3_step1014/pytorch_model.pt
VISION_MODEL=./checkpoint/blip2_vision_model
LANGUAGE_MODEL=../../CKPT/meta-llama/Llama-2-7b-chat-hf
PROCESSOR=../../CKPT/Salesforce/blip2-flan-t5-xxl
SD_BASE=../../CKPT/stabilityai/stable-diffusion-xl-base-1.0
SD_REFINER=../../CKPT/stabilityai/stable-diffusion-xl-refiner-1.0

python mmbench_evaluation.py \
    --fp16 \
    --max_output_length 256 \
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