import os
import json 
import copy
import torch
import deepspeed
import transformers
import argparse
import numpy as np
import webdataset as wds
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel, LlamaTokenizer, Blip2Processor
from typing import List, Dict, Sequence
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import Blip2ForConditionalGeneration, Blip2VisionModel
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from model import MIMModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='mim', help='Wandb project name')
    parser.add_argument('--deepspeed_config', type=str, default='deepspeed_config.json', help='DeepSpeed configuration file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (-1: not distributed)')
    parser.add_argument('--train_data_path', type=str, default='data/train.json', help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default='data/val.json', help='Path to validation data')
    parser.add_argument('--image_dir', type=str, default=None, help='Path to image directory')
    parser.add_argument('--inference_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoints')
    parser.add_argument('--save_modules', type=str, nargs='+', default=['model'], help='State keys to save')
    
    # training parameters
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--compute_loss', action='store_true', help='Compute loss')
    parser.add_argument('--stage', type=str, help='Training stage')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--max_input_length', type=int, default=512, help='Maximum input length')
    parser.add_argument('--max_num_images', type=int, default=32, help='Maximum number of images')
    parser.add_argument('--with_epoch', type=int, default=0, help='for wds')
    parser.add_argument('--with_num_works', type=int, default=1, help="for wds")
    parser.add_argument('--save_per_steps', type=int, default=1000000, help='Save model per number of steps')
    parser.add_argument('--training_lm', action='store_true', default=False, help='Train language model')
    parser.add_argument('--training_vm', action='store_true', default=False, help='Train language model')
    
    # inference parameters
    parser.add_argument('--max_output_length', type=int, default=256, help='Maximum generation length')
    parser.add_argument('--generate_image', action='store_true', help='Generate image')
    parser.add_argument('--top_p', type=float, default=None, help='Top p')
    
    # model parameters
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--bf16', action='store_true', help='Use bf16')
    parser.add_argument('--vision_model', type=str, default='openai/clip-vit-base-patch32', help='Vision model')
    parser.add_argument('--language_model', type=str, default='openai/clip-vit-base-patch32', help='Language model')
    parser.add_argument('--sd_base', type=str, default="runwayml/stable-diffusion-v1-5", help='Stable Diffusion model')
    parser.add_argument('--sd_refiner', type=str, default="runwayml/stable-diffusion-v1-5", help='Stable Diffusion model')
    parser.add_argument('--processor', type=str, default='clip', help='Processor')
    parser.add_argument('--num_query_tokens', type=int, default=32, help='Number of query tokens')
    parser.add_argument('--num_qformer_attention_heads', type=int, default=16, help='Number of query tokens')
    parser.add_argument('--num_qformer_hidden_layers', type=int, default=12, help='Number of query tokens')
    parser.add_argument('--qformer_hidden_size', type=int, default=1024, help='Number of query tokens')
    parser.add_argument('--qformer_intermediate_size', type=int, default=1408, help='Number of query tokens')

    # demo parameters
    parser.add_argument('--port', default=8081, help='Port to run the demo')
    parser.add_argument('--model_list', type=str, default="", help='path to the info of model list')
    parser.add_argument('--demo_example_path', type=str, default="", help='path to the example data')
    parser.add_argument('--url_prefix', type=str, default="", help='add prefix to the url')
    parser.add_argument('--safe_image_num', type=int, default=16, help='maximum number of images appearing in conversation')
    parser.add_argument('--safe_word_num', type=int, default=768, help='maximum number of words appearing in conversationl')
        
    return parser.parse_args()

def build_model_and_processor(args):
    
    tokenizer = LlamaTokenizer.from_pretrained(args.language_model)
    add_tokens = ["<image>", "<start>", "<end>"]
    tokenizer.add_special_tokens(({"additional_special_tokens": add_tokens}))    
    args.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    args.caption_start_id = tokenizer.convert_tokens_to_ids("<start>")
    args.caption_end_id = tokenizer.convert_tokens_to_ids("<end>")
    args.num_new_tokens = len(add_tokens)
    image_processor = Blip2Processor.from_pretrained(args.processor)
    model = MIMModel(args)
    model.language_model.resize_token_embeddings(len(tokenizer))

    if args.checkpoint:
        state_dict = torch.load(os.path.join(args.checkpoint))
        model.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint from %s" % args.checkpoint)
        print("Loaded modules: %s" % set([key.split(".")[0] for key in state_dict.keys()]))
    
    return model, tokenizer, image_processor


def smart_tokenizer_and_embedding_resize(
    additional_tokens: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_tokens(additional_tokens)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def compute_clip_score(
    model: transformers.CLIPModel,
    processor: transformers.CLIPProcessor,
    image: Image,
    caption: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_input = processor(text=[caption], images=[image.cpu()], return_tensors="pt", padding=True)

    img_features = model.get_image_features(processed_input["pixel_values"].to(device))
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

    txt_features = model.get_text_features(
        processed_input["input_ids"][:, :77].to(device), processed_input["attention_mask"][:, :77].to(device)
    )
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)
    
    # cosine similarity between feature vectors
    score = (img_features * txt_features).sum(axis=-1).item()

    return score


if __name__ == "__main__":
    
    import torch
    from transformers import Blip2ForConditionalGeneration
    model = Blip2ForConditionalGeneration.from_pretrained("../../CKPT/Salesforce/blip2-flan-t5-xxl")
    vision_model = model.vision_model
    vision_model.save_pretrained("checkpoint/blip2_vision_model")

    state_dict = model.state_dict()
    state_dict = {key: value for key, value in state_dict.items() if key.split(".")[0] in ["query_tokens", "qformer"]}
    torch.save(state_dict, "checkpoint/blip2_qformer.pt")
                
