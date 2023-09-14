import os
import time
import json
import math
import torch
import wandb
import shutil
import argparse
import deepspeed
import numpy as np
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, Blip2Processor, get_cosine_schedule_with_warmup
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from model import MIMModel
from data_utils import load_mim_dataset, load_pair_dataset
from utils import parse_args, build_model_and_processor

os.environ["WANDB_API_KEY"] = "85a3c5af1814c40a13d5d9e64783857cf260b506"
os.environ["WANDB_MODE"] = "dryrun"

def save_checkpoint(args, model_engine, checkpoint_dir):
    model_engine.save_checkpoint(checkpoint_dir)
    if args.local_rank in [-1, 0]:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        state_dict = {key: value for key, value in state_dict.items() if key.split(".")[0] in args.save_modules}
        
        shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(checkpoint_dir, 'pytorch_model.pt'))
    torch.distributed.barrier()
        
def train(args):   
        
    model, tokenizer, image_processor = build_model_and_processor(args)
    
    if not args.training_lm:
        for param in model.language_model.parameters():
            param.requires_grad = False

    if not args.training_vm:
        for param in model.vision_model.parameters():
            param.requires_grad = False
    
    args.world_size = int(os.environ["WORLD_SIZE"])
    if args.stage == "first":
        train_dataset, collate_fn = load_pair_dataset(args=args, tokenizer=tokenizer, image_processor=image_processor)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_micro_batch_size_per_gpu, num_workers=args.with_num_works, collate_fn=partial(collate_fn, args=args), pin_memory=True)
        num_training_steps = args.num_epochs * args.with_epoch // ( args.world_size * args.train_micro_batch_size_per_gpu * args.gradient_accumulation_steps)
        lr_scheduler = partial(get_cosine_schedule_with_warmup, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
        model_engine, _, _, _ = deepspeed.initialize(args=args, model=model, lr_scheduler=lr_scheduler)
        train_dataloader_len = args.with_epoch // (args.world_size * args.train_micro_batch_size_per_gpu)
    else:        
        train_dataset, collate_fn = load_mim_dataset(args=args, tokenizer=tokenizer, image_processor=image_processor)
        num_training_steps = args.num_epochs * len(train_dataset) // (args.world_size * args.train_micro_batch_size_per_gpu * args.gradient_accumulation_steps)
        lr_scheduler = partial(get_cosine_schedule_with_warmup, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
        model_engine, _, train_dataloader, _ = deepspeed.initialize(args=args, model=model, lr_scheduler=lr_scheduler, training_data=train_dataset, collate_fn=partial(collate_fn, args=args))
        train_dataloader_len = len(train_dataloader)
        
    print("Total training steps: ", num_training_steps)
    
    # Training loop
    model_engine.train()
    current_step = 0
    total_instance = 0
    for epoch in tqdm(range(args.num_epochs), desc='Epoch', unit='epoch'):
        step_progress = tqdm(enumerate(train_dataloader), desc='Step', leave=False, unit='step', total=train_dataloader_len)
        for step, batch in step_progress:
            batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
            
            bsz = batch["input_ids"].shape[0]
            total_instance += bsz

            # Compute Loss
            loss = model_engine(**batch)
            model_engine.backward(loss)
            model_engine.step()

            current_step += 1
            step_progress.set_description(f'Epoch {epoch} Step {current_step} - Loss: {loss:.4f}')
            wandb.log({"loss": loss})
            
            if current_step % args.save_per_steps == 0:
                checkpoint_dir = os.path.join(args.save_dir, f'checkpoint_step{current_step // args.gradient_accumulation_steps}')
                save_checkpoint(args, model_engine, checkpoint_dir)
        print (f"{epoch+1} finished: {total_instance} instances")
        checkpoint_dir = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}_step{current_step // args.gradient_accumulation_steps}')
        save_checkpoint(args, model_engine, checkpoint_dir)

@torch.no_grad()
def validation(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer, image_processor = build_model_and_processor(args)
    model = model.half().to(device) if args.fp16 else model.to(device)
    model.eval()

    if args.stage == "first":
        val_dataset, collate_fn = load_pair_dataset(args=args, tokenizer=tokenizer, image_processor=image_processor)
        val_dataloader = DataLoader(val_dataset, batch_size=args.train_micro_batch_size_per_gpu, num_workers=args.with_num_works, collate_fn=partial(collate_fn, args=args), pin_memory=True, shuffle=False)
        val_dataloader_len = args.with_epoch // args.train_micro_batch_size_per_gpu
    else:
        val_dataset, collate_fn = load_mim_dataset(args=args, tokenizer=tokenizer, image_processor=image_processor)
        val_dataloader = DataLoader(val_dataset, batch_size=args.train_micro_batch_size_per_gpu, num_workers=args.with_num_works, collate_fn=partial(collate_fn, args=args), pin_memory=True, shuffle=False)    
        val_dataloader_len = len(val_dataloader)
        
    loss = []
    total_instance = 0
    step_progress = tqdm(enumerate(val_dataloader), desc='Step', leave=False, unit='step', total=val_dataloader_len)
    for step, batch in step_progress:
        batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
                
        bsz = batch["input_ids"].shape[0]
        total_instance += bsz

        # Compute Loss
        _loss = model(**batch)
        loss.append(_loss.item())

        step_progress.set_description(f'Step {step} - Loss: {_loss:.4f}')
    
    print (f"Validation finished: {total_instance} instances, {len(loss)} batches")
    print (f"Validation loss: {np.mean(loss)}")

def main():
    args = parse_args()   

    config = vars(args)
    deepspeed_config = json.load(open(args.deepspeed_config))
    config.update(deepspeed_config) 
    for key, value in deepspeed_config.items():
        setattr(args, key, value)

    args.save_dir = os.path.join("checkpoint", args.project_name)
    if args.local_rank in [0, -1] and args.train and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.train:            
        wandb.init(
            project=args.project_name,
            group="ddp",
            config=config,
            dir=args.save_dir
        )
        
        # from flash_attention import replace_llama_attn_with_flash_attn
        # replace_llama_attn_with_flash_attn()
        
        train(args)
        wandb.finish()
    else:
        validation(args)
    
    
if __name__ == '__main__':
    main()
