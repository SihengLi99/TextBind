import os
import json
import time
import torch
import random
import argparse
import numpy as np
import webdataset as wds
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from transformers import Blip2Processor
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DiffusionPipeline, AutoencoderKL, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from data_utils import ImageCaptionTemplates, ImageGenerationTemplates



def get_sd_latents(sd_pipe, generator, image):
    with torch.no_grad():
        latents = sd_pipe(prompt="", image=image, generator=generator, strength=0.0, output_type="latent").images
    return latents

def decode_with_latents(sd_pipe, latents):
    with torch.no_grad():
        image = sd_pipe.vae.decode(latents / sd_pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = sd_pipe.watermark.apply_watermark(image)
        image = sd_pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image

def webdataset_map(example):

    image = Image.open(BytesIO(example["image"])).convert("RGB")
    caption = example["caption"].decode("utf-8")
    key = json.loads(example["meta"])["key"]
    
    return {
        "image": image,
        "caption": caption,
        "key": key
    }
    
def webdataset_collate_fn(batch):
    
    images = [example["image"] for example in batch]
    captions = [example["caption"] for example in batch]
    keys = [example["key"] for example in batch]
    
    return {
        "image": images,
        "caption": captions,
        "key": keys
    }

def generate_val_generation_dataset():

    urls = "/apdcephfs/share_733425/jcykcai/sihengli/MIM/mim/data/minigpt4/cc_sbu/cc_sbu_dataset/01255.tar"
    dataset = wds.WebDataset(urls).rename(image="jpg;png", meta="json", caption="txt").map(webdataset_map)
    
    image_dir = "./data/val_image_generation_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    target_path = "./data/val_image_generation.json"
    corpus = []
    for data in dataset:
        caption = data["caption"]
        image = data["image"]
        template = random.choice(ImageGenerationTemplates)
        input_text = f"Human: {template[0].format(image_caption=caption)}\n"
        output_text = f"Assistant: {template[1]}"            
        
        cur_idx = len(corpus)
        image_path = os.path.join(image_dir, f"{cur_idx}.png")
        image.save(image_path)
        corpus.append({
            "input": input_text,
            "output": output_text,
            "key": data["key"],
            "caption": caption,
            "input_image_list": [],
            "output_image_list": [image_path]
        })
        if len(corpus) == 100:
            break
    json.dump(corpus, open(target_path, "w"), indent=4)

def generate_val_caption_dataset():

    urls = "/apdcephfs/share_733425/jcykcai/sihengli/MIM/mim/data/minigpt4/cc_sbu/cc_sbu_dataset/01255.tar"
    dataset = wds.WebDataset(urls).rename(image="jpg;png", meta="json", caption="txt").map(webdataset_map)
    
    image_dir = "./data/val_image_caption_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    target_path = "./data/val_image_caption.json"
    corpus = []
    for data in dataset:
        caption = data["caption"]
        image = data["image"]
        template = random.choice(ImageCaptionTemplates)
        input_text = f"Human: {template}\n"
        output_text = f"Assistant: {caption}"
        
        cur_idx = len(corpus)
        image_path = os.path.join(image_dir, f"{cur_idx}.png")
        image.save(image_path)
        corpus.append({
            "input": input_text,
            "output": output_text,
            "key": data["key"],
            "caption": caption,
            "input_image_list": [image_path],
            "output_image_list": []
        })
        if len(corpus) == 100:
            break
    json.dump(corpus, open(target_path, "w"), indent=4)
    

def generate_sd2_prompt_embeds():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = "/apdcephfs/share_733425/jcykcai/sihengli/CKPT/stabilityai/stable-diffusion-2-1"
    sd_pipe = StableDiffusionPipeline.from_pretrained(ckpt, use_safetensors=True, variant="fp16")
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to("cuda")
      
    urls = "/apdcephfs/share_733425/jcykcai/sihengli/MIM/MIM/data/minigpt4/cc_sbu/cc_sbu_dataset/{01001..01200}.tar"
    dataset = wds.WebDataset(urls).rename(image="jpg;png", meta="json", caption="txt").map(webdataset_map)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, collate_fn=webdataset_collate_fn, num_workers=4, pin_memory=True, drop_last=False)
        
    save_dir = "./data/sd2_prompt_embeds"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, batch in tqdm(enumerate(data_loader)):
    
        with torch.no_grad():
            prompt_embeds = sd_pipe._encode_prompt(prompt=batch["caption"], device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)  
            for id1 in range(prompt_embeds.shape[0]):
                np.save(os.path.join(save_dir, "{}.npy".format(batch['key'][id1])), prompt_embeds[id1].cpu().numpy())          
    
def generate_sd2_prompt_embeds_mim():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = "/apdcephfs/share_733425/jcykcai/sihengli/CKPT/stabilityai/stable-diffusion-2-1"
    sd_pipe = StableDiffusionPipeline.from_pretrained(ckpt, use_safetensors=True, variant="fp16")
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to("cuda")
      
    save_dir = "./data/sd2_prompt_embeds_mim"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # corpus = json.load(open("./data/train.s2.v3.img-cap.json"))[25000:]
    corpus = json.load(open("./data/stage1_sample.json"))
    print(len(corpus))
    for idx, data in tqdm(enumerate(corpus)):   
    
        with torch.no_grad():
            prompt_embeds = sd_pipe._encode_prompt(prompt=data["caption"], device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)  
            # image = sd_pipe(prompt_embeds=prompt_embeds, generator=generator).images[0]
            # print(prompt_embeds.shape)
            # image = sd_pipe(prompt=caption, generator=generator).images[0]
            np.save(os.path.join(save_dir, "{}.npy".format(data["image"])), prompt_embeds[0].cpu().numpy())          

def decode_with_sd2_prompt_embeds(sd_pipe, generator, prompt_embeds):
    
    with torch.no_grad():
        image = sd_pipe(prompt_embeds=prompt_embeds, generator=generator).images[0]
    return image

def sample_stage1_data():
    urls = "/apdcephfs/share_733425/jcykcai/sihengli/MIM/MIM/data/minigpt4/cc_sbu/cc_sbu_dataset/{00000..01200}.tar"
    dataset = wds.WebDataset(urls).rename(image="jpg;png", meta="json", caption="txt").map(webdataset_map)

    sample_num = 20000
    corpus = []
    if not os.path.exists("./data/stage1_sample"):
        os.makedirs("./data/stage1_sample")
    for data in dataset:
        print(data["key"])
        image = data["image"]
        caption = data["caption"]
        key = data["key"]
        image_name = f"stage1_{key}.png"
        image.save(os.path.join("./data/stage1_sample", image_name))
        image.save(os.path.join("./data/mim_images", image_name))
        
        template = random.choice(ImageGenerationTemplates)  
        data = {
            "conversation": [
                {
                    "role": "user",
                    "content": template[0].format(image_caption=caption),
                    "image_list": []
                },
                {
                    "role": "assistant",
                    "content": template[1],
                    "image_list": [image_name],
                }
            ],
            "image": image_name,
            "caption": caption,
        }
        corpus.append(data)
        if len(corpus) == sample_num:
            break
    print(len(corpus))
    json.dump(corpus, open("./data/stage1_sample.json", "w"), indent=4)
    
def blender_stage1_and_stage2():
    
    corpus1 = json.load(open("./data/stage1_sample.json"))
    corpus2 = json.load(open("./data/train_mim.json"))
    corpus = corpus1 + corpus2 
    
    random.shuffle(corpus)
    print(len(corpus))
    json.dump(corpus, open("./data/train_mim_blender.json", "w"), indent=4)
    

def decode_with_sdxl(prompt, base, refiner, generator):
    n_steps = 40
    high_noise_frac = 0.8

    # run both experts
    image = base(
        prompt=prompt,
        generator=generator,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        generator=generator,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    
    return image

if __name__ == "__main__":

    # generate_sd2_prompt_embeds()
    # generate_val_generation_dataset()
    # generate_val_caption_dataset()
    # generate_sd2_prompt_embeds_mim()
    # sample_stage1_data()
    # blender_stage1_and_stage2()
    
    # generate_sd2_prompt_embeds_mim()

    device = "cuda:1"
    ckpt = "/apdcephfs/share_733425/jcykcai/sihengli/CKPT/stabilityai/stable-diffusion-xl-base-1.0"
    base = DiffusionPipeline.from_pretrained(
        ckpt, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to(device)
    
    ckpt = "/apdcephfs/share_733425/jcykcai/sihengli/CKPT/stabilityai/stable-diffusion-xl-refiner-1.0"
    refiner = DiffusionPipeline.from_pretrained(
        ckpt,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to(device)
    
    generator = torch.Generator(device="cuda").manual_seed(53)
    
    prompt = "Hi, I'm looking for inspiration on how to relax outdoors while still being productive."
    
    image = decode_with_sdxl(prompt, base, refiner, generator)
    
    image.save("./data/relax_outdoors.png")