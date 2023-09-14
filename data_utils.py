import os
import csv
import sys
import json
import torch
import random
import logging
import argparse
import functools
import transformers
import numpy as np
from copy import deepcopy
from PIL import Image
from io import BytesIO
from functools import partial
from transformers import LlamaTokenizer
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List, Tuple, Callable
import webdataset as wds
from transformers import Blip2Processor

ImageCaptionTemplates = [
    "<image> Provide a short and precise description of the image displayed.",
    "Give a brief and accurate depiction of the picture shown. <image>",
    "<image> Present a concise and clear summary of the image seen.",
    "Offer a short and straightforward representation of the image provided. <image>",
    "<image> Share a brief, yet informative account of the picture presented.",
    "Deliver a concise and comprehensible explanation of the image shown. <image>",
    "<image> Express a succinct and clear narrative of the picture displayed.",
    "Convey a brief and unambiguous description of the image presented. <image>",
    "<image> Present a short and coherent account of the picture shown.",
    "Provide a compact and lucid representation of the image displayed. <image>",
    "<image> Give a brief and distinct explanation of the picture provided.",
    "Share a concise and easy-to-understand description of the image shown. <image>",
    "<image> Offer a clear and to-the-point narrative of the picture presented.",
    "Express a brief and well-defined interpretation of the image provided. <image>",
    "<image> Deliver a concise and intelligible account of the image displayed.",
    "Convey a short and easily grasped summary of the picture shown. <image>",
    "<image> Present a succinct and clear-cut representation of the image presented.",
    "Provide a brief and sharp explanation of the picture provided. <image>",
    "<image> Give a concise and articulate description of the image shown.",
    "Share a short and comprehensible narrative of the picture displayed. <image>"
]

ImageGenerationTemplates = [
    [
        "Now, I need you to creat an image for me, the main content is: {image_caption}, Thanks!",
        "My pleasure! Here you are <image>."
    ],
    [
        "Could you create an image for me with this content: {image_caption}? Thanks a lot!",
        "Here's the image you requested: <image>.",
    ],
    [
        "I'd appreciate if you have an image with this description: {image_caption}.",
        "Happy to help! Here's your image: <image>.",
    ],
    [   
        "Can you please create an image with the following caption: {image_caption}?", 
        "Sure thing! Here's the image you asked for: <image>.", 
    ], 
    [   
        "I was wondering if you could make an image based on this: {image_caption}.", 
        "Absolutely! Here's the image you're looking for: <image>.", 
    ], 
    [   
        "I need an image that represents this idea: {image_caption}. Can you help?", 
        "Of course! Here's the image that fits your description: <image>.", 
    ], 
    [   
        "Please create an image for me with this concept: {image_caption}.", 
        "No problem! Here's the image you requested: <image>.", 
    ], 
    [   
        "It would be great if you could make an image with this theme: {image_caption}.", 
        "I'm happy to help! Here's the image based on your theme: <image>.", 
    ], 
    [ 
        "I'd love to see an image that captures this: {image_caption}.", 
        "I've got you covered! Check out this image: <image>.", 
    ], 
    [ 
        "Can you come up with an image that has this content: {image_caption}?",
        "Sure! Here's an image with the content you described: <image>.", 
    ], 
    [ 
        "Please provide me with an image that showcases this: {image_caption}.",
        "Here's an image that showcases your request: <image>.", 
    ], 
    [   
        "I'm looking for an image that depicts this: {image_caption}. Can you help?",
        "Certainly! Here's an image that depicts your request: <image>.",
    ], 
    [ 
        "Could you generate an image based on this idea: {image_caption}?", 
        "Here's an image generated based on your idea: <image>.", 
    ], 
    [ 
        "I'd be grateful if you could create an image with this subject: {image_caption}.", 
        "I'm happy to assist! Here's the image with the subject you mentioned: <image>.",
    ], 
    [ 
        "Can you design an image for me that includes this: {image_caption}?",
        "I'd be happy to! Here's the image that includes your request: <image>.", 
    ], 
    [   
        "Please make an image for me that captures this essence: {image_caption}.", 
        "Here's an image that captures the essence you described: <image>.", 
    ], 
    [ 
        "I need an image that conveys this message: {image_caption}. Can you create one?", 
        "Sure! Here's an image that conveys the message you described: <image>.",
    ], 
    [ 
        "I'd like to see an image that represents this concept: {image_caption}.", 
        "Here's an image that represents the concept you mentioned: <image>.", 
    ], 
    [ 
        "Can you please come up with an image that illustrates this: {image_caption}?",
        "Here's an image that illustrates your request: <image>.", 
    ], 
    [ 
        "I'm looking for an image that embodies this idea: {image_caption}. Can you make one?", 
        "Of course! Here's an image that embodies the idea you described: <image>.", 
    ],    
]


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

def preprocess_llama(tokenizer, dialog, training=True):
    input_ids = []
    labels = []
    input_images = []
    output_image_list = []
    output_caption_list = []
    
    unsafe = any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
                "image_list": dialog[0]['image_list'] + dialog[1]['image_list'],
            }
        ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    
    for prompt, answer in zip(dialog[::2], dialog[1::2]):
        prompt_ids = tokenizer.encode(
            f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}",
            add_special_tokens=False
        )
        answer_ids = tokenizer.encode(
            f"{(answer['content']).strip()} {tokenizer.eos_token}",
            add_special_tokens=False
        )
        input_ids = input_ids + prompt_ids + answer_ids
        labels = labels + [-100] * len(prompt_ids) + answer_ids
        input_images = input_images + prompt['image_list'] + answer['image_list']
        output_image_list = output_image_list + answer['image_list']
        output_caption_list = output_caption_list + answer["caption_list"]
    
    if not training:
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        prompt_ids = tokenizer.encode(
            f"{tokenizer.bos_token}{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            add_special_tokens=False
        )
        input_ids = input_ids + prompt_ids
        labels = labels + [-100] * len(prompt_ids)
        input_images = input_images + dialog[-1]['image_list']
        
    return input_ids, labels, input_images, output_image_list, output_caption_list

def preprocess_mim(data, tokenizer, image_processor, args, training):
        
    # setting placeholder for image tokens
    # when generating <image>, it means that we need to generate an image
    # num_query_tokens is the number of image tokens 
    image_placeholder = "".join(["<image>"] * args.num_query_tokens)
    
    for turn in data["conversation"]:
        turn['content'] = turn['content'].replace("<image>", image_placeholder)
    input_ids, labels, input_images, output_image_list, output_caption_list = preprocess_llama(tokenizer, data["conversation"], training)
    input_image_index = [idx for idx, value in enumerate(input_ids) if value == args.image_token_id]
        
    # image processing for both input image and output image;
    # for mim dataset, we need to load the image from disk
    if len(input_images) > 0 and isinstance(input_images[0], str):
        input_images = [
            Image.open(
                os.path.join(data["image_dir"] if "image_dir" in data else args.image_dir, image)
            ).convert("RGB")
            for image in input_images
        ]

    input_images = [
        image_processor(
            images=image,
            return_tensors='pt'
        )['pixel_values'][0]
        for image in input_images
    ]
    
    if not training:
        return {
            "input_ids": input_ids,
            "input_images": input_images if len(input_images) > 0 else None,
            "input_image_index": input_image_index if len(input_image_index) > 0 else None,
        }

    idx = 0
    seqlen = len(input_ids)
    attention_mask = [[1] * seqlen for _ in range(seqlen)]
    position_ids = list(range(seqlen))
    cur_image_index = 0
    while idx < seqlen:
        # caption generation start
        if input_ids[idx] == args.image_token_id and labels[idx] != -100:
            
            caption = output_caption_list[cur_image_index]
            caption_ids = [args.caption_start_id] + tokenizer.encode(caption, add_special_tokens=False) + [args.caption_end_id]
            cur_image_index += 1
            
            input_ids.extend(caption_ids)
            labels[idx] = args.caption_start_id
            labels[idx+1:idx+args.num_query_tokens] = [-100] * (args.num_query_tokens-1)
            labels.extend([-100 if idx==0 else token_id for idx, token_id in enumerate(caption_ids)])
            attention_mask.extend([[1] * idx + [0] * (seqlen-idx)] * len(caption_ids))
            position_ids.extend(list(range(idx, idx+len(caption_ids))))
            
            idx += args.num_query_tokens
        else:
            idx += 1
    
    if len(input_ids) > args.max_input_length:
        
        # do not truncate in the image tokens
        end_position = args.max_input_length
        while input_ids[end_position] == args.image_token_id:
            end_position += 1
        
        input_ids = input_ids[:end_position]
        labels = labels[:end_position]
        attention_mask = [mask[:end_position] for mask in attention_mask[:end_position]]
        position_ids = position_ids[:end_position]
        
        input_images = input_images[:(input_ids.count(args.image_token_id) // args.num_query_tokens)]
        input_image_index = [idx for idx, value in enumerate(input_ids) if value == args.image_token_id]
        
    assert len(input_ids) == len(labels) == len(attention_mask) == len(position_ids)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "input_images": input_images,
        "input_image_index": input_image_index,
    }

def collate_fn(batch: List[Dict[str, torch.Tensor]], args) -> Dict[str, torch.Tensor]:
    
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(x["input_ids"]) for x in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(x["labels"]) for x in batch], batch_first=True, padding_value=-100)
    position_ids = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(x["position_ids"]) for x in batch], batch_first=True, padding_value=0)
    
    bsz, seqlen = input_ids.shape

    attention_mask = torch.ones(bsz, seqlen, seqlen, dtype=torch.int64)
    for i, x in enumerate(batch):
        for j, mask in enumerate(x["attention_mask"]):
            attention_mask[i, j, : len(mask)] = torch.LongTensor(mask)
    
    # since we really do not care the last token
    input_image_index = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(x["input_image_index"]) for x in batch], batch_first=True, padding_value=seqlen-1)
    if input_image_index.shape[-1] == 0:
        input_image_index = torch.nn.utils.rnn.pad_sequence([torch.LongTensor([seqlen-1] * args.num_query_tokens) for x in batch], batch_first=True, padding_value=seqlen-1)
    
    max_input_images = max(1, max( len(x["input_images"]) for x in batch))
    input_images = torch.zeros(bsz, max_input_images, 3, 224, 224)
    for i, x in enumerate(batch):
        for j, image in enumerate(x["input_images"]):
            input_images[i, j] = image
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "input_images": input_images,
        "input_image_index": input_image_index,
    }


class MIMDataset(Dataset):
    def __init__(self, args: argparse.Namespace,
                 data_path: str,                 
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor: transformers.CLIPImageProcessor):
        super(MIMDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data = deepcopy(self.list_data_dict[i])
        return preprocess_mim(data, self.tokenizer, self.image_processor, self.args, True)


def webdataset_map(example, args, image_processor, tokenizer):

    image = Image.open(BytesIO(example["image"])).convert("RGB")
    caption = example["caption"].decode("utf-8")
    key = json.loads(example["meta"])["key"]
    
    template = random.choice(ImageCaptionTemplates)
    
    data = {
        "conversation": [
            {
                "role": "user",
                "content": template,
                "image_list": [image], 
                "caption_list": [caption],
            },
            {
                "role": "assistant",
                "content": caption,
                "image_list": [],
                "caption_list": [],
            }
        ]
    }
        
    return preprocess_mim(data, tokenizer, image_processor, args, True)

def load_mim_dataset(args: argparse.Namespace,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor: transformers.CLIPImageProcessor,
                 ) -> Tuple[torch.utils.data.Dataset, Callable]:
    
    dataset = MIMDataset(args, (args.train_data_path if args.train else args.val_data_path), tokenizer, image_processor)

    return dataset, collate_fn
   
def load_pair_dataset(args: argparse.Namespace,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor: transformers.CLIPImageProcessor,
                 ) -> Tuple[torch.utils.data.Dataset, Callable]:
    """Load dataset."""
    
    if args.train:
        dataset = wds.WebDataset(args.train_data_path, resampled=True) \
                    .shuffle(1000) \
                    .rename(image="jpg;png", meta="json", caption="txt") \
                    .map(functools.partial(webdataset_map, args=args, tokenizer=tokenizer, image_processor=image_processor)) \
                    .with_epoch(args.with_epoch // (args.world_size * args.with_num_works) ) 
    else:
        dataset = wds.WebDataset(args.val_data_path) \
                    .rename(image="jpg;png", meta="json", caption="txt") \
                    .map(functools.partial(webdataset_map, args=args, tokenizer=tokenizer, image_processor=image_processor))       
    
    return dataset, collate_fn

if __name__ == "__main__":

    class Dummy:
        def __call__(self, images, return_tensors):
            return {'pixel_values': [torch.zeros(3, 224, 224)]}
    image_processor = Dummy()

    args = argparse.Namespace(num_query_tokens=32, max_num_images=5, language_model="../../CKPT/meta-llama/Llama-2-7b-chat-hf", train=True)
    tokenizer = LlamaTokenizer.from_pretrained(args.language_model)
    add_tokens = ["<image>", "<start>", "<end>"]
    tokenizer.add_special_tokens(({"additional_special_tokens": add_tokens}))    
    args.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    args.caption_start_id = tokenizer.convert_tokens_to_ids("<start>")
    args.caption_end_id = tokenizer.convert_tokens_to_ids("<end>")
    args.num_new_tokens = len(add_tokens)
    # check mim dataset
    args.train_data_path = "./data/multi_instruct_processed_1000_54281.json"    
    args.num_prompt_tokens = 8
    args.max_input_length = 768
    dataset, collate_fn = load_mim_dataset(args, tokenizer, image_processor)
    
    # data = dataset[1]
    # print (data["input_ids"])
    # print (tokenizer.decode(data['input_ids']))
    # print ()
    # print (tokenizer.decode( [x if x >0 else tokenizer.unk_token_id for x in data['labels'] ]) )
    
    # print (data["position_ids"])
    # exit()
    
    corpus = json.load(open(args.train_data_path, "r"))
    print("Number of data:", len(corpus))
    data = dataset[0]
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=8, collate_fn=partial(collate_fn, args=args), shuffle=True)
    print("Number of batch:", len(dataloader))
    length = []
    total_cnt = 0
    large_cnt = 0
    for idx, batch in enumerate(dataloader):
        length.append(batch["input_ids"].shape[-1])
        print("Average length: ", np.mean(length))
        
        total_cnt += 1
        if length[-1] > 600:
            large_cnt += 1
    print(total_cnt)
    print(large_cnt)
        
    # args.train_data_path = "./data/minigpt4/cc_sbu/cc_sbu_dataset/{00000..01254}.tar"
    # args.with_epoch = 1000000
    # args.sd_latents_dir = "./data/sd2_prompt_embeds"
    # args.max_input_length = 512
    # args.num_prompt_tokens = 8
    # dataset, collate_fn = load_pair_dataset(args, tokenizer, image_processor)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4, collate_fn=partial(collate_fn, args=args))
    # total_num = 0
    # large_than_256 = 0
    # for batch in dataloader:
    #     total_num += 1
    #     if batch["input_ids"].shape[1] > 256:
    #         large_than_256 += 1
    #     print("Total: {}, Large than 256: {}".format(total_num, large_than_256))