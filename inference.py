import os
import re
import time
import copy
import json
import uuid
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import LlamaTokenizer, Blip2Processor

from model import MIMModel
from data_utils import preprocess_mim
from utils import parse_args, build_model_and_processor

def postprocesss(s, args):

    s = s.replace("<image>" * args.num_query_tokens, "<image>")
    s = s.replace(" ".join(["<image>"] * args.num_query_tokens), "<image>")
    s = s.replace("  ", " ")
    pattern = "<start>.*?<end>"
    s = re.sub(pattern, "<image>", s)
    return s

def inference(
    data, 
    args,
    model,
    tokenizer,
    image_processor,
    sd_base,
    sd_refiner,
    generator,
    device,
    ):
    
    # process mim input
    inputs = preprocess_mim(data, tokenizer, image_processor, args, False)        
    
    input_ids = torch.LongTensor(inputs["input_ids"]).unsqueeze(0).to(device)
    input_images = (None if inputs["input_images"] is None else torch.stack(inputs["input_images"], 0).unsqueeze(0).to(device))
    input_image_index = (None if inputs["input_image_index"] is None else torch.LongTensor(inputs["input_image_index"]).unsqueeze(0).to(device)) 
    
    # print("Using none cache generation")
    # time1 = time.time()
    # outputs = model.none_cache_generation(
    #     input_ids=input_ids,
    #     input_images=input_images,
    #     input_image_index=input_image_index,
    #     prompt_token_index=prompt_token_index,
    #     image_start_id=args.image_start_id,
    #     image_token_id=args.image_token_id,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     sd_pipe=sd_pipe,
    #     generator=generator,
    #     generate_image=args.generate_image,
    #     max_output_length=args.max_output_length,
    # )
    # time2 = time.time()
    # print("Time: ", time2 - time1)
    
    # time1 = time.time()
    # print("Using cache generation")
    outputs = model.cache_generation(
        input_ids=input_ids,
        input_images=input_images,
        input_image_index=input_image_index,
        caption_start_id=args.caption_start_id,
        caption_end_id=args.caption_end_id,
        tokenizer=tokenizer,
        image_processor=image_processor,
        sd_base=sd_base,
        sd_refiner=sd_refiner,
        generator=generator,
        generate_image=args.generate_image,
        max_output_length=args.max_output_length,
        top_p=args.top_p,
    )
    # time2 = time.time()
    # print("Time cost: ", time2 - time1)
    
    total_text = tokenizer.decode(outputs["sequences"][0].tolist(), skip_special_tokens=False)
    generation = total_text.split("[/INST]")[-1].strip()
    
    image_list = []
    for image in outputs["image_list"]:
        name = uuid.uuid4()
        image.save(os.path.join(args.inference_dir, f"{name}.png"))
        image_list.append(f"{name}.png")
                
    data["conversation"].append(
        {
            "role": "assistant",
            "content": generation,
            "image_list": image_list,
            "caption_list": outputs["caption_list"],
            "tags": "generation"
        }
    )
    
    for turn in data["conversation"]:
        turn['content'] = postprocesss(turn['content'], args)
        
    return data

class MIMPipeline:
    def __init__(self, args, device):

        model, tokenizer, image_processor = build_model_and_processor(args)

        model = model.half().to(device) if args.fp16 else model.to(device)
        model.eval()
        
        if args.model_list:
            print("Loading engine list from {}...".format(args.model_list))
            engines = json.load(open(args.model_list))

            model_list = {}
            for m_info in engines:
                checkpoint = m_info['path']
                model_id = m_info['id']

                _model = copy.deepcopy(model)
                _model.load_state_dict(torch.load(checkpoint), strict=False)
                _model.eval()
                
                
                del _model.language_model
                del _model.vision_model
                
                model_list[model_id] = {}
                for module in args.save_modules:
                    model_list[model_id][module] = getattr(_model, module).half().to(device) if args.fp16 else getattr(_model, module).to(device)

                print(f"Load checkpoint: {checkpoint}")
        else:
            print("Engine list was not provided...")
            model_list = {}
            engines = []
            

        sd_base = DiffusionPipeline.from_pretrained(
            args.sd_base, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(device)
        
        sd_refiner = DiffusionPipeline.from_pretrained(
            args.sd_refiner,
            text_encoder_2=sd_base.text_encoder_2,
            vae=sd_base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)
        
        generator = torch.Generator(device=device).manual_seed(42)

        if not os.path.exists(args.inference_dir):
            os.makedirs(args.inference_dir)

        self.args = args
        self.device = device
        self.model = model
        self.model_list = model_list
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.sd_base = sd_base 
        self.sd_refiner = sd_refiner
        self.generator = generator
        self.engines = engines
    
    def run(self, data, selection=None):
        if selection and len(self.model_list) > 0:
            for module in self.args.save_modules:
                setattr(self.model, module, self.model_list[selection][module])
                print("Loaded module: %s from model %s" % (module, selection))
        
        data = inference(data, self.args, self.model, self.tokenizer, self.image_processor, self.sd_base, self.sd_refiner, self.generator, self.device)
        
        return data

def evaluate(args, device):
    agent = MIMPipeline(args, device)

    corpus = json.load(open(args.val_data_path, "r"))
        
    inference_results = []
    for idx, data in enumerate(corpus):
        # if len(data["conversation"]) <= 5:
        #     continue
        # data["conversation"] = data["conversation"][:5]
        
        data["conversation"] = data["conversation"][:-1]
        data = agent.run(data, None)
        
        # save the input image to the inference directory for checking
        for turn in data["conversation"][:-1]:
            for image_path in turn ["image_list"]:
                image = Image.open(os.path.join(data["image_dir"], image_path)).convert("RGB")
                image.save(os.path.join(args.inference_dir, image_path))
            
        inference_results.append(data)
        print(data)
        print("=====================================")
    json.dump(inference_results, open(os.path.join(args.inference_dir, "inference_results.json"), "w"), indent=4)
    
if __name__ == "__main__":

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args, device)
