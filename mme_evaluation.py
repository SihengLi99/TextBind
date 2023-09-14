import os
import time
import json
import torch
import wandb
import argparse
import deepspeed
import numpy as np
from tqdm import tqdm
from PIL import Image
from accelerate import infer_auto_device_map, dispatch_model
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel, CLIPProcessor, LlamaTokenizer, Blip2Processor

from utils import parse_args
from inference import MIMPipeline

def load_mme():
    
    total_corpus = {}
    total_num = 0
    for dir_name in os.listdir("./MME_Benchmark_release_version"):
        if dir_name in ["readme.txt", "eval_tool"]:
            continue
        
        image_dir = os.path.join("./MME_Benchmark_release_version", dir_name)
        if dir_name == "artwork":
            image_dir = os.path.join(image_dir, "./images/toy_dataset")
        elif os.path.exists(os.path.join(image_dir, "images")):
            image_dir = os.path.join(image_dir, "images")
        
        question_dir = os.path.join("./MME_Benchmark_release_version", dir_name)
        if os.path.exists(os.path.join(question_dir, "questions_answers_YN")):
            question_dir = os.path.join(question_dir, "questions_answers_YN")
        
        corpus = []
        for file in os.listdir(question_dir):
            if ".txt" in file:
                with open(os.path.join(question_dir, file), "r") as f:
                    questions_answers = [line.split("\t") for line in f.readlines()]
                    for question, answer in questions_answers:
                        
                        corpus.append({
                            "image_id": file.split(".txt")[0],
                            "image_dir": image_dir,
                            "image": file.replace(".txt", ".jpg") if os.path.exists(os.path.join(image_dir, file.replace(".txt", ".jpg"))) else file.replace(".txt", ".png"),
                            "question": question.strip(),
                            "answer": answer.strip()
                        })
        
        print(f"dir_name: {dir_name}, corpus length: {len(corpus)}")
        total_corpus[dir_name] = corpus
        total_num += len(corpus)
        
    print(f"total corpus length: {len(total_corpus)}")
    print(f"total num: {total_num}")
    
    return total_corpus

def evaluate(args, agent):

    corpus = load_mme()
    for id1, subject_name in enumerate(corpus):
        print(f"subject_name: {subject_name}")
        for id2, data in enumerate(tqdm(corpus[subject_name])):
            image = data["image"]
            text = data["question"]
            args.image_dir = data["image_dir"]
            
            inference_data = {
                "conversation": [
                    {
                        "role": "user",
                        "content": f"<image> {text}",
                        "image_list": [image],
                        "caption_list": [],
                    }
                ],
                "image_dir": data["image_dir"]
            }
            
            inference_data = agent.run(inference_data)
            data["prediction"] = inference_data["conversation"][-1]["content"]
            print(f"generation: ", data["prediction"])
    
    if not os.path.exists(args.inference_dir):
        os.makedirs(args.inference_dir)
    
    for subject_name in corpus:
        
        with open(os.path.join("./MME_Benchmark_release_version/eval_tool/Your_Results", f"{subject_name}.txt"), "r") as f:
            groundtruth = f.readlines()
        
        predictions = corpus[subject_name]
        with open(os.path.join(args.inference_dir, f"{subject_name}.txt"), "w") as f:
            for line in groundtruth:
                line = line.strip()
                image_name, question, groundtruth_answer = line.split("\t")
                
                for data in predictions:
                    if data["image"] == image_name and data["question"] == question:
                        prediction_answer = data["prediction"]
                
                f.write(f"{image_name}\t{question}\t{groundtruth_answer}\t{prediction_answer}\n")    
    
if __name__ == "__main__":
    
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = MIMPipeline(args, device)
    
    evaluate(args, agent)
