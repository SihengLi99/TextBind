import os
import re
import json
import copy
from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image


def process_m3it():
    proxies = {
        "http": "http://127.0.0.1:8118",
        "https": "http://127.0.0.1:8118",
    }
    ds_name = "coco"  # change the dataset name here
    dataset = load_dataset("MMInstruction/M3IT", ds_name, cache_dir="/apdcephfs/share_733425/jcykcai/sihengli/Dataset")

    # for train_instance in dataset['train']:
    #     instruction = train_instance["instruction"]  # str
    #     inputs = train_instance["inputs"]  # str
    #     outputs = train_instance["outputs"]  # str
    #     image_base64_str_list = train_instance["image_base64_str"]  # str (base64)
    #     image_0 = Image.open(BytesIO(b64decode(image_base64_str_list[0])))
    
def process_llava():
    
    corpus = json.load(open("./data/llava_instruct_150k.json"))
    new_corpus = []
    
    for id1, data in enumerate(corpus):
        new_data = {"conversation": [], "image_dir": "./data/train2017"}
        for id2, turn in enumerate(data["conversations"]):
            new_turn = {}
            if turn["from"] == "human":
                new_turn["role"] = "user"
            else:
                new_turn["role"] = "assistant"
            new_turn["content"] = turn["value"]
            if "<image>" in turn["value"]:
                new_turn["image_list"] = [data["image"]]
            else:
                new_turn["image_list"] = []
                
            new_turn["caption_list"] = []
            
            new_data["conversation"].append(new_turn)
        new_corpus.append(new_data)
    
    print("Number of data: {}".format(len(new_corpus)))
    json.dump(new_corpus, open("./data/llava_processed.json", "w"), indent=4)
    

def process_alpaca_gpt4():

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    corpus = json.load(open("./data/alpaca_gpt4_data.json"))
    new_corpus = []
    
    for idx, data in enumerate(corpus):
        
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        
        user_content = prompt_input.format_map(data) if data.get("input", "") != "" else prompt_no_input.format_map(data)
        user_turn = {"role": "user", "content": user_content, "image_list": [], "caption_list": []}
        
        assistant_turn = {"role": "assistant", "content": data["output"], "image_list": [], "caption_list": []}
        
        new_data = {"conversation": [user_turn, assistant_turn]}
        new_corpus.append(new_data)
        
    print("Number of data: {}".format(len(new_corpus)))
    json.dump(new_corpus, open("./data/alpaca_gpt4_processed.json", "w"), indent=4)


def process_minigpt4():
    
    corpus = json.load(open("./data/cc_sbu_align/filter_cap.json"))["annotations"]
    new_corpus = []
    for idx, data in enumerate(corpus):
        
        user_content = "<image>\nDescribe this image in detail. Give as many details as possible. Say everything you see."
        user_turn = {"role": "user", "content": user_content, "image_list": [data["image_id"] + ".jpg"], "caption_list": []}
        
        assistant_turn = {"role": "assistant", "content": data["caption"], "image_list": [], "caption_list": []}
        
        new_data = {"conversation": [user_turn, assistant_turn], "image_dir": "./data/minigpt4_images"}
        new_corpus.append(new_data)
    
    print("Number of data: {}".format(len(new_corpus)))
    json.dump(new_corpus, open("./data/minigpt4_processed.json", "w"), indent=4)
    
    
def process_platypus():

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }

    dataset = load_dataset("garage-bAInd/Open-Platypus")["train"]
    new_corpus = []
    
    for idx, data in enumerate(dataset):
        
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        
        user_content = prompt_input.format_map(data) if data.get("input", "") != "" else prompt_no_input.format_map(data)
        user_turn = {"role": "user", "content": user_content.replace("<image>", ""), "image_list": [], "caption_list": []}
        
        assistant_turn = {"role": "assistant", "content": data["output"].replace("<image>", ""), "image_list": [], "caption_list": []}
        
        new_data = {"conversation": [user_turn, assistant_turn]}
        new_corpus.append(new_data)
        
    print("Number of data: {}".format(len(new_corpus)))
    json.dump(new_corpus, open("./data/platypus_processed.json", "w"), indent=4)


def process_multi_instruct():
            
    corpus = []
    per_dataset_num = 5000
    for path in [f"./data/train_{per_dataset_num}.jsonl", f"./data/test_{per_dataset_num}.jsonl"]:
        with open(path, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                corpus.append(json_obj)          

    new_corpus = []
    for idx, data in enumerate(corpus):
        
        user_turn = {"role": "user", "content": "<image>\n"+data["prompt"], "image_list": [data["image_path"]], "caption_list": []}
        
        assistant_turn = {"role": "assistant", "content": data["target"], "image_list": [], "caption_list": []}
        
        new_data = {"conversation": [user_turn, assistant_turn], "image_dir": "./data"}
        new_corpus.append(new_data)
        
    print("Number of data: {}".format(len(new_corpus)))
    json.dump(new_corpus, open(f"./data/multi_instruct_processed_{per_dataset_num}_{len(new_corpus)}.json", "w"), indent=4)
    
def process_shikra():
    
    corpus = []
    for path in ["./data/GPT4GEN_BoxCoT_test.jsonl", "./data/GPT4GEN_BoxCoT_train.jsonl", "./data/GPT4GEN_RD_BoxCoT_train.jsonl"]:
        with open(path, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                corpus.append(json_obj)
    
    print(len(corpus))    
    
    new_corpus = []
    for idx, data in enumerate(corpus):
        
        user_turn = {"role": "user", "content": "<image>\n"+data["question"].replace("<ph_st>", "").replace("<ph_ed>", "").replace("  ", "").replace(" ,", ",").replace(" .", "."), "image_list": [data["img_path"]], "caption_list": []}
        
        assistant_turn = {"role": "assistant", "content": data["cot_with_ans"].replace("<ph_st>", "").replace("<ph_ed>", "").replace("  ", "").replace(" ,", ",").replace(" .", "."), "image_list": [], "caption_list": []}
        
        new_data = {"conversation": [user_turn, assistant_turn], "image_dir": "./data/flickr30k-images"}
        new_corpus.append(new_data)
        
    print("Number of data: {}".format(len(new_corpus)))
    json.dump(new_corpus, open("./data/shikra_processed.json", "w"), indent=4)

def process_mim():
    
    corpus = json.load(open("./data/train.s2.v4.clean.reform.train.json", "r"))
    for idx, data in enumerate(corpus):
        data["image_dir"] = "./data/mim_images"
        data["conversation"] = data["conversation"][:-1]
        
        for turn in data["conversation"]:
            for idx, caption in enumerate(turn["caption_list"]):
                turn["caption_list"][idx] = re.sub(r'<img\d+>|<\/img\d+>', '', caption).strip()
        
    print("Number of data: {}".format(len(corpus)))
    json.dump(corpus, open("./data/mim_processed.json", "w"), indent=4)
    

def blender():
    
    num_llava = 0
    multi_instruct_data = "./data/multi_instruct_processed_2000_105745.json"
    
    corpus1 = []
    corpus2 = []
    corpus3 = []
    corpus4 = []
    corpus5 = []
    corpus6 = []
    
    corpus1 = json.load(open("./data/platypus_processed.json", "r"))
    corpus2 = json.load(open("./data/minigpt4_processed.json", "r"))
    corpus3 = json.load(open("./data/shikra_processed.json", "r"))
    corpus4 = json.load(open("./data/llava_processed.json", "r"))
    corpus5 = json.load(open(multi_instruct_data, "r"))
    corpus6 = json.load(open("./data/mim_processed.json", "r"))
    print("platypus: ", len(corpus1))
    print("minigpt4: ", len(corpus2))
    print("shikra: ", len(corpus3))
    print("llava: ", len(corpus4))
    print("multi_instruct: ", len(corpus5))
    print("mim: ", len(corpus6))
    
    final_data = f"./data/platypus_{len(corpus1)}_minigpt4_{len(corpus2)}_shikra_{len(corpus3)}_llava_{len(corpus4)}_multi_instruct_{len(corpus5)}_mim_{len(corpus6)}.json"
    
    corpus = corpus1 + corpus2 + corpus3 + corpus4 + corpus5 + corpus6
    print("final: ", len(corpus))
    json.dump(corpus, open(final_data, "w"), indent=4)


if __name__ == "__main__":
    
    # process_llava()
    # process_alpaca_gpt4()            
    # process_minigpt4()                 
    # process_platypus()
    # process_multi_instruct()    
    # process_shikra()
    # process_mim()
    
    blender()