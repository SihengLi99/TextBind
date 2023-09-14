import os
import io
import torch
import base64
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import parse_args
from inference import MIMPipeline

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image)
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')
        data = {
            'img': image,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_dict': options,
            'index': index,
            'context': hint,
        }
        return data
    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None
        
def evaluate(args, agent):

    dataset = MMBenchDataset("./data/mmbench_test_20230712.tsv")
    results = []
    for data in tqdm(dataset):

        if data['context'] is not None:
            prompt = data['context'] + ' ' + data['question'] + ' ' + data['options']
        else:
            prompt = data['question'] + ' ' + data['options']
        
        inference_data = {
            "conversation": [
                {
                    "role": "user",
                    "content": f"<image> {prompt}",
                    "image_list": [data["img"]],
                    "caption_list": [],
                }
            ],
        }
        
        inference_data = agent.run(inference_data)
        prediction = inference_data["conversation"][-1]["content"].replace("</s>", "").strip()
        # print(f"prediction: ", prediction)
        
        options = data["options"].split("\n")
        results.append({
            "question": data["question"],
            "A": options[0] if len(options) > 0 else "",
            "B": options[1] if len(options) > 1 else "",
            "C": options[2] if len(options) > 2 else "",
            "D": options[3] if len(options) > 3 else "",
            "prediction": prediction,
            "category": data["category"],
            "l2-category": data["l2-category"],
            "index": data["index"],
        })

    if not os.path.exists(args.inference_dir):
        os.makedirs(args.inference_dir)

    df = pd.DataFrame(results)
    df.to_excel(os.path.join(args.inference_dir, "submission.xlsx"), index=False)
        
if __name__ == "__main__":
    
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = MIMPipeline(args, device)
    
    evaluate(args, agent)
