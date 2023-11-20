# -*- encoding: utf-8 -*-
"""
A model worker executes the model.
"""
import argparse
import asyncio
import dataclasses
import logging
import json
import time
from typing import List, Union
import threading
import uuid
from tqdm import tqdm

import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from functools import partial
import base64
import io
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, InterpolationMode

import os

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
global_counter = 0

model_semaphore = None


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class MMEDataset(Dataset):
    def __init__(self,
                 data_folder,
                 category,):
        self.data_folder = data_folder
        category_root = f'{self.data_folder}/{category}'
        if os.path.exists(f'{category_root}/images'):
            category_root = f'{category_root}/images'
        with open(f'{self.data_folder}/eval_tool/Your_Results/{category}.txt',"r",encoding="utf-8") as f:
            lines = f.readlines()
                
        self.filelist = []
        for line in lines:
            line = line.strip()
            img_id, question, gt_answer = line.split('\t')
            self.filelist.append({'image': f'{category_root}/{img_id}', 'question': question, 'gt_answer': gt_answer})

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        item = self.filelist[idx]
        image = item['image']
        # image = decode_base64_to_image(image)
        question = item['question']
        gt_answer = item['gt_answer']

        # hint = self.load_from_df(idx, 'hint')
        
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
        data = {
            'img': image,
            'question': question,
            'gt_answer': gt_answer,
            'prompt': prompt,
        }
        return data
    
    def load_from_df(self, idx, key):
            if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
                return self.df.iloc[idx][key]
            else:
                return None
    

def insert_fixed_prefix(split_str,prefix,prompt):
    res = prompt.rsplit(split_str,1)
    res = split_str.join(res[:1],[prefix + res[1]])
    return res


def get_substring_after_second_last_occurrence(string, target):
    target_indices = [i for i in range(len(string)) if string.startswith(target, i)]
    start_index = target_indices[-2] + len(target)
    return string[start_index:]


def insert_after_last_substring(original_string, substring, string_to_insert):
    index = original_string.rfind(substring)
    index_after_substring = index + len(substring)
    return original_string[:index_after_substring] + string_to_insert + original_string[index_after_substring:]


def prepare_envs(folder, dtype='bf16'):
    import sys
    sys.path.append("./third_party/EVA-CLIP/rei")
    sys.path.append("./model_zoo/hf_models")
    from eva_clip import create_model_and_transforms
    model_name = "EVA02-CLIP-L-14-336to672"
    vision_tower = "./model_zoo/EVA02_CLIP_L_336to672_psz14_s6B.pt"
    _, _, image_processor = create_model_and_transforms(model_name, vision_tower, force_custom_clip=True)

    from modeling_internmlm import InternMLMForCausalLM
    if dtype == 'bf16':
        model = InternMLMForCausalLM.from_pretrained(folder, torch_dtype=torch.bfloat16).cuda()
    elif dtype == 'fp32':
        model = InternMLMForCausalLM.from_pretrained(folder, torch_dtype=torch.float32).cuda()
        model.model.vision_tower.type(torch.float32)
        # model.model.vision_tower.type(torch.float32)
    elif dtype == 'fp16':
        model = InternMLMForCausalLM.from_pretrained(folder, torch_dtype=torch.bfloat16).cuda()
    else:
        raise ValueError('dtype must be bf16, fp16, or fp32.')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)

    return image_processor, tokenizer, model

@torch.inference_mode()
def inference(img_pth, prompt, image_processor, tokenizer, model,beam_search=False, dtype='bf16'):
    if dtype == 'bf16':
        dtype = torch.bfloat16
    elif dtype == 'fp32':
        dtype = torch.float32
    elif dtype == 'fp16':
        dtype = torch.float16
    else:
        raise ValueError('dtype must be bf16, fp16, or fp32.')

    with open(img_pth, 'rb') as fp:
        image = Image.open(io.BytesIO(fp.read()))
        image = Resize((672, 672), interpolation=InterpolationMode.BICUBIC)(image)
        image_tk = image_processor(image)
        image_tk = image_tk.unsqueeze(0).cuda().to(dtype)

    prompt = "<|User|>:" + prompt + "<eoh>\n"
    human_ids = tokenizer.encode(prompt)
    # human_ids += [103167, 13]  # add special tokens + \n
    human_ids += tokenizer.encode("<|Bot|>:")[1:]
    #human_ids = human_ids[0:6] + [1] * 577 + human_ids[6:]  # add image features placeholder
    human_ids = human_ids[0:1] + [1] * 256 + human_ids[1:]
    inputs = torch.tensor(human_ids)
    inputs = inputs.unsqueeze(dim=0).cuda()

    if(beam_search):
        generate_ids = model.generate(inputs, max_new_tokens=512, images=image_tk, num_beams=5,
                                    length_penalty=20.0,
                                    num_return_sequences=1,
                                    )
    else:
        generate_ids = model.generate(inputs, max_new_tokens=512, images=image_tk)

    pure_output = generate_ids[:, inputs.shape[-1]:]
    pure_output = tokenizer.batch_decode(pure_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    pure_output = pure_output.replace("<TOKENS_UNUSED_139>", "")
    pure_output = pure_output.replace("<eoa>", "")
    
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], pure_output




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./model_zoo/hf_models")
    parser.add_argument("--benchmark", type=str, default="./MME_Benchmark_release_version")
    parser.add_argument("--output_dir", type=str, default="./tmp")
    parser.add_argument("--beam_search", action="store_true")
    parser.add_argument("--dtype", type=str, default='bf16')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dtype == 'bf16':
        dtype = torch.bfloat16
    elif args.dtype == 'fp32':
        dtype = torch.float32
    elif args.dtype == 'fp16':
        dtype = torch.float16
    else:
        raise ValueError('dtype must be bf16, fp16, or fp32.')

    image_processor, tokenizer, model = prepare_envs(args.model_path, args.dtype)
    model = model.to(dtype)
    
    default_categories = [category_file.split('.')[0] for category_file in os.listdir(f'{args.benchmark}/eval_tool/Your_Results')]
    for category in default_categories:
        benchmark = MMEDataset(args.benchmark, category)
        outputs = {}
        # conv = conv_templates[args.conv_template].copy()
        result_file = f'{args.output_dir}/{category}.txt'
        lines = []
        print(f"evaluate on ", category)
        for cnt, sample in enumerate(tqdm(benchmark)):
            prompt = sample['prompt']
            img = sample['img']
            gt_answer = sample['gt_answer']
            _, cur_out = inference(img_pth=img, prompt=prompt, image_processor=image_processor, tokenizer=tokenizer, model=model,beam_search=args.beam_search, dtype=args.dtype)
            
            cur_out = cur_out.replace('\n', ' ')
            cur_out = cur_out.replace('\t', ' ')
            print(cur_out)
            print('============================================================================')
            lines.append(f'{img.split("/")[-1]}\t{sample["question"]}\t{gt_answer}\t{cur_out}\n')
        with open(result_file, 'w',encoding="utf-8") as f:
            f.writelines(lines)
