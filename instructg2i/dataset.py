import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image, ImageFile
from transformers import CLIPImageProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def read_metadata(img_dir, img_key, text_key, tokenizer):
    data = []
    with open(os.path.join(img_dir, 'metadata.jsonl')) as f:
        readin = f.readlines()
        for line in tqdm(readin):
            tmp = json.loads(line)
            
            # tokenize text
            tmp['input_ids'] = tokenizer(
                tmp[text_key], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            
            # read image
            img_path = os.path.join(img_dir, tmp[img_key])
            tmp['readin_image'] = Image.open(img_path).convert("RGB")
            data.append(tmp)
    return data

def read_metadata_graph(img_dir, img_key, text_key, neighbor_key, tokenizer):
    data = []
    with open(os.path.join(img_dir, 'metadata.jsonl')) as f:
        readin = f.readlines()
        for line in tqdm(readin):
            tmp = json.loads(line)
            
            # tokenize text
            tmp['input_ids'] = tokenizer(
                tmp[text_key], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            
            data.append(tmp)
    return data

class ImageTextDataset(Dataset):
    def __init__(self, img_dir, img_key, text_key, text_tokenizer, transform=None):
        self.meta_data = read_metadata(img_dir, img_key, text_key, text_tokenizer)
        self.img_dir = img_dir
        self.img_key = img_key
        self.text_key = text_key
        self.transform = transform
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        example = {}
        if self.transform:
            pixel_values = self.transform(self.meta_data[idx]['readin_image'])
        example["pixel_values"] = pixel_values
        example["input_ids"] = self.meta_data[idx]["input_ids"]
        return example


class GraphImageTextDataset(Dataset):
    def __init__(self, neighbor_num, img_dir, img_key, text_key, neighbor_key, text_tokenizer, center_transform, neighbor_transform):
        self.meta_data = read_metadata_graph(img_dir, img_key, text_key, neighbor_key, text_tokenizer)
        self.neighbor_num = neighbor_num
        self.img_dir = img_dir
        self.img_key = img_key
        self.text_key = text_key
        self.neighbor_key = neighbor_key
        self.center_transform = center_transform
        self.neighbor_transform = neighbor_transform
        self.text_tokenizer = text_tokenizer
        self.clip_image_processor = CLIPImageProcessor()

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        example = {}
        img_path = os.path.join(self.img_dir, self.meta_data[idx][self.img_key])
        image = Image.open(img_path).convert("RGB")

        neighbor_image = [Image.open(os.path.join(self.img_dir, n_file)).convert("RGB") for n_file in self.meta_data[idx][self.neighbor_key][:self.neighbor_num]]
        neighbor_mask = [1] * len(neighbor_image) + [0] * (self.neighbor_num - len(neighbor_image))
        neighbor_image += [Image.fromarray(np.uint8(np.zeros_like(np.array(image)))).convert('RGB')] * (self.neighbor_num - len(neighbor_image))
        
        pixel_values = self.center_transform(image)
        neighbor_pixel_values = [self.clip_image_processor(self.neighbor_transform(img), return_tensors="pt").pixel_values for img in neighbor_image]

        example["pixel_values"] = pixel_values
        example["input_ids"] = self.meta_data[idx]["input_ids"]
        example["neighbor_pixel_values"] = torch.stack(neighbor_pixel_values).squeeze(1)
        example["neighbor_mask"] = torch.LongTensor(neighbor_mask)
        return example


class ImageImageTextDataset(Dataset):
    def __init__(self, img_dir, img_key, text_key, neighbor_key, text_tokenizer, train_transforms):
        self.meta_data = read_metadata_graph(img_dir, img_key, text_key, neighbor_key, text_tokenizer)
        self.img_dir = img_dir
        self.img_key = img_key
        self.text_key = text_key
        self.neighbor_key = neighbor_key
        self.train_transforms = train_transforms
        self.text_tokenizer = text_tokenizer
        self.clip_image_processor = CLIPImageProcessor()

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        example = {}
        img_path = os.path.join(self.img_dir, self.meta_data[idx][self.img_key])
        image = Image.open(img_path).convert("RGB")
        if len(self.meta_data[idx][self.neighbor_key]) != 0:
            neighbor_image = Image.open(os.path.join(self.img_dir, self.meta_data[idx][self.neighbor_key][0])).convert("RGB")
        else:
            neighbor_image = Image.fromarray(np.uint8(np.zeros_like(np.array(image)))).convert('RGB')

        pixel_values = self.train_transforms(image)
        neighbor_pixel_values = self.train_transforms(neighbor_image)

        example["pixel_values"] = pixel_values
        example["input_ids"] = self.meta_data[idx]["input_ids"]
        example["neighbor_pixel_values"] = neighbor_pixel_values
        return example


class GraphImageTextDataset_p2p(Dataset):
    def __init__(self, neighbor_num, img_dir, img_key, text_key, neighbor_key, text_tokenizer, train_transforms):
        self.meta_data = read_metadata_graph(img_dir, img_key, text_key, neighbor_key, text_tokenizer)
        self.neighbor_num = neighbor_num
        self.img_dir = img_dir
        self.img_key = img_key
        self.text_key = text_key
        self.neighbor_key = neighbor_key
        self.train_transforms = train_transforms
        self.text_tokenizer = text_tokenizer
        self.clip_image_processor = CLIPImageProcessor()

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        example = {}
        img_path = os.path.join(self.img_dir, self.meta_data[idx][self.img_key])
        image = Image.open(img_path).convert("RGB")

        neighbor_image = [Image.open(os.path.join(self.img_dir, n_file)).convert("RGB") for n_file in self.meta_data[idx][self.neighbor_key][:self.neighbor_num]]
        neighbor_mask = [1] * len(neighbor_image) + [0] * (self.neighbor_num - len(neighbor_image))
        neighbor_image += [Image.fromarray(np.uint8(np.zeros_like(np.array(image)))).convert('RGB')] * (self.neighbor_num - len(neighbor_image))
        
        pixel_values = self.train_transforms(image)
        neighbor_pixel_values = [self.train_transforms(img) for img in neighbor_image]

        example["pixel_values"] = pixel_values
        example["input_ids"] = self.meta_data[idx]["input_ids"]
        example["neighbor_pixel_values"] = torch.stack(neighbor_pixel_values).squeeze(1)
        example["neighbor_mask"] = torch.LongTensor(neighbor_mask)
        return example
