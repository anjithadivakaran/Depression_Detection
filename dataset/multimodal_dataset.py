import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from tqdm import tqdm
import numpy as np
from utils import config

class MultimodelDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
        self.image_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx]
        label = user["label"]
        posts = user["posts"]

        # Sort by timestamp
        posts = sorted(posts, key=lambda x: datetime.strptime(x["timestamp"], "%a %b %d %H:%M:%S %z %Y"))[:config.MAX_SEQ_LEN]

        input_ids, attn_masks, images = [], [], []
        for post in posts:
            text = post.get("text", "")
            encoded = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            input_ids.append(encoded["input_ids"].squeeze(0))
            attn_masks.append(encoded["attention_mask"].squeeze(0))

            img_tensor = torch.zeros(3, config.IMG_SIZE, config.IMG_SIZE)
            if post.get("has_image") and os.path.exists(post["image_path"]):
                try:
                    img = Image.open(post["image_path"]).convert("RGB")
                    img_tensor = self.image_transform(img)
                except:
                    pass
            images.append(img_tensor)

        input_ids = torch.stack(input_ids)
        attn_masks = torch.stack(attn_masks)
        images = torch.stack(images)

        return input_ids, attn_masks, images, torch.tensor(label, dtype=torch.float32)