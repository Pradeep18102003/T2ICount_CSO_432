import torch.nn as nn
import glob
from torch.utils.data import Dataset
import os
from pathlib import Path
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer
import torch


def _load_clip_tokenizer(version="openai/clip-vit-large-patch14"):
    candidates = [
        os.environ.get("T2ICOUNT_CLIP_PATH"),
        str(Path(__file__).resolve().parents[1] / "models" / "clip-tokenizer"),
        str(Path(__file__).resolve().parents[1] / "models" / "clip-vit-large-patch14"),
    ]

    try:
        return CLIPTokenizer.from_pretrained(version)
    except Exception:
        pass

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            try:
                return CLIPTokenizer.from_pretrained(candidate, local_files_only=True)
            except Exception:
                continue

    raise OSError(
        "Unable to load CLIP tokenizer from HuggingFace or local fallback paths. "
        f"Tried: {candidates}."
    )


class CARPK(Dataset):
    def __init__(self, root, info):
        self.im_list = sorted(glob.glob(os.path.join(root, 'Images/*.png')))
                
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.label_list = []
        try:
            with open(info) as f:
                for i in f:
                    self.label_list.append(i.strip())
        except Exception:
            raise Exception("The path to label list is incorrect.")
        
        split_list = []
        for i in self.im_list:
            if os.path.basename(i).replace('.png', '') in self.label_list:
                split_list.append(i)
        self.im_list = split_list
        self.tokenizer = _load_clip_tokenizer()

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):

        im_path = self.im_list[item]

        img = Image.open(im_path).convert('RGB')
        ann_path = im_path.replace('png', 'txt').replace('Images', 'Annotations')
        print(os.path.basename(im_path))

        with open(ann_path, 'r') as f:
            count = sum(1 for line in f)

        prompt = 'cars'

        prompt_attn_mask = torch.zeros(77)
        cls_name_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        cls_name_length = cls_name_tokens['input_ids'].shape[1]
        prompt_attn_mask[1: 1 + cls_name_length] = 1

        return self.transform(img), count, prompt, prompt_attn_mask, os.path.basename(im_path).split('.')[0]
