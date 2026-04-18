import os.path
import os
from pathlib import Path
import time
import sys

from models.reg_model import Count
import torch
import numpy as np
from utils.tools import extract_patches, reassemble_patches
from transformers import CLIPTokenizer
from torchvision import transforms
from PIL import Image
import json


def _load_clip_tokenizer(version="openai/clip-vit-large-patch14"):
    candidates = [
        os.environ.get("T2ICOUNT_CLIP_PATH"),
        str(Path(__file__).resolve().parent / "models" / "clip-tokenizer"),
        str(Path(__file__).resolve().parent / "models" / "clip-vit-large-patch14"),
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


with open('FSC-147-S.json', 'r') as f:
    data = json.load(f)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


torch.manual_seed(15)
np.random.seed(15)

config = 'configs/v1-inference.yaml'
sd_path = 'configs/v1-5-pruned-emaonly.ckpt'
crop_size = 384
model = Count(config, sd_path, unet_config={'base_size': crop_size, 'max_attn_size': crop_size // 8,
                                            'attn_selector': 'down_cross+up_cross'})

model.load_state_dict(torch.load('best_model_paper.pth', map_location='cpu')) # Change the path to the pretrained weight
model = model.to('cpu')
model.clip.device = 'cpu'
tokenizer = _load_clip_tokenizer()

error = []
all_images = list(data.keys())
total_images = len(all_images)
start_time = time.time()

for idx, img_file in enumerate(all_images, start=1):
    gt = data[img_file]['count']
    cls_name = data[img_file]['class']
    prompt_attn_mask = torch.zeros(77)
    cls_name_tokens = tokenizer(cls_name, add_special_tokens=False, return_tensors='pt')
    cls_name_length = cls_name_tokens['input_ids'].shape[1]
    prompt_attn_mask[1: 1 + cls_name_length] = 1
    prompt_attn_mask = prompt_attn_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).to('cpu')
    cls_name = (cls_name,)
    img_path = 'data/FSC/images_384_VarV2/' + img_file

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    im = Image.open(img_path).convert('RGB')
    im = transform(im).unsqueeze(0).to('cpu')

    cropped_imgs, num_h, num_w = extract_patches(im, patch_size=384, stride=384)
    outputs = []

    with torch.set_grad_enabled(False):
        num_chunks = (cropped_imgs.size(0) + 4 - 1) // 4
        for i in range(num_chunks):
            start_idx = i * 4
            end_idx = min((i + 1) * 4, cropped_imgs.size(0))
            outputs_partial = model(cropped_imgs[start_idx:end_idx], cls_name * (end_idx - start_idx),
                                    prompt_attn_mask.repeat((end_idx - start_idx), 1, 1, 1))[0]
            outputs.append(outputs_partial)
        results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, im.size(2), im.size(3),
                                         patch_size=384, stride=384).detach().cpu().squeeze(0).squeeze(0) / 60

        error.append(abs(gt - results.sum().item()))

    if idx == 1 or idx % 10 == 0 or idx == total_images:
        elapsed = time.time() - start_time
        avg_per_image = elapsed / idx
        remaining = avg_per_image * (total_images - idx)
        print(f"[{idx}/{total_images}] elapsed: {elapsed/60:.1f} min | "
              f"eta: {remaining/60:.1f} min | last_abs_err: {error[-1]:.3f}", flush=True)

mae = np.array(error).mean()
mse = np.sqrt(np.mean(np.square(error)))
print('MAE:', mae, 'MSE:', mse, flush=True)

