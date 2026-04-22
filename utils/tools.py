import torch
import torch.nn.functional as F

def extract_patches(img, patch_size=512, stride=512):
    _, _, h, w = img.size()
    if patch_size <= 0 or stride <= 0:
        raise ValueError(f"patch_size and stride must be positive, got patch_size={patch_size}, stride={stride}")

    patch_h = min(patch_size, h)
    patch_w = min(patch_size, w)
    num_h = max((h - patch_h + stride - 1) // stride + 1, 1)
    num_w = max((w - patch_w + stride - 1) // stride + 1, 1)
    patches = []
    for i in range(num_h):
        for j in range(num_w):
            y_start = min(i * stride, h - patch_h)
            x_start = min(j * stride, w - patch_w)
            patch = img[:, :, y_start:y_start + patch_h, x_start:x_start + patch_w]
            patches.append(patch)
    patches = torch.cat(patches, dim=0)
    return patches, num_h, num_w


def reassemble_patches(patches, num_h, num_w, h, w, patch_size=512, stride=256):
    result = torch.zeros(1, patches.size(1), h, w).to(patches.device)
    norm_map = torch.zeros(1, 1, h, w).to(patches.device)
    patches = F.interpolate(patches, scale_factor=8, mode='bilinear') / 64

    patch_h = min(patch_size, h)
    patch_w = min(patch_size, w)

    patch_idx = 0
    for i in range(num_h):
        for j in range(num_w):
            y_start = min(i * stride, h - patch_h)
            x_start = min(j * stride, w - patch_w)
            result[:, :, y_start:y_start + patch_h, x_start:x_start + patch_w] += patches[patch_idx]
            norm_map[:, :, y_start:y_start + patch_h, x_start:x_start + patch_w] += 1
            patch_idx += 1

    # Avoid 0/0 for uncovered regions when stride/shape combinations leave gaps.
    result = result / torch.clamp(norm_map, min=1.0)
    return result
