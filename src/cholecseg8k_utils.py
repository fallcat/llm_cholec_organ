# -*- coding: utf-8 -*-
"""
Utilities for CholecSeg8k:
- Visualization: display_image
- Mask -> label IDs: color_mask_to_labels
- Example to torch tensors: example_to_tensors
- Presence QA targets: presence_qas_from_example, labels_to_presence_vector
- VLM prompting (Yes/No): build_system_prompt, build_user_prompt, to_yes_no,
  ask_vlm_yes_no, vlm_presence_pipeline
"""

from typing import Callable, Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import re
from torchvision.transforms import functional as tvtf


# =========================
# Constants & Class Mappings
# =========================

ID2LABEL: Dict[int, str] = {
    0: "Black Background",
    1: "Abdominal Wall",
    2: "Liver",
    3: "Gastrointestinal Tract",
    4: "Fat",
    5: "Grasper",
    6: "Connective Tissue",
    7: "Blood",
    8: "Cystic Duct",
    9: "L-hook Electrocautery",
    10: "Gallbladder",
    11: "Hepatic Vein",
    12: "Liver Ligament",
}
LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}
LABEL_IDS: List[int] = [k for k in sorted(ID2LABEL) if k != 0]  # 1..12

# Color → class-id mapping for color_mask
COLOR_CLASS_MAPPING: Dict[Tuple[int, int, int], int] = {
    (127, 127, 127): 0,
    (210, 140, 140): 1,
    (255, 114, 114): 2,
    (231, 70, 156): 3,
    (186, 183, 75): 4,
    (170, 255, 0): 5,
    (255, 85, 0): 6,
    (255, 0, 0): 7,
    (255, 255, 0): 8,
    (169, 255, 184): 9,
    (255, 160, 165): 10,
    (0, 50, 128): 11,
    (111, 74, 0): 12,
}


# ================
# Visualization
# ================

def display_image(dataset, image_index: int) -> None:
    """Display the RGB image and its three masks for a given index."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for ax in axs.flat:
        ax.axis('off')

    axs[0, 0].imshow(dataset['train'][image_index]['image'])
    axs[0, 1].imshow(dataset['train'][image_index]['color_mask'])
    axs[1, 0].imshow(dataset['train'][image_index]['watershed_mask'])
    axs[1, 1].imshow(dataset['train'][image_index]['annotation_mask'])

    plt.subplots_adjust(wspace=0.01, hspace=-0.6)
    plt.show()


# ==========================
# Masks → Labels / Tensors
# ==========================

def color_mask_to_labels(pil_img: Image.Image,
                         mapping: Dict[Tuple[int, int, int], int] = COLOR_CLASS_MAPPING,
                         ignore_value: int = -1) -> np.ndarray:
    """
    Convert a color mask (PIL) to an HxW label array using `mapping`.
    Unmapped colors (and fully transparent pixels) -> ignore_value.
    """
    alpha = None
    if pil_img.mode == 'RGBA':
        arr = np.array(pil_img)            # H x W x 4
        alpha = arr[..., 3]
        arr = arr[..., :3]
    elif pil_img.mode == 'P':
        arr = np.array(pil_img.convert('RGB'))
    elif pil_img.mode != 'RGB':
        arr = np.array(pil_img.convert('RGB'))
    else:
        arr = np.array(pil_img)

    h, w = arr.shape[:2]
    labels = np.full((h, w), ignore_value, dtype=np.int16)

    # Assign each mapped color
    for (r, g, b), cls in mapping.items():
        m = (arr[..., 0] == r) & (arr[..., 1] == g) & (arr[..., 2] == b)
        labels[m] = cls

    if alpha is not None:
        labels[alpha == 0] = ignore_value

    return labels


def example_to_tensors(example,
                       resize_to: Tuple[int, int] = None,
                       ignore_value: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert one HF dataset example into:
        image_t: torch.float32 [3,H,W] in [0,1]
        labels_t: torch.int64  [H,W] with ignore_value for unmapped
    """
    img = example['image']        # PIL Image
    mask = example['color_mask']  # PIL Image (color-coded)

    if resize_to is not None:
        img = img.resize(resize_to, resample=Image.BILINEAR)
        mask = mask.resize(resize_to, resample=Image.NEAREST)

    # Image -> torch [3,H,W], float32 in [0,1]
    image_t = tvtf.to_tensor(img).contiguous()

    # Color mask -> labels -> torch [H,W], int64
    labels_np = color_mask_to_labels(mask, mapping=COLOR_CLASS_MAPPING,
                                     ignore_value=ignore_value)
    labels_t = torch.from_numpy(labels_np.astype(np.int64)).contiguous()

    # Safety: spatial dims must match
    _, H, W = image_t.shape
    if labels_t.shape != (H, W):
        raise ValueError(f"Shape mismatch: image [{H},{W}] vs labels {tuple(labels_t.shape)}")

    return image_t, labels_t


# =================================
# Presence targets (Q/A & vectors)
# =================================

def presence_qas_from_example(img_t: torch.Tensor,
                              lab_t: torch.Tensor,
                              ignore_index: int = -1,
                              min_pixels: int = 1):
    """
    Build binary presence questions/answers from a label map.

    Returns:
        qas: list of dicts [{label_id, label, question, answer, count}, ...] (len=12)
        y:  torch.LongTensor [12] of 0/1 in order of IDs 1..12 (excludes background=0)
    """
    if isinstance(lab_t, np.ndarray):
        lab_t = torch.from_numpy(lab_t)
    lab_t = lab_t.to(torch.long)

    valid = lab_t != ignore_index
    flat = lab_t[valid].view(-1)

    num_classes = max(ID2LABEL.keys()) + 1  # 13
    counts = torch.zeros(num_classes, dtype=torch.long)
    if flat.numel() > 0:
        counts = torch.bincount(flat, minlength=num_classes)

    ids = list(range(1, num_classes))
    y = (counts[ids] >= min_pixels).to(torch.long)  # [12]

    qas = []
    for idx, cls_id in enumerate(ids):
        label = ID2LABEL[cls_id]
        qas.append({
            "label_id": cls_id,
            "label": label,
            "question": f"Is {label} visible in the image?",
            "answer": int(y[idx].item()),
            "count": int(counts[cls_id].item()),
        })

    return qas, y


def labels_to_presence_vector(lab_t: torch.Tensor,
                              ignore_index: int = -1,
                              min_pixels: int = 1) -> torch.LongTensor:
    """Return y [12] with 0/1 presence for class IDs 1..12 (excludes 0)."""
    if isinstance(lab_t, np.ndarray):
        lab_t = torch.from_numpy(lab_t)
    lab_t = lab_t.to(torch.long)

    valid = lab_t != ignore_index
    flat = lab_t[valid].view(-1)

    num_classes = max(ID2LABEL.keys()) + 1  # 13
    counts = torch.zeros(num_classes, dtype=torch.long)
    if flat.numel() > 0:
        counts = torch.bincount(flat, minlength=num_classes)

    y = (counts[LABEL_IDS] >= min_pixels).to(torch.long)  # [12]
    return y


# ==========================
# VLM Prompting (Yes / No)
# ==========================

def build_system_prompt() -> str:
    return (
        "You are a surgical vision validator. You will be shown one laparoscopic image.\n"
        "Answer STRICTLY with a single word: Yes or No.\n"
        "Rules:\n"
        "- 'Yes' only if ANY visible part of the named structure is present in the image.\n"
        "- If uncertain/occluded/blurred, answer 'No'.\n"
        "- Do not include punctuation, explanation, JSON, or extra words.\n"
    )


def build_user_prompt(organ_name: str) -> str:
    return f"Question: Is {organ_name} visible in the image?\nAnswer:"


YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def to_yes_no(raw_text: str) -> str:
    """Extract a single 'Yes' or 'No' from model output (fallback: 'No')."""
    if not isinstance(raw_text, str):
        return "No"
    m = YESNO_RE.search(raw_text.strip())
    return "Yes" if (m and m.group(1).lower() == "yes") else "No"


def ask_vlm_yes_no(model: Callable[..., str],
                   img_t: torch.Tensor,
                   organ_name: str,
                   *,
                   reprompt_once: bool = True) -> str:
    """
    Call your VLM once per organ and return 'Yes' or 'No'.
    The model must support: model(prompt_or_tuple, system_prompt=...).
    """
    pil_img = tvtf.to_pil_image(img_t.clamp(0, 1))
    system = build_system_prompt()
    user = build_user_prompt(organ_name)

    out = model((pil_img, user), system_prompt=system)
    ans = to_yes_no(out)

    if reprompt_once and ans not in ("Yes", "No"):
        stricter = system + (
            "\nAnswer with exactly one token: Yes or No. "
            "Do not include any other characters."
        )
        out2 = model((pil_img, user), system_prompt=stricter)
        ans = to_yes_no(out2)

    return ans if ans in ("Yes", "No") else "No"


def vlm_presence_pipeline(model: Callable[..., str],
                          img_t: torch.Tensor,
                          lab_t: torch.Tensor,
                          *,
                          ignore_index: int = -1,
                          min_pixels: int = 1) -> Tuple[List[Dict], torch.LongTensor, torch.LongTensor]:
    """
    Run 12 Yes/No queries (IDs 1..12).
    Returns:
      qa_rows: list of dicts with {'label_id','label','question','prediction','gt'}
      y_pred: torch.LongTensor [12] (0/1)
      y_true: torch.LongTensor [12] (0/1)
    """
    y_true = labels_to_presence_vector(lab_t, ignore_index, min_pixels)  # [12]
    y_pred_list: List[int] = []
    qa_rows: List[Dict] = []

    for idx, cls_id in enumerate(LABEL_IDS):
        label = ID2LABEL[cls_id]
        pred = ask_vlm_yes_no(model, img_t, label)  # 'Yes'/'No'
        pred01 = 1 if pred == "Yes" else 0
        y_pred_list.append(pred01)

        qa_rows.append({
            "label_id": cls_id,
            "label": label,
            "question": f"Is {label} visible in the image?",
            "prediction": pred,
            "gt": int(y_true[idx].item()),
        })

    y_pred = torch.tensor(y_pred_list, dtype=torch.long)  # [12]
    return qa_rows, y_pred, y_true


# ================
# Minimal demo
# ================

if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("minwoosun/CholecSeg8k")
    img_t, lab_t = example_to_tensors(ds['train'][800])

    print(img_t.shape, img_t.dtype, img_t.min().item(), img_t.max().item())
    print(lab_t.shape, lab_t.dtype, torch.unique(lab_t)[:10])

    # Example Q/A targets from labels
    qas, y = presence_qas_from_example(img_t, lab_t)
    print(qas[0])
    print(y.shape, y)
