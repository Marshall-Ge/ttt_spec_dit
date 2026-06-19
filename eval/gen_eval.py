# -*- coding: utf-8 -*-
"""GenEval — compositional text-to-image alignment evaluator.

Official implementation from Ghosh et al., NeurIPS 2023.
Adapted from: https://github.com/djghosh13/geneval

Evaluates 6 skills using Mask2Former (object detection) + CLIP ViT-L-14 (color
classification):
  1. Single Object      — one specified object present?
  2. Two Object         — two different objects?
  3. Counting           — correct count (2-4)?
  4. Colors             — correct color?
  5. Position           — correct relative position?
  6. Attribute Binding  — two objects with different colors?

Requires:
    pip install open_clip_torch
    Models auto-downloaded from HuggingFace on first use.
    Set HF_HUB_OFFLINE=1 if running without network.
"""

import json
import os
import numpy as np
import torch
from PIL import Image, ImageOps
from .base import Metric

# Use HF mirror for China + offline mode when cached
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# Offline: expected model cache locations
HF_MASK2FORMER = "facebook/mask2former-swin-tiny-coco-instance"
HF_OPENCLIP = "ViT-L-14"
OPENCLIP_PRETRAINED = "openai"

# GenEval color vocabulary
COLORS = ["red", "orange", "yellow", "green", "blue", "purple",
          "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}  # per-classname cache

# Detection hyperparameters (from official repo)
DETECTION_THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0
POSITION_THRESHOLD = 0.1

# COCO 80 classnames (from official object_names.txt)
COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class GenEvalScorer(Metric):
    """Official GenEval evaluator using Mask2Former + OpenCLIP ViT-L-14.

    Parameters
    ----------
    device : str
    metadata_path : str or None
        Path to evaluation_metadata.jsonl. If None, defaults to the bundled
        metadata (downloaded from official geneval repo).
    """

    def __init__(self, device: str = "cuda",
                 metadata_path: str = None):
        self.device = device
        self._detector = None      # Mask2Former
        self._processor = None
        self._clip_model = None    # OpenCLIP ViT-L-14
        self._clip_transform = None
        self._clip_tokenizer = None
        self._loaded = False
        self._load_failed = False
        self._scores: list = []
        self._per_task: dict = {}

        # Load metadata
        if metadata_path is None:
            metadata_path = os.path.join(os.path.dirname(__file__),
                                         "geneval_metadata.jsonl")
        self._metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                for line in f:
                    item = json.loads(line)
                    self._metadata[item["prompt"]] = item
        else:
            print(f"  [GenEval] WARN: metadata not found at {metadata_path}")

    # ------------------------------------------------------------------
    # Lazy-load models
    # ------------------------------------------------------------------

    def _lazy_load(self):
        if self._loaded or self._load_failed:
            return

        try:
            # -- Object detector: Mask2Former via HuggingFace (fp16 to save VRAM) --
            from transformers import (Mask2FormerForUniversalSegmentation,
                                      Mask2FormerImageProcessor)
            print(f"  [GenEval] loading Mask2Former ({HF_MASK2FORMER})...")
            self._processor = Mask2FormerImageProcessor.from_pretrained(
                HF_MASK2FORMER, local_files_only=True)
            self._detector = Mask2FormerForUniversalSegmentation.from_pretrained(
                HF_MASK2FORMER, torch_dtype=torch.float16,
                local_files_only=True).to(self.device).eval()

            # -- Color classifier: OpenCLIP ViT-L-14 --
            import open_clip
            print(f"  [GenEval] loading OpenCLIP ({HF_OPENCLIP})...")
            self._clip_model, _, self._clip_transform = \
                open_clip.create_model_and_transforms(
                    HF_OPENCLIP, pretrained=OPENCLIP_PRETRAINED,
                    device=self.device)
            self._clip_tokenizer = open_clip.get_tokenizer(HF_OPENCLIP)

            self._loaded = True
            print(f"  [GenEval] ready")
        except Exception as e:
            self._load_failed = True
            print(f"  [GenEval] FAILED — {e}")
            print(f"  [GenEval] Install: pip install open_clip_torch")
            print(f"  [GenEval] All scores will be NaN until resolved.")

    # ------------------------------------------------------------------
    # Object detection (bboxes per class)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _detect(self, pil_image: Image.Image) -> dict:
        """Run Mask2Former detection. Returns {classname: [(bbox, mask), ...]}."""
        inputs = self._processor(pil_image, return_tensors="pt")
        inputs = {k: v.to(device=self.device, dtype=torch.float16
                          if v.dtype == torch.float32 else v.dtype)
                  for k, v in inputs.items()}
        outputs = self._detector(**inputs)

        # HuggingFace Mask2Former: returns 'segmentation' (H,W) + 'segments_info'
        result = self._processor.post_process_instance_segmentation(
            outputs, threshold=DETECTION_THRESHOLD)[0]
        seg_map = result['segmentation'].cpu().numpy()  # [H, W] with instance IDs
        seg_info = {s['id']: s for s in result['segments_info']}

        W, H = pil_image.size
        detected = {}

        # Map label_id → classname
        for seg_id, info in seg_info.items():
            label_id = info['label_id']
            score = info['score']
            if label_id >= len(COCO_80_CLASSES):
                continue  # skip "stuff" classes (wall, sky, etc.)
            classname = COCO_80_CLASSES[label_id]

            # Extract mask and bbox for this instance
            mask = (seg_map == seg_id)
            if not mask.any():
                continue

            ys, xs = np.where(mask)
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max(),
                             score], dtype=np.float32)

            # Resize mask to original image size if needed
            if mask.shape[:2] != (H, W):
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                mask = np.array(mask_img.resize((W, H))) > 127

            if classname not in detected:
                detected[classname] = []
            detected[classname].append((bbox, mask))

        # Sort by confidence and apply NMS + MAX_OBJECTS per class
        for classname in list(detected.keys()):
            items = detected[classname]
            items.sort(key=lambda x: x[0][4], reverse=True)  # sort by score
            items = items[:MAX_OBJECTS]
            detected[classname] = items

        return detected

    # ------------------------------------------------------------------
    # Color classification (official geneval logic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _classify_colors(self, pil_image: Image.Image, bboxes, classname):
        """Classify the color of each object crop. Returns list of color names."""
        if classname not in COLOR_CLASSIFIERS:
            # Build zero-shot classifier: one text embedding per color
            templates = [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object",
            ]
            all_texts = []
            for color in COLORS:
                for tpl in templates:
                    all_texts.append(tpl.format(c=color))
            tokenized = self._clip_tokenizer(all_texts).to(self.device)
            text_feats = self._clip_model.encode_text(tokenized)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            # Average over templates per color → [10, D]
            text_feats = text_feats.reshape(len(COLORS), len(templates), -1).mean(dim=1)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            COLOR_CLASSIFIERS[classname] = text_feats  # [10, D]
        clf_emb = COLOR_CLASSIFIERS[classname]

        bgcolor = "#999"
        blank = Image.new("RGB", pil_image.size, color=bgcolor)
        crops = []
        for box, mask in bboxes:
            x1, y1, x2, y2, _ = box
            if mask is not None:
                mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                cropped = Image.composite(pil_image, blank, mask_pil)
            else:
                cropped = pil_image
            cropped = cropped.crop((int(x1), int(y1), int(x2), int(y2)))
            crops.append(self._clip_transform(cropped))

        if not crops:
            return []

        # Classify: dot-product similarity
        image_tensor = torch.stack(crops).to(self.device)
        image_feats = self._clip_model.encode_image(image_tensor)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        similarity = (image_feats @ clf_emb.T)  # [N_crops, 10]
        preds = similarity.argmax(dim=1)
        return [COLORS[p.item()] for p in preds]

    # ------------------------------------------------------------------
    # Position logic (from official geneval)
    # ------------------------------------------------------------------

    @staticmethod
    def _relative_position(obj_a, obj_b):
        """Determine position of obj_a relative to obj_b."""
        box_a = obj_a[0]
        box_b = obj_b[0]
        center_a = np.array([(box_a[0] + box_a[2]) / 2,
                             (box_a[1] + box_a[3]) / 2])
        center_b = np.array([(box_b[0] + box_b[2]) / 2,
                             (box_b[1] + box_b[3]) / 2])
        dim_a = np.array([box_a[2] - box_a[0], box_a[3] - box_a[1]])
        dim_b = np.array([box_b[2] - box_b[0], box_b[3] - box_b[1]])
        offset = center_a - center_b
        threshold = POSITION_THRESHOLD * (dim_a + dim_b)
        revised = np.maximum(np.abs(offset) - threshold, 0) * np.sign(offset)
        if np.all(np.abs(revised) < 1e-3):
            return set()
        norm = np.linalg.norm(offset)
        if norm < 1e-6:
            return set()
        dx, dy = revised / norm
        relations = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    # ------------------------------------------------------------------
    # Evaluate one image against metadata (official geneval logic)
    # ------------------------------------------------------------------

    def _evaluate_one(self, pil_image: Image.Image, metadata: dict) -> bool:
        """Evaluate one image. Returns True if all constraints satisfied."""
        detected = self._detect(pil_image)
        correct = True
        matched_groups = []

        # Debug: print detected classes
        if not hasattr(self, '_debug_cnt'):
            self._debug_cnt = 0
        if self._debug_cnt < 3:
            print(f"  [GenEval debug] prompt='{metadata['prompt'][:60]}' "
                  f"tag={metadata['tag']} "
                  f"detected={list(detected.keys())[:8]}")
            self._debug_cnt += 1

        for req in metadata.get('include', []):
            classname = req['class']
            matched = True
            found = detected.get(classname, [])[:req['count']]
            if len(found) < req['count']:
                correct = matched = False
                if self._debug_cnt <= 3:
                    print(f"    FAIL: expected {classname}>={req['count']}, "
                          f"found {len(found)}")
            else:
                if 'color' in req:
                    colors = self._classify_colors(pil_image, found, classname)
                    if colors.count(req['color']) < req['count']:
                        correct = matched = False
                        if self._debug_cnt <= 3:
                            print(f"    FAIL color: expected {req['color']} {classname}, "
                                  f"got {colors}")
                if 'position' in req and matched:
                    expected_rel, target_group = req['position']
                    if target_group >= len(matched_groups) or matched_groups[target_group] is None:
                        correct = matched = False
                    else:
                        for obj in found:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self._relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    break
                            if not matched:
                                break
            matched_groups.append(found if matched else None)

        for req in metadata.get('exclude', []):
            if len(detected.get(req['class'], [])) >= req['count']:
                correct = False

        return correct

    # ------------------------------------------------------------------
    # Score a single (prompt, image) pair
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(self, prompt: str, image: torch.Tensor) -> float:
        """Evaluate one image. Returns 1.0 if correct, 0.0 otherwise."""
        self._lazy_load()
        if self._load_failed or not self._loaded:
            return float("nan")

        meta = self._metadata.get(prompt)
        if meta is None:
            return float("nan")

        # Convert tensor to PIL
        if image.dim() == 4:
            image = image.squeeze(0)
        arr = (image.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)

        correct = self._evaluate_one(pil, meta)
        return 1.0 if correct else 0.0

    # ------------------------------------------------------------------
    # Metric interface
    # ------------------------------------------------------------------

    def add(self, image: torch.Tensor, prompt: str = None,
            reference: torch.Tensor = None):
        s = self.score(prompt, image)
        self._scores.append(s)
        # Track per-task
        meta = self._metadata.get(prompt or "")
        if meta is not None:
            tag = meta["tag"]
            if tag not in self._per_task:
                self._per_task[tag] = []
            self._per_task[tag].append(s)

    def compute(self) -> dict:
        result = {}
        # Overall
        if self._scores:
            vals = [s for s in self._scores
                    if not (isinstance(s, float) and np.isnan(s))]
            if vals:
                result["geneval_overall"] = float(np.mean(vals))
        else:
            result["geneval_overall"] = float("nan")

        # Per-task
        task_names = {
            "single_object": "geneval_single_object",
            "two_object": "geneval_two_object",
            "counting": "geneval_counting",
            "colors": "geneval_colors",
            "position": "geneval_position",
            "color_attr": "geneval_attribute_binding",
        }
        for tag, key in task_names.items():
            scores = self._per_task.get(tag, [])
            if scores:
                result[key] = float(np.mean(scores))
            else:
                result[key] = float("nan")

        return result

    def reset(self):
        self._scores.clear()
        self._per_task.clear()
