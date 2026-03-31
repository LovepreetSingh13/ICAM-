# Generated from: lungs_dataloader.ipynb
# Converted at: 2026-01-21T08:21:46.573Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class LungTBDataset2D(torch.utils.data.Dataset):
    """
    2D Lung TB Chest X-ray Dataset
    Folder structure:
    root_dir/
        Normal/          -> label 0 (healthy)
        Tuberculosis/    -> label 1 (anomaly)
    """
    def __init__(self, image_path, class_label, transform=None, augment=False):
        """
        Args:
            image_path (str): root dataset directory
            class_label (int): 0 = Normal, 1 = Tuberculosis
            transform (callable, optional): image transforms
            augment (bool): whether to apply augmentation (only for TB class)
        """
        self.image_path = image_path
        self.class_label = class_label
        self.transform = transform
        self.augment = augment
        self.samples = []

        if class_label == 0:
            data_dir = os.path.join(image_path, "Normal")
            label = 0
        else:
            data_dir = os.path.join(image_path, "Tuberculosis")
            label = 1

        for img in sorted(os.listdir(data_dir)):
            if img.lower().endswith(".png"):
                self.samples.append(
                    (os.path.join(data_dir, img), label)
                )

        # If augment=True (TB class only), generate augmented versions
        # to bring TB samples from ~700 up to ~3500 (matching Normal class)
        if self.augment and class_label == 1:
            original_samples = list(self.samples)
            target_size = 3500  # match Normal class size
            aug_needed = target_size - len(original_samples)
            print(f'[Augmentation] TB original: {len(original_samples)}, '
                  f'augmenting to: {target_size} (+{aug_needed} samples)')
            # Cycle through original samples and mark as augmented
            for i in range(aug_needed):
                src = original_samples[i % len(original_samples)]
                # Store as (path, label, is_augmented)
                self.samples.append((src[0], src[1], True))
            # Mark originals as not augmented
            self.samples = [
                (s[0], s[1], False) if len(s) == 2 else s
                for s in self.samples
            ]
        else:
            # No augmentation — mark all as not augmented
            self.samples = [(s[0], s[1], False) for s in self.samples]

    def _apply_augmentation(self, image):
        """
        Safe augmentations for chest X-rays.
        All operations preserve diagnostic features.
        """
        # 1. Random horizontal flip (chest X-rays can be mirrored)
        if random.random() > 0.5:
            image = TF.hflip(image)

        # 2. Random small rotation (±10 degrees max for medical images)
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle)

        # 3. Random brightness adjustment (simulates X-ray exposure variation)
        brightness_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_brightness(image, brightness_factor)

        # 4. Random contrast adjustment (simulates different X-ray densities)
        contrast_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_contrast(image, contrast_factor)

        # 5. Random zoom/crop (simulates different patient positioning)
        if random.random() > 0.5:
            # Random crop then resize back to 128x128
            crop_size = random.randint(100, 128)
            i = random.randint(0, 128 - crop_size)
            j = random.randint(0, 128 - crop_size)
            image = TF.crop(image, i, j, crop_size, crop_size)
            image = TF.resize(image, [128, 128])

        # 6. Add small Gaussian noise (simulates X-ray sensor noise)
        if random.random() > 0.5:
            noise = torch.randn_like(image) * 0.02  # very small noise
            image = torch.clamp(image + noise, 0.0, 1.0)

        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, is_augmented = self.samples[idx]

        # Load and preprocess image
        image = Image.open(img_path).convert("L")
        image = image.resize((128, 128))
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        # Apply augmentation only if this sample is marked as augmented
        if is_augmented:
            image = self._apply_augmentation(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([label], dtype=torch.float32)

        # ICAM expects mask always
        mask = torch.zeros(1)

        return image, label, mask
