# Generated from: lungs_dataloader.ipynb
# Converted at: 2026-01-21T08:21:46.573Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



class LungTBDataset2D(torch.utils.data.Dataset):
    """
    2D Lung TB Chest X-ray Dataset

    Folder structure:
    root_dir/
        Normal/          -> label 0 (healthy)
        Tuberculosis/    -> label 1 (anomaly)
    """

    def __init__(self, image_path, class_label, transform=None):
        """
        Args:
            image_path (str): root dataset directory
            class_label (int): 0 = Normal, 1 = Tuberculosis
            transform (callable, optional): image transforms
        """
        self.image_path = image_path
        self.class_label = class_label
        self.transform = transform
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # ✅ REQUIRED for ICAM
        image = Image.open(img_path).convert("L")
        image = image.resize((128, 128))          # 🔥 THIS WAS MISSING
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)
        label = torch.tensor([label], dtype=torch.float32)


        # ICAM expects mask always
        mask = torch.zeros(1)

        return image, label, mask
