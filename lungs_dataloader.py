# Generated from: lungs_dataloader.ipynb
# Converted at: 2026-01-21T08:21:46.573Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

class LungTBDataset2D(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        normal_dir = os.path.join(root_dir, "Normal")
        tb_dir = os.path.join(root_dir, "Tuberculosis")

        for img in os.listdir(normal_dir):
            if img.lower().endswith(".png"):
                self.samples.append((os.path.join(normal_dir, img), 0))

        for img in os.listdir(tb_dir):
            if img.lower().endswith(".png"):
                self.samples.append((os.path.join(tb_dir, img), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).long()
        mask = torch.zeros(1)

        return image, label, mask