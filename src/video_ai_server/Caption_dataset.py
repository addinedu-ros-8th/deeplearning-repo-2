import torch
import os
import json
from PIL import Image

class Caption_dataset(torch.utils.data.Dataset):
    def __init__(self,img_folder, caption_file, transforms=None):
        super().__init__()
        self.img_folder = img_folder
        self.caption_file = caption_file
        self.transforms = transforms

        with open(caption_file, 'r') as f:
            self.captions = json.load(f)

        self.img_files = list(self.captions.keys())
        self.labels = ['normal', 'fight', 'fainting', 'smoking']

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_name)

        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        
        caption = self.captions[img_name]
        label = self.labels.index(caption)

        return img, torch.tensor(label, dtype=torch.long)