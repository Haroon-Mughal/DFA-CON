from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import List, Dict
import os

class ContrastiveTrainDataset(Dataset):
    """
    Dataset for contrastive training.
    Each sample returns an anchor and its list of positives (forged versions).
    """
    def __init__(self, anchor_to_positives: Dict[str, List[str]], root_dir: str, transform=None):
        """
        Args:
            anchor_to_positives: dict mapping anchor image paths to list of positive image paths
            root_dir: base directory where image paths are relative to
            transform: torchvision transforms to apply
        """
        self.anchor_to_positives = anchor_to_positives
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.anchor_list = list(anchor_to_positives.keys())

    def __len__(self):
        return len(self.anchor_list)

    def __getitem__(self, idx):
        anchor_path = self.anchor_list[idx]
        pos_paths = self.anchor_to_positives[anchor_path]

        # Load anchor image
        anchor_img = Image.open(os.path.join(self.root_dir, anchor_path)).convert("RGB")
        anchor_img = self.transform(anchor_img)

        # Load all positive images
        pos_imgs = []
        for p in pos_paths:
            img = Image.open(os.path.join(self.root_dir, p)).convert("RGB")
            pos_imgs.append(self.transform(img))

        return {
            "anchor": anchor_img,
            "positives": pos_imgs,
            "group_id": idx  # Each anchor group has a unique ID
        }



class InferenceDataset(Dataset):
    """
    Dataset for loading and transforming images for inference time.
    Input: list of image paths
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
       path = self.image_paths[idx]
       img = Image.open(path).convert("RGB")
    
       if self.transform:
           img = self.transform(img)
       else:
           # Manually convert to tensor if no transform provided
           img = transforms.ToTensor()(img)
    
       return img, path


