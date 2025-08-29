import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import image_utils


class SubCellDataset(Dataset):
    """PyTorch Dataset for SubCell image processing"""

    def __init__(self, path_list_file, model_channels="rybg", minmax_norm=False):
        """
        Args:
            path_list_file (str): Path to the CSV file containing image paths
            model_channels (str): Channel configuration (rybg, rbg, ybg, bg)
            minmax_norm (bool): Apply min-max normalization to images
        """
        self.model_channels = model_channels
        self.minmax_norm = minmax_norm
        self.data_list = []
        
        # Define channel mapping
        self.channel_mapping = {
            'r': 'r_image',
            'y': 'y_image', 
            'b': 'b_image',
            'g': 'g_image'
        }

        # Read CSV
        df = pd.read_csv(path_list_file)

        # Remove the '#' from column names if present
        df.columns = df.columns.str.lstrip('#')
        self.data_list = df.to_dict('records')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """Load and preprocess a single image set"""
        item = self.data_list[idx]
        
        # Load images based on model channels configuration
        cell_data = []
        
        # Only process channels specified in model_channels
        for channel_name in self.model_channels:
            channel_key = self.channel_mapping[channel_name]
            # load the channel image
            img = image_utils.read_grayscale_image(item[channel_key], minmax_norm=self.minmax_norm)
            cell_data.append(img)
        
        # Stack images along channel dimension
        cell_data = np.stack(cell_data, axis=0)  # Shape: (channels, height, width)

        return {
            "images": cell_data.astype(np.float32),
            "output_folder": item["output_folder"],
            "output_prefix": item["output_prefix"],
            "original_item": item,
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    # Stack numpy arrays first, then convert to tensor
    images_np = np.stack([item["images"] for item in batch])
    images = torch.from_numpy(images_np)
    
    return {
        "images": images,
        "output_folders": [item["output_folder"] for item in batch],
        "output_prefixes": [item["output_prefix"] for item in batch],
        "original_items": [item["original_item"] for item in batch],
    }

