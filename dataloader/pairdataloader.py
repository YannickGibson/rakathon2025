import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
from skimage.transform import resize

from dataloader import RTStructSliceDataset

DATASET_PATH = "dataloader/data/full/SAMPLE_001"


class PairDataloader(DataLoader):
    """
    Dataset for loading GTV, CTV, and PTV contours from a single patient's RT Structure Set.
    Converts contours to 128x128 bitmap images for each slice.
    """
    """
        def __init__(self, rtstruct_path, img_size=(128, 128)):

            self.dataset = RTStructSliceDataset(rtstruct_path, img_size)

    """
    def __init__(self, rtstruct_path, img_size=(128, 128), batch_size=1, shuffle=False, **kwargs):
        self.dataset = RTStructSliceDataset(rtstruct_path, img_size)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def _load_ct_image(self, ui):
        """Load and scale CT image based on UI"""
        self.dataset.load_ct_image(ui)

    def __len__(self):
        return len(self.slices / 2)

    def __getitem__(self, idx):

        item1 = self.dataset.__getitem__(idx * 2)
        item2 = self.dataset.__getitem__(idx * 2 + 1)

        if item1["review_date"] < item2["review_date"]:

            switch = item1
            item1 = item2
            item2 = switch

        return {
            "item1": item1,
            "item2": item2,
        }
    
    def get_all_slices_info(self):
        return self.dataset.get_all_slices_info()


# Example usage
if __name__ == "__main__":
    # Path to the specific RTSTRUCT file
    rtstruct_path = "dataloader/data/full/SAMPLE_001/RS.1.2.246.352.221.46272062591570509005209218152822185346.dcm"

    # Create dataset
    dataset = RTStructSliceDataset(rtstruct_path)

    dataset[0]
    # Print slice information
    
    slice_info = dataset.get_all_slices_info()
    
    for info in slice_info:
        print(
            f"Slice {info['index']}: Z={info['z_position']}, "
            f"GTV={info['num_GTV_contours']}, "
            f"CTV={info['num_CTV_contours']}, "
            f"PTV={info['num_PTV_contours']}"
        )

    # Create DataLoader
    dataloader = PairDataloader(dataset, batch_size=4, shuffle=False)

    # Example of iterating through the dataloader
    for batch in dataloader:
        masks = batch["masks"]
        z_positions = batch["z_position"]
        print(f"Batch shape: {masks.shape}, Z positions: {z_positions}")

    # Visualize a few slices
    for i in range(len(dataset)):
        fig = dataset.visualize_item(i + 35)
        plt.show()  # This will display the figure
        plt.close(fig)
