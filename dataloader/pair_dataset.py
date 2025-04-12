import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
from skimage.transform import resize
import pprint
from rs_dataset import RSDataset

DATASET_PATH = "dataloader/data/full/SAMPLE_001"


class PairDataset(Dataset):
    """
    Dataset for loading GTV, CTV, and PTV contours from a single patient's RT Structure Set.
    Converts contours to 128x128 bitmap images for each slice.
    """

    def __init__(self, rtstruct_path1, rtstruct_path2, img_size=(128, 128)):

        self.dataset1 = RSDataset(rtstruct_path1, img_size)
        self.dataset2 = RSDataset(rtstruct_path2, img_size)
        self.lowest_z_position = 0
        self.lowest_z_position_1 = 0
        self.lowest_z_position_2 = 0
        self.max_z_position = 0
        self.max_z_position_1 = 0
        self.max_z_position_2 = 0
        self.offset = 0
        self.review_date_1 = None
        self.review_date_2 = None
        self.is_problematic = False
        self.sort_map = self._sort_slices()

    def _sort_slices(self):
        """Sort slices by z_position"""

        map = {}
        map[self.dataset1[0]["review_date"]] = {}
        map[self.dataset2[0]["review_date"]] = {}

        for i in range(0, len(self.dataset1)):
            item = self.dataset1[i]
            map[item["review_date"]][item["z_position"]] = item["index"]

        for i in range(0, len(self.dataset2)):
            item = self.dataset2[i]
            map[item["review_date"]][item["z_position"]] = item["index"]

        lowest_z_positions = {}
        max_z_positions = {}
        for review_date, slices in map.items():

            lowest_z_positions[review_date] = min(slices.keys())
            max_z_positions[review_date] = max(slices.keys())

 

        self.lowest_z_position = max(lowest_z_positions.values())
        self.max_z_position = min(max_z_positions.values())

        self.review_date_1 = self.dataset1[0]["review_date"]
        self.review_date_2 = self.dataset2[0]["review_date"]

        self.lowest_z_position_1 = lowest_z_positions[self.review_date_1]
        self.lowest_z_position_2 = lowest_z_positions[self.review_date_2]

        self.max_z_position_1 = max_z_positions[self.review_date_1]
        self.max_z_position_2 = max_z_positions[self.review_date_2]

        if (
            self.lowest_z_position_1 > self.max_z_position_2
            or self.lowest_z_position_2 > self.max_z_position_1
        ):
            print("Problematic dataset XD")
            self.is_problematic = True
            return map

        self.offset = (
            self.dataset2[0]["z_position"] - self.dataset1[0]["z_position"]
        ) % 3
        if self.offset == 2:
            self.offset = -1

        is_dataset_1_minimum = self.lowest_z_position_1 == self.lowest_z_position

        if is_dataset_1_minimum:
            while abs(self.lowest_z_position_2 - self.lowest_z_position_1) > 1:
                self.lowest_z_position_2 = self.lowest_z_position_2 + 3
        else:
            while abs(self.lowest_z_position_1 - self.lowest_z_position_2) > 1:
                self.lowest_z_position_1 = self.lowest_z_position_1 + 3

        is_dataset_1_maximum = self.max_z_position_1 == self.max_z_position

        if is_dataset_1_maximum:
            while abs(self.max_z_position_2 - self.max_z_position_1) > 1:
                self.max_z_position_2 = self.max_z_position_2 - 3
        else:
            while abs(self.max_z_position_1 - self.max_z_position_2) > 1:
                self.max_z_position_1 = self.max_z_position_1 - 3

 

        return map

    def _load_ct_image(self, ui):
        """Load and scale CT image based on UI"""
        self.dataset.load_ct_image(ui)

    def __len__(self):

        if self.is_problematic:
            return 0

        return int(abs(self.lowest_z_position_1 - self.max_z_position_1) // 3) + 1

    def _get_indexes_of_datasets(self, idx):
        z_position_1 = int(self.lowest_z_position_1 + idx * 3)
        z_position_2 = int(self.lowest_z_position_2 + idx * 3)

        print(f"z_position_1: {z_position_1}, z_position_2: {z_position_2}")

        item1_position = self.sort_map[self.dataset1[0]["review_date"]].get(
            z_position_1, "None"
        )
        item2_position = self.sort_map[self.dataset2[0]["review_date"]].get(
            z_position_2, "None"
        )

        print(f"item1_position: {item1_position}, item2_position: {item2_position}")
        return item1_position, item2_position

    def __getitem__(self, idx):

        if self.is_problematic:
            raise IndexError("Dataset is problematic")

        z_position = int(self.lowest_z_position + idx * 3)

        item1_position, item2_position = self._get_indexes_of_datasets(idx)

        item1 = self.dataset1[item1_position]
        item2 = self.dataset2[item2_position]

        return {
            "z_position": z_position,
            "item1": item1,
            "item2": item2,
        }

    def get_all_slices_info(self):
        return self.dataset.get_all_slices_info()

    def visualize_item(self, idx):

        item1_position, item2_position = self._get_indexes_of_datasets(idx)

        fig1 = self.dataset1.visualize_item(item1_position)
        fig2 = self.dataset2.visualize_item(item2_position)
        return fig1, fig2


# Example usage
if __name__ == "__main__":
    # Path to the specific RTSTRUCT file

    rtstruct_path1 = (
        DATASET_PATH + "/RS.1.2.246.352.221.53086809173815688567595866456863246500.dcm"
    )
    rtstruct_path2 = (
        DATASET_PATH + "/RS.1.2.246.352.221.46272062591570509005209218152822185346.dcm"
    )

    rtstruct_path2_empty = (
        DATASET_PATH + "/RS.1.2.246.352.221.575810977437501802411405694909401015700.dcm"
    )
    rtstruct_path2_empty = (
        DATASET_PATH + "/RS.1.2.246.352.221.572473946963029837111301088203390015649.dcm"
    )
    rtstruct_path2_empty = (
        DATASET_PATH + "/RS.1.2.246.352.221.532615965879045657310555204002629787284.dcm"
    )
    rtstruct_path2_empty = (
        DATASET_PATH + "/RS.1.2.246.352.221.53650830509301879903201092234093239207.dcm"
    )
    rtstruct_path2 = (
        DATASET_PATH + "/RS.1.2.246.352.221.46648924540845111847267152667592345525.dcm"
    )

    rtstruct_path = "dataloader/data/full//SAMPLE_001/RS.1.2.246.352.221.46648924540845111847267152667592345525.dcm"

    rtstruct_path2 = "dataloader/data/full//SAMPLE_001/RS.1.2.246.352.221.474069323621439861613904667800073459614.dcm"

    # Create dataset
    dataset = RSDataset(rtstruct_path1)

    dataset = RSDataset(rtstruct_path2)

    dataset = PairDataset(rtstruct_path, rtstruct_path2)
    dataset = PairDataset(rtstruct_path2, rtstruct_path)

    print(dataset[0]["z_position"])
    print(dataset[56]["z_position"])

    # Print slice information

    slice_info = dataset.dataset1.get_all_slices_info()

    for info in slice_info:
        print(
            f"Slice {info['index']}: Z={info['z_position']}, "
            f"GTV={info['num_GTV_contours']}, "
            f"CTV={info['num_CTV_contours']}, "
            f"PTV={info['num_PTV_contours']}"
        )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Example of iterating through the dataloader
    for batch in dataloader:
        masks1 = batch["item1"]["masks"]
        masks2 = batch["item2"]["masks"]
        z_positions1 = batch["item1"]["z_position"]
        z_positions2 = batch["item2"]["z_position"]
        print(f"1 Batch shape: {masks1.shape}, Z positions: {z_positions1}")
        print(f"2 Batch shape: {masks2.shape}, Z positions: {z_positions2}")

    # Visualize a few slices
    for i in range(min((len(dataset)), 4)):
        fig1, fig2 = dataset.visualize_item(i)
        plt.show()  # This will display the figure
        plt.close(fig1)
        plt.close(fig2)
