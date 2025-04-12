import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
from skimage.transform import resize
import pprint
from dataloader import RTStructSliceDataset

DATASET_PATH = "dataloader/data/full/SAMPLE_001"


class PairDataset(RTStructSliceDataset):
    """
    Dataset for loading GTV, CTV, and PTV contours from a single patient's RT Structure Set.
    Converts contours to 128x128 bitmap images for each slice.
    """

    def __init__(self, rtstruct_path1, rtstruct_path2, img_size=(128, 128)):

        self.dataset1 = RTStructSliceDataset(rtstruct_path1, img_size)
        self.dataset2 = RTStructSliceDataset(rtstruct_path2, img_size)
        self.lowest_z_position = 0
        self.max_z_position = 0
        self.sort_map = self._sort_slices()

    def _sort_slices(self):
        """Sort slices by z_position"""

      #   pp = pprint.PrettyPrinter(indent=4)

      #   self.print_slices_info()

        map = {}
        map[self.dataset1[0]["review_date"]] = {}
        map[self.dataset2[0]["review_date"]] = {}

        for i in range(0, len(self.dataset1)):
            item = self.dataset1[i]
            map[item["review_date"]][item["z_position"]] = item["index"]

        for i in range(0, len(self.dataset2)):
            item = self.dataset2[i]
            map[item["review_date"]][item["z_position"]] = item["index"]

        # Find the lowest z_position for every review_date
        lowest_z_positions = {}
        max_z_positions = {}
        for review_date, slices in map.items():
        #     pp.pprint("review_date")
          #   pp.pprint(review_date)
       #      pp.pprint("slices")
       #      pp.pprint(slices)

            lowest_z_positions[review_date] = min(slices.keys())
            max_z_positions[review_date] = max(slices.keys())

      #   print("Lowest z_positions for each review_date: ", lowest_z_positions)
     #    print("Lowest z_positions for each review_date: ", max_z_positions)

        self.lowest_z_position = min(lowest_z_positions.values())
        self.max_z_position = max(max_z_positions.values())

        self.lowest_z_position = max(lowest_z_positions.values())
        self.max_z_position = min(max_z_positions.values())

      #   print("map: ", map)

      #   pp.pprint(map)
      #   print("self.dataset.get_all_slices_info...")

        return map

    def print_slices_info(self):
        """Prints the information about all slices."""
        #  slices_info = self.dataset.get_all_slices_info()
        """
        for slice_info in slices_info:
            print(f"Slice Index: {slice_info['index']}")
            print(f"  Z Position: {slice_info['z_position']}")
            print(f"  GTV Contours: {slice_info['num_GTV_contours']}")
            print(f"  CTV Contours: {slice_info['num_CTV_contours']}")
            print(f"  PTV Contours: {slice_info['num_PTV_contours']}")
            print("-" * 20) 

                """
        """
        for i in range(0, len(self.dataset)):
            item = self.dataset[i]
            print(f"Slice Index: {item['index']}")
            print(f"  Z Position: {item['z_position']}")
            print(f" review_date: {item['review_date']}")

"""

    def _load_ct_image(self, ui):
        """Load and scale CT image based on UI"""
        self.dataset.load_ct_image(ui)

    def __len__(self):
        return int(abs(self.lowest_z_position - self.max_z_position) // 3) + 1

    def __getitem__(self, idx):

        z_position = int(self.lowest_z_position + idx * 3)

      #  print("z_position  " + str(z_position))
      #   print(self.sort_map)

        item1_position = self.sort_map[self.dataset1[0]["review_date"]].get(
            z_position, "None"
        )
        item2_position = self.sort_map[self.dataset2[0]["review_date"]].get(
            z_position, "None"
        )

       #  print("item1_position  " + str(item1_position))
      #   print("item2_position  " + str(item2_position))

        item1 = self.dataset1[item1_position]
        item2 = self.dataset2[item2_position]

        return {
            "z_position": z_position,
            "item1": item1,
            "item2": item2,
        }

        """

        if item1["review_date"] < item2["review_date"]:

            switch = item1
            item1 = item2
            item2 = switch

        return {
            "item1": item1,
            "item2": item2,
        }
    
    """

    def get_all_slices_info(self):
        return self.dataset.get_all_slices_info()

    def visualize_item(self, idx):

        fig1 = self.dataset.visualize_item(idx)
        fig2 = self.dataset.visualize_item(idx + 1)
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

    # Create dataset
    dataset = PairDataset(rtstruct_path1, rtstruct_path2)

    dataset[0]
    dataset[1]

    print(" dataset[0]")
    print(dataset[0])
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
        print("batch")
        print(batch["item1"])
        print(batch)
        masks1 = batch["item1"]["masks"]
        masks2 = batch["item2"]["masks"]
        z_positions1 = batch["item1"]["z_position"]
        z_positions2 = batch["item2"]["z_position"]
        print(f"1 Batch shape: {masks1.shape}, Z positions: {z_positions1}")
        print(f"2 Batch shape: {masks2.shape}, Z positions: {z_positions2}")
    """   

"""
"""
    # Visualize a few slices
    for i in range(min((len(dataset.dataset1)), 1)):
        fig1, fig2 = dataset.dataset1.visualize_item(i)
        plt.show()  # This will display the figure
        plt.close(fig1)
        plt.close(fig2)
"""
