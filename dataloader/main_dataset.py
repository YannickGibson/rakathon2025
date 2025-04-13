import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import pydicom
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm

from pair_dataset import PairDataset
from rs_dataset import RSDataset

class MainDataset(Dataset):
    """
    Dataset for loading GTV, CTV, and PTV contours from a single patient's RT Structure Set.
    Converts contours to img_size bitmap images for each slice.
    """
    def __init__(self, DATASET_PATH: str, img_size=(512, 512), limit_samples: int | None = None, limit_rs_pairs: int | None = None, mock=False, verbose=True):
        """
        Args:
            DATASET_PATH is the full dataset path with all patients.
        Task: 
            1. Load all RS prefixed datasets.
            2. Claim their 'review_date'.
            3. Create a dict where key is review_date and value is the dataset path
            4. Create N pair datasets, each using 2 datasets paths for initialization.
            5. Get item returns get_item from PairDataset
        """
        self.DATASET_PATH = DATASET_PATH
        self.img_size = img_size
        self.verbose = verbose

        # Load all sample paths (folders in DATASET_PATH prefixed with sample)
        self.SAMPLE_PATHs = []
        for sample in os.listdir(DATASET_PATH):
            if sample.startswith("SAMPLE_"):
                self.SAMPLE_PATHs.append(f"{DATASET_PATH}/{sample}")
            
        self.SAMPLE_PATHs.sort()

        if limit_samples is not None:
            self.SAMPLE_PATHs = self.SAMPLE_PATHs[:limit_samples]

        from_base_folder = os.path.basename(self.SAMPLE_PATHs[0])   
        to_base_folder = os.path.basename(self.SAMPLE_PATHs[-1])
        print(f"Loading {len(self.SAMPLE_PATHs)} samples from {from_base_folder} to {to_base_folder}")


        date_paths = []
        if mock:
            date_path_dict = {'20230616': f'{DATASET_PATH}/SAMPLE_001/RS.1.2.246.352.221.46272062591570509005209218152822185346.dcm', '20230719': f'{DATASET_PATH}/SAMPLE_001/RS.1.2.246.352.221.46648924540845111847267152667592345525.dcm', '20240306': f'{DATASET_PATH}/SAMPLE_001/RS.1.2.246.352.221.474069323621439861613904667800073459614.dcm'}
            date_paths.append(date_path_dict)
            sorted_keys = sorted(list(date_path_dict.keys()))

            self.pair_datasets = []
            self.pair_dataset_start = []
            self.total_count = 0

            if verbose:
                pbar = tqdm(range(len(sorted_keys)-1), desc=f"Loading+Filtering {len(sorted_keys)-1} Pair Datasets")
            else:
                pbar = range(len(sorted_keys)-1)
            for i in pbar:
                date1 = sorted_keys[i]
                date2 = sorted_keys[i+1]
                path1 = date_path_dict[date1]
                path2 = date_path_dict[date2]
                # Now we load pair dataset
                pair_dataset = PairDataset(path1, path2, img_size=self.img_size)
                if len(pair_dataset) > 0:
                    self.pair_datasets.append(pair_dataset)
                    self.pair_dataset_start.append(self.total_count)
                    self.total_count += len(pair_dataset)
        else:

            self.pair_datasets = []
            self.pair_dataset_start = []
            self.total_count = 0
            for sample_path in self.SAMPLE_PATHs:
                # Find all rs files
                rs_files = []
                for file in os.listdir(sample_path):
                    if file.startswith("RS."):
                        rs_files.append(f"{sample_path}/{file}")

                rs_files.sort()
                if limit_rs_pairs is not None:
                    rs_files = rs_files[:limit_rs_pairs]


                date_path_dict = {}
                
                sample_folder = sample_path.replace('\\', "/").split("/")[-1]
                if self.verbose:
                    pbar = tqdm(rs_files, desc=f"Loading RS files from '{sample_folder}'")
                else:
                    pbar = rs_files
                for rs_file in pbar:
                    dataset = RSDataset(rs_file, img_size=self.img_size, verbose=False)
                    if len(dataset) > 0:
                        date = dataset[0]['review_date']
                        date_path_dict[date] = rs_file

                date_paths.append(date_path_dict)
                
                sorted_keys = sorted(list(date_path_dict.keys()))

                if verbose:
                    pbar = tqdm(range(len(sorted_keys)-1), desc=f"Loading+Filtering {len(sorted_keys)-1} Pair Datasets")
                else:
                    pbar = range(len(sorted_keys)-1)
                for i in pbar:
                    date1 = sorted_keys[i]
                    date2 = sorted_keys[i+1]
                    path1 = date_path_dict[date1]
                    path2 = date_path_dict[date2]
                    # Now we load pair dataset
                    pair_dataset = PairDataset(path1, path2, img_size=self.img_size)
                    if len(pair_dataset) > 0:
                        print("appending pair dataset")
                        self.pair_datasets.append(pair_dataset)
                        self.pair_dataset_start.append(self.total_count)
                        self.total_count += len(pair_dataset)
                print("Sample done")

        if verbose:
            print(f"Loaded {len(self.pair_datasets)} Pair Datasets")


    def __len__(self):
        return self.total_count


    def __getitem__(self, idx):
        # get outer index is our pair dataset
        if idx >= self.total_count:
            raise IndexError(f"Index {idx} is out of range. Max index is {self.total_count - 1}")
        idx %= self.total_count
        for outer_index in range(len(self.pair_dataset_start)):
            if outer_index == len(self.pair_dataset_start) - 1 or idx < self.pair_dataset_start[outer_index + 1]:
                inner_index = idx - self.pair_dataset_start[outer_index]
                return self.pair_datasets[outer_index][inner_index]
        raise IndexError 

if __name__ == "__main__":

    LOAD_PICKLE = False

    if LOAD_PICKLE:
        import pickle
        print("Loading dataset via pickle")
        with open('_main_dataset.pkl', 'rb') as f:
            main_dataset = pickle.load(f)
    else:
        DATASET_PATH = "dataloader/data/full/"

        main_dataset = MainDataset(
            DATASET_PATH=DATASET_PATH,
            img_size=(1024, 1024),
            limit_samples=1, limit_rs_pairs=10,
            #limit_samples=3, limit_rs_pairs=40,
            mock=True, verbose=True)

        save_name = "_new_small_main_dataset"
        print("Saving dataset via pickle to", save_name)
        import pickle
        with open(f'{save_name}.pkl', 'wb') as f:
            pickle.dump(main_dataset, f)
    print(f"Amount of pairs: {len(main_dataset.pair_datasets)}")
    #print(main_dataset[0])

        # save as pickle

    print("Done.")