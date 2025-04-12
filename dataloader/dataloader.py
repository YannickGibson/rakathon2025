import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
from skimage.transform import resize

DATASET_PATH = "dataloader/data/full/SAMPLE_001"

class RTStructSliceDataset(Dataset):
    """
    Dataset for loading GTV, CTV, and PTV contours from a single patient's RT Structure Set.
    Converts contours to 128x128 bitmap images for each slice.
    """
    def __init__(self, rtstruct_path, img_size=(128, 128), verbose=False):
        self.rtstruct_path = rtstruct_path
        self.img_size = img_size
        self.slices = []
        self.verbose = verbose
        
        # Load the RTSTRUCT file
        print(f"Loading RT Structure file: {rtstruct_path}")
        self.rtstruct = pydicom.dcmread(rtstruct_path)
        
        # Extract contours
        self.contours, self.instance_iuds = self._extract_contours()
        
        # Focus on GTV, CTV, and PTV contours
        self.target_contours, self.target_instance_uids = self._filter_target_contours()
        
        # Group contours by slice (z-position)
        self._group_contours_by_slice()

        self.slice_uis = []
        
        print(f"Found {len(self.slices)} slices with contours")
        

    def _extract_instance_uids(self):
        uis = []
        """
        ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[0].ReferencedSOPInstanceUID)
        """
        self.rtstruct
    def _extract_contours(self):
        """Extract all contours from the RTSTRUCT file"""
        contours = {}
        instance_iuds = {}
        
        # Get ROI names and numbers
        roi_names = {roi.ROINumber: roi.ROIName for roi in self.rtstruct.StructureSetROISequence}
        #print(f"Available ROIs: {list(roi_names.values())}")
        
        # Extract contour data
        for roi in self.rtstruct.ROIContourSequence:
            roi_number = roi.ReferencedROINumber
            roi_name = roi_names[roi_number]
            contours[roi_name] = []
            instance_iuds[roi_name] = []
            
            # Check if this ROI has contours
            if hasattr(roi, 'ContourSequence'):
                for contour in roi.ContourSequence:
                    instance_iud = roi.ContourSequence[0].ContourImageSequence[0].ReferencedSOPInstanceUID
                    instance_iuds[roi_name].append(instance_iud)
                    contour_data = contour.ContourData
                    points = np.array(contour_data).reshape(-1, 3)
                    contours[roi_name].append(points)
        
        return contours, instance_iuds
    
    def _filter_target_contours(self):
        """Filter contours to include only GTV, CTV, and PTV"""
        target_contours = {}
        target_instance_uids = {}
        
        for roi_name, roi_contours in self.contours.items():
            # Check if the ROI name contains GTV, CTV, or PTV
            if any(target in roi_name for target in ['GTV', 'CTV', 'PTV']):
                target_contours[roi_name] = roi_contours
                target_instance_uids[roi_name] = self.instance_iuds[roi_name]

        
        #print(f"Target ROIs: {list(target_contours.keys())}")
        return target_contours, target_instance_uids
    
    def _group_contours_by_slice(self):
        """Group contours by Z position (slice) and assign the correct SOP Instance UID."""
        # Find all unique Z positions
        all_z_positions = set()

        for roi_name, roi_contours in self.target_contours.items():
            for contour in roi_contours:
                # Get the Z position (assume all points in a contour have the same Z)
                z_pos = round(np.mean(contour[:, 2]), 1)  # Round to 1 decimal place
                all_z_positions.add(z_pos)

        # Sort Z positions
        sorted_z_positions = sorted(all_z_positions)
        if self.verbose:
            vis_pos = [float(z) for z in sorted_z_positions]
            print(f"Z positions: {vis_pos}")

        # For each Z position, collect contours
        for z_pos in sorted_z_positions:
            slice_contours = {
                'GTV': [],
                'CTV': [],
                'PTV': []
            }


            instance_uid = None
            for roi_name, roi_contours in self.target_contours.items():
                instance_uids = self.target_instance_uids[roi_name]
                for contour, _instance_uid in zip(roi_contours, instance_uids):
                    contour_z = round(np.mean(contour[:, 2]), 1)
                    mm_tolerance = 0.1
                    if abs(contour_z - z_pos) < mm_tolerance:  # Small tolerance for floating point comparison
                        # Determine target type
                        if 'GTV' in roi_name:
                            slice_contours['GTV'].append(contour)
                        elif 'CTV' in roi_name:
                            slice_contours['CTV'].append(contour)
                        elif 'PTV' in roi_name:
                            slice_contours['PTV'].append(contour)

                        # we are in same depth so we should be on same ct scan
                        instance_uid = _instance_uid
                        break
                        

            # RTSTRUCT UID
            rs_uid = self.rtstruct.SOPInstanceUID

            self.slices.append({
                'z_position': z_pos,
                'contours': slice_contours,
                'rs_uid': rs_uid,  # Store the correct UI for this slice
                "instance_uid": instance_uid
            })
    
    def _contour_to_mask(self, contour, img_size):
        """Convert contour points to a binary mask with specified bounds"""
        if len(contour) < 3:  # Need at least 3 points for a polygon
            return np.zeros(img_size, dtype=np.bool_)
        
        # Extract x and y coordinates
        x_points = contour[:, 0]
        y_points = contour[:, 1]
        
        # Define the bounds
        x_min, x_max = -300, 300
        y_min, y_max = -200, 400
        
        # Scale points to fit within the bounds
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Scale to image dimensions
        x_img = (x_points - x_min) / x_range * img_size[1]
        y_img = (y_points - y_min) / y_range * img_size[0]
        
        # Create polygon vertices
        vertices = np.column_stack((y_img, x_img))
        
        # Create mask from polygon
        try:
            mask = polygon2mask(img_size, vertices)
            # Flip the mask vertically before returning
            mask = np.flipud(mask)
            return mask  # polygon2mask already returns boolean array
        except:
            print(f"Error creating mask for contour with {len(contour)} points")
            return np.zeros(img_size, dtype=np.bool_)
    
    def _load_ct_image(self, uid):
        """Load and scale CT image based on UI"""
        ct_path = f"{DATASET_PATH}/CT.{uid}.dcm"
        try:
            ct_dicom = pydicom.dcmread(ct_path)
            ct_array = ct_dicom.pixel_array
            
            # Scale to 128x128
            ct_scaled = resize(ct_array, (128, 128), anti_aliasing=True)
            
            # Normalize to 0-1 range
            ct_normalized = (ct_scaled - ct_scaled.min()) / (ct_scaled.max() - ct_scaled.min())
            
            return ct_normalized
        except Exception as e:
            print(f"Error loading CT image: {e}")
            return np.zeros((128, 128), dtype=np.float32)
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        slice_data = self.slices[idx]
        instance_uid = slice_data["instance_uid"]
        
        # Initialize empty masks
        gtv_mask = np.zeros(self.img_size, dtype=np.bool_)
        ctv_mask = np.zeros(self.img_size, dtype=np.bool_)
        ptv_mask = np.zeros(self.img_size, dtype=np.bool_)
        
        # Fill masks based on contours
        for gtv_contour in slice_data['contours']['GTV']:
            gtv_mask |= self._contour_to_mask(gtv_contour, self.img_size)
        
        for ctv_contour in slice_data['contours']['CTV']:
            ctv_mask |= self._contour_to_mask(ctv_contour, self.img_size)
        
        for ptv_contour in slice_data['contours']['PTV']:
            msk = self._contour_to_mask(ptv_contour, self.img_size)
            ptv_mask |= msk
        
        # Convert to torch tensors
        gtv_tensor = torch.tensor(gtv_mask, dtype=torch.float32).unsqueeze(0)
        ctv_tensor = torch.tensor(ctv_mask, dtype=torch.float32).unsqueeze(0)
        ptv_tensor = torch.tensor(ptv_mask, dtype=torch.float32).unsqueeze(0)
        
        # Stack into a 3-channel tensor
        masks = torch.cat([gtv_tensor, ctv_tensor, ptv_tensor], dim=0)
        
        # Load and scale CT image using the slice-specific UI
        ct_array = self._load_ct_image(instance_uid)
        ct_tensor = torch.tensor(ct_array, dtype=torch.float32).unsqueeze(0)
        
        return {
            'masks': masks,  # Shape: [3, 128, 128]
            'ct': ct_tensor,  # Shape: [1, 128, 128]
            'z_position': slice_data['z_position'],
            'index': idx,
            'rs_uid': slice_data['rs_uid'],  # Include the UI in the returned dictionary
            "instance_uid": instance_uid
        }
    
    def visualize_item(self, idx):
        """Visualize a specific slice with its contours"""
        item = self.__getitem__(idx)
        masks = item['masks'].numpy()
        ct = item['ct'].numpy()[0]  # Remove channel dimension
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # Plot CT image
        axes[0].imshow(ct, cmap='gray')
        axes[0].set_title('CT')
        axes[0].axis('off')
        
        # Plot individual masks
        axes[1].imshow(masks[0], cmap='gray')
        axes[1].set_title('GTV')
        axes[1].axis('off')
        
        axes[2].imshow(masks[1], cmap='gray')
        axes[2].set_title('CTV')
        axes[2].axis('off')
        
        axes[3].imshow(masks[2], cmap='gray')
        axes[3].set_title('PTV')
        axes[3].axis('off')
        
        # Initialize combined image with zeros
        combined = np.zeros((*self.img_size, 3))

        # Define color mappings for each mask
        colors = {
            0: [1, 0.75, 1],  # GTV - Pink
            1: [1, 0.647, 0],  # CTV - Orange
            2: [1, 0, 0],  # PTV - Red
        }

        # Specify the desired order for applying the masks
        mask_order = [2, 1, 0]  # Mask 2 first, then mask 1, then mask 0

        # Apply masks in the specified order
        for i in mask_order:
            mask = masks[i]
            combined[..., 0] = np.where(mask == 1, colors[i][0], combined[..., 0])  # Red channel
            combined[..., 1] = np.where(mask == 1, colors[i][1], combined[..., 1])  # Green channel
            combined[..., 2] = np.where(mask == 1, colors[i][2], combined[..., 2])  # Blue channel

        # Display the combined mask
        axes[4].imshow(combined)
        axes[4].set_title('Combined (RGB)')
        axes[4].axis('off')
        
        plt.suptitle(f"Slice {idx} - Z position: {item['z_position']}\nUID: {item['instance_uid']}")
        plt.tight_layout()
        return fig
    
    def get_all_slices_info(self):
        """Return information about all slices"""
        info = []
        for i, slice_data in enumerate(self.slices):
            info.append({
                'index': i,
                'z_position': slice_data['z_position'],
                'num_GTV_contours': len(slice_data['contours']['GTV']),
                'num_CTV_contours': len(slice_data['contours']['CTV']),
                'num_PTV_contours': len(slice_data['contours']['PTV'])
            })
        return info


# Example usage
if __name__ == "__main__":
    # Path to the specific RTSTRUCT file
    rtstruct_path = f"{DATASET_PATH}/RS.1.2.246.352.221.46272062591570509005209218152822185346.dcm"
    
    # Create dataset
    dataset = RTStructSliceDataset(rtstruct_path, verbose= True)
    
    dataset[0]
    # # Print slice information
    # slice_info = dataset.get_all_slices_info()
    # for info in slice_info:
    #    print(f"Slice {info['index']}: Z={info['z_position']}, "
    #          f"GTV={info['num_GTV_contours']}, "
    #          f"CTV={info['num_CTV_contours']}, "
    #          f"PTV={info['num_PTV_contours']}")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Example of iterating through the dataloader
    for batch in dataloader:
        masks = batch['masks']
        z_positions = batch['z_position']
        #print(f"Batch shape: {masks.shape}, Z positions: {z_positions}")
        
    # Visualize a few slices
    for i in range(0, len(dataset), 5):
        fig = dataset.visualize_item(i)
        plt.show(block=True)  # This will display the figure
        #plt.close(fig)
