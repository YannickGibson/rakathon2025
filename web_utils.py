import SimpleITK as sitk
import numpy as np
import cv2
from tranformation_utils import *

def transform_masks(masks, fixed_image, global_transform, final_bspline_transform ):
    transformed_masks = []
    for mask in masks:
        # Convert to SimpleITK image
        mask_sitk = sitk.GetImageFromArray(mask)
        moved_mask = sitk.Resample(
            mask_sitk,
            fixed_image,
            global_transform,
            sitk.sitkLinear,
            0,
            mask_sitk.GetPixelID()
        )
        # Resample the mask using the transform
        deformed_mask = resample_with_transform(moved_mask, fixed_image, final_bspline_transform)

        transformed_masks.append(sitk.GetArrayFromImage(deformed_mask))
    return np.array(transformed_masks)

def draw_contours(img, masks):
    ct_img_colored = np.stack([img, img, img], axis=-1)
    colors = ['orange', 'red', 'blue']
    colors_rgb = {'orange': [1.0, 0.65, 0], 'red': [1.0, 0, 0], 'blue': [0, 0, 1.0]}
    
    for i in range(3):
        mask_layer = masks[i]
        # Convert to uint8 for findContours
        mask_uint8 = (mask_layer * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours in the corresponding color
        color = colors_rgb[colors[i]]
        for contour in contours:
            cv2.drawContours(ct_img_colored, [contour], -1, color, 2)
    return ct_img_colored





# Convert tensors to NumPy arrays
# ct_img1 = ct_batch1[i, 0].detach().cpu().numpy()  # [H, W]
# ct_img2 = ct_batch2[i, 0].detach().cpu().numpy()
# z_pos = z_positions[i].item() if hasattr(z_positions[i], "item") else z_positions[i]
# masks = batch_masks[i].detach().cpu().numpy()
# masks2 = batch_mask2[i].detach().cpu().numpy()


def transform_ct(ct_img1, masks, ct_img2, masks2, z_pos, y_cut=False, plot=False):


    # Convert to SimpleITK images
    planning_ct_img = sitk.GetImageFromArray(ct_img1)
    cbct_img = sitk.GetImageFromArray(ct_img2)

    # Apply smoothing
    smoothed_planning = sitk.SmoothingRecursiveGaussian(planning_ct_img, sigma=2.0)
    smoothed_cbct = sitk.SmoothingRecursiveGaussian(cbct_img, sigma=2.0)

    planning_with_contours = draw_contours(sitk.GetArrayFromImage(smoothed_planning), masks)
    cbct_with_contours = draw_contours(sitk.GetArrayFromImage(smoothed_cbct), masks2)
    plot_images_np(planning_with_contours, cbct_with_contours, title=f"Planning CT with contours - {z_pos}", title2=f"CBCT - {z_pos}")

    # Convert to SimpleITK image
    # Perform rigid registration
    global_transform = perform_rigid_registration_v2(smoothed_cbct, smoothed_planning)

    # Align the planning CT to the CBCT using the global transform
    moved_planning_ct = sitk.Resample(
    smoothed_planning,
    smoothed_cbct,
    global_transform,
    sitk.sitkLinear,
    0,
    smoothed_planning.GetPixelID()
    )


    # break
    fixed_image = smoothed_cbct   # e.g., CBCT (fixed)
    moving_image = moved_planning_ct   # e.g., Planning CT (moving)

    # Perform deformable registration.
    final_bspline_transform = perform_bspline_registration(fixed_image, moving_image, grid_physical_spacing=[50.0, 50.0, 50.0], number_of_iterations=50)

    # Apply the BSpline transform to get the deformed (fused) image.
    deformed_planned_ct = resample_with_transform(moving_image, fixed_image, final_bspline_transform)
    transformed_masks = transform_masks(masks, fixed_image, global_transform, final_bspline_transform)
    
    deformed_planned_ct = sitk.GetArrayFromImage(deformed_planned_ct)
    deformed_with_contours = draw_contours(deformed_planned_ct, transformed_masks)

    if y_cut:
        size = ct_img1.shape[-1]
        y_cutoff = 500 * int(size / 128)
        fixed_image_np = sitk.GetArrayFromImage(fixed_image)
        deformed_with_contours[y_cutoff:, :] = (0, 0, 0)
        fixed_image_np[y_cutoff:, :] = 0

    if plot:
        plot_images_np(deformed_with_contours, cbct_with_contours, title=f"Deformed Planning CT with contours - {z_pos}", title2=f"CBCT - {z_pos}")

    return deformed_planned_ct, transformed_masks