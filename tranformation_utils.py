import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np



def plot_images(planning_ct, cbct, title='Planning CT', title2='CBCT'):
    planning_ct_img = sitk.GetArrayFromImage(planning_ct)
    cbct_img = sitk.GetArrayFromImage(cbct)   
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(planning_ct_img, cmap='gray')
    plt.title(title)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cbct_img, cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_images_np(planning_ct, cbct, title='Planning CT', title2='CBCT'):
    planning_ct_img = planning_ct
    cbct_img = cbct   
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(planning_ct_img, cmap='gray')
    plt.title(title)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cbct_img, cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_difference(planning_ct, cbct_ct):
    planning_img = sitk.GetArrayFromImage(planning_ct)
    cbct_img = sitk.GetArrayFromImage(cbct_ct)
    diff = planning_img - cbct_img
    diff[diff < 0] = 0
    diff[diff > 0] = 255
    plt.imshow(diff, cmap='gray')
    plt.title('Difference Image')   
    plt.show()
    
def perform_rigid_registration_v2(fixed_image, moving_image, sampling_percentage=0.1, learning_rate=1.0, num_iterations=100):
    """
    Performs rigid registration using SimpleITK's ImageRegistrationMethod.

    Args:
        fixed_image (sitk.Image): The reference image.
        moving_image (sitk.Image): The image to align to the fixed image.
        sampling_percentage (float): Fraction of pixels used for metric evaluation (e.g., 0.1 for 10%).
        learning_rate (float): Step size for the optimizer.
        num_iterations (int): Maximum number of optimizer iterations.

    Returns:
        sitk.Transform: The calculated rigid transform, or None if registration fails.
    """
    if fixed_image.GetDimension() != moving_image.GetDimension():
        print("Error: Fixed and moving images must have the same dimension.")
        return None
    # --- Initial Transform ---
    # Initialize with a rigid transform centered using image geometry.
    is_3d = fixed_image.GetDimension() == 3
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform() if is_3d else sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # --- Registration Method Setup ---
    registration_method = sitk.ImageRegistrationMethod()

    # 1. Metric: Mattes Mutual Information
    # Good for multi-modality or mono-modality registration.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # Use a percentage of pixels for metric calculation for speed. Increase for stability.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(sampling_percentage)

    # 2. Interpolator: Linear interpolation is standard.
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 3. Optimizer: Gradient Descent
    # Often more robust than RegularStepGradientDescent.
    # Adjust learningRate and numberOfIterations as needed.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-6, # Stop if metric value change is small
        convergenceWindowSize=10      # Over how many iterations to check convergence
    )

    # 4. Optimizer Scaling: Crucial for balancing rotation and translation steps.
    # Scales parameters based on the expected physical shift caused by a unit change.
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # 5. Initial Transform: Set the starting point for the optimizer.
    registration_method.SetInitialTransform(initial_transform)

    try:
        final_transform = registration_method.Execute(
            sitk.Cast(fixed_image, sitk.sitkFloat32), # Ensure float type
            sitk.Cast(moving_image, sitk.sitkFloat32)
        )

        #print(final_transform)
        

        return final_transform

    except Exception as e:
        print(f"Error during registration execution: {e}")
        return None

def perform_bspline_registration(fixed_image, moving_image, grid_physical_spacing=[50.0, 50.0, 50.0], 
                                 number_of_iterations=200):
    # 1. Determine the mesh size based on physical spacing.
    #    This defines the resolution of the BSpline grid.
    fixed_size = fixed_image.GetSize()
    fixed_spacing = fixed_image.GetSpacing()
    mesh_size = [int(np.round((fixed_size[i] * fixed_spacing[i]) / grid_physical_spacing[i])) for i in range(fixed_image.GetDimension())]
    
    # Initialize the BSpline transform.
    initial_transform = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
    
    # 2. Configure the image registration method.
    registration_method = sitk.ImageRegistrationMethod()
    
    # Use Mattes Mutual Information for the metric (suitable for multimodal data)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # Interpolate using linear interpolation
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Use LBFGSB optimizer for this non-rigid optimization.
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                                             numberOfIterations=number_of_iterations,
                                             maximumNumberOfCorrections=5,
                                             maximumNumberOfFunctionEvaluations=1000,
                                             costFunctionConvergenceFactor=1e+7)
    
    # Set the initial BSpline transform.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # Multi-resolution framework for a robust and faster convergence.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 2])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # 3. Execute registration.
    final_transform = registration_method.Execute(fixed_image, moving_image)
    #print("Final BSpline registration metric value: {0}".format(registration_method.GetMetricValue()))
    #print("Optimizer's stopping condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
    
    return final_transform

def resample_with_transform(moving_image, fixed_image, transform):
    # Resample the moving image onto the fixed image grid using the provided transform.
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    # Make sure the orientation matches the fixed image explicitly.
    resampler.SetOutputDirection(fixed_image.GetDirection())
    return resampler.Execute(moving_image)

# Assume 'global_transform' is the 2D result from rigid registration or None
# Assume 'plot_images' function exists and handles 2D images

def perform_bspline_deformable_registration_2d(fixed_image, moving_image,
                                             grid_physical_spacing=[50.0, 50.0], # Provide 2 values for 2D!
                                             sampling_percentage=0.1,
                                             num_iterations=50,
                                             spline_order=3):
    """
    Performs 2D B-Spline Deformable Registration.
    Corrected version: Uses BSplineTransformInitializer with positional image argument.

    Args:
        fixed_image (sitk.Image): The 2D reference image.
        moving_image (sitk.Image): The 2D image to deformably align.
        initial_transform (sitk.Transform, optional): 2D global transform to initialize with.
        grid_physical_spacing (list[float]): Spacing (mm) between B-Spline control points (must have len 2).
        sampling_percentage (float): Fraction of pixels for metric evaluation.
        num_iterations (int): Max iterations *per multi-resolution level*.
        spline_order (int): Order of the B-spline (e.g., 3 for cubic).

    Returns:
        sitk.Transform: The calculated final (possibly composite) 2D transform, or None if fails.
    """


    # --- Initialize B-Spline Transform using BSplineTransformInitializer ---
    mesh_size = [int(np.ceil(sz * spc / gspc)) + spline_order -1
                 for sz, spc, gspc in zip(fixed_image.GetSize(),
                                          fixed_image.GetSpacing(),
                                          grid_physical_spacing)]

    bspline_transform_part = sitk.BSplineTransformInitializer(
        fixed_image, # Use as positional argument 1
        transformDomainMeshSize=mesh_size,
        order=spline_order
    )


    moving_image_globally_aligned = moving_image

    # --- Registration Method Setup ---
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(sampling_percentage)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5, numberOfIterations=num_iterations,
        maximumNumberOfCorrections=5,
        costFunctionConvergenceFactor=1e+7 )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Initial Transform (set the B-spline part *for optimization*)
    registration_method.SetInitialTransform(bspline_transform_part, inPlace=False)

    # Multi-Resolution Framework (adjust shrink factors if needed for 2D)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


    registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image_globally_aligned, sitk.sitkFloat32)
    )
    print(f"Final metric value: {registration_method.GetMetricValue():.6f}")
    return bspline_transform_part