import torch
import numpy as np
from pycpd import RigidRegistration
import sys

def align_point_clouds(source_cloud, target_cloud):
    """ Aligns two point clouds using the CPD algorithm. """
    reg = RigidRegistration(X=source_cloud, Y=target_cloud)
    result = reg.register()
    import pdb; pdb.set_trace()
    return result  # Transposed to match the input shape

def scale_prism(min_values, max_values, scale_factor):
    """ Scale the min and max values of a prism by a given scale factor. """
    center = (min_values + max_values) / 2
    half_size_scaled = (max_values - min_values) * scale_factor / 2
    new_min = center - half_size_scaled
    new_max = center + half_size_scaled
    return new_min, new_max

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py source.pt target.pt")
        sys.exit(1)

    source_file = sys.argv[1]
    target_file = sys.argv[2]

    # Load point clouds from files
    source_cloud = torch.load(source_file).detach().cpu().numpy()
    target_cloud = torch.load(target_file).detach().cpu().numpy()

    # Define the local neighborhood of source in the target point cloud and discard other target_cloud
    max_x = source_cloud[:, 0].max()
    max_y = source_cloud[:, 1].max()
    max_z = source_cloud[:, 2].max()
    min_x = source_cloud[:, 0].min()
    min_y = source_cloud[:, 1].min()
    min_z = source_cloud[:, 2].min()

    min_values = np.array([min_x, min_y, min_z * 1.1])
    max_values = np.array([max_x, max_y, max_z * 1.1])

    scale_factor = 3
    scaled_min, scaled_max = scale_prism(min_values, max_values, scale_factor)

    mask_x = (target_cloud[:, 0] >= scaled_min[0]) & (target_cloud[:, 0] <= scaled_max[0])
    mask_y = (target_cloud[:, 1] >= scaled_min[1]) & (target_cloud[:, 1] <= scaled_max[1])
    mask_z = (target_cloud[:, 2] >= scaled_min[2]) & (target_cloud[:, 2] <= scaled_max[2])
    combined_mask = mask_x & mask_y & mask_z

    target_cloud = target_cloud[combined_mask]

    # Now, half of local neighborhood for the source pointcloud (face)
    min_values = np.array([min_x, min_y, min_z])
    max_values = np.array([max_x, max_y, (min_z + max_z)/2])

    scale_factor = 1
    scaled_min, scaled_max = scale_prism(min_values, max_values, scale_factor)

    mask_x = (source_cloud[:, 0] >= scaled_min[0]) & (source_cloud[:, 0] <= scaled_max[0])
    mask_y = (source_cloud[:, 1] >= scaled_min[1]) & (source_cloud[:, 1] <= scaled_max[1])
    mask_z = (source_cloud[:, 2] >= scaled_min[2]) & (source_cloud[:, 2] <= scaled_max[2])
    combined_mask = mask_x & mask_y & mask_z

    source_cloud = source_cloud[combined_mask]
    
    import pdb; pdb.set_trace()

    # Align the point clouds
    aligned_cloud, transform_params  = align_point_clouds(source_cloud, target_cloud)
    aligned_cloud = np.dot(aligned_cloud, transform_params['R'].T) + transform_params['t'].T

    # Optionally, save the aligned point cloud to a file
    torch.save(torch.from_numpy(aligned_cloud), 'aligned_cloud.pt')

    print("Alignment complete. Aligned point cloud saved as 'aligned_cloud.pt'.")