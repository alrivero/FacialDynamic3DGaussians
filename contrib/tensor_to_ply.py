import torch
import sys
import open3d as o3d

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script_name.py in_file out_file batch_index")
        sys.exit(1)
    
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    batch_idx = int(sys.argv[3])

    # Load the tensor
    data = torch.load(in_file)

    # Check tensor dimensions
    if len(data.shape) != 3 or data.shape[2] != 3:
        print("Input tensor should be of shape [B, N, 3].")
        sys.exit(1)

    # Extract the desired batch
    points = data[batch_idx].detach().cpu().numpy()

    # Create an open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save to PLY
    o3d.io.write_point_cloud(out_file, pcd)

    print(f"Saved {out_file} from batch index {batch_idx} of {in_file}.")
