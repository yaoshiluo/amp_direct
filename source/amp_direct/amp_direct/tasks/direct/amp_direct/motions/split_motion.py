import numpy as np
import os
import argparse

def split_motion(file_path, segment_length=180, output_dir=None):
    data = np.load(file_path, allow_pickle=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    motion_keys = [
        "body_positions", "body_rotations", "body_linear_velocities",
        "body_angular_velocities", "dof_positions", "dof_velocities"
    ]

    fps = data["fps"].item() if "fps" in data else 60
    dof_names = data["dof_names"] if "dof_names" in data else None
    body_names = data["body_names"] if "body_names" in data else None

    total_len = data[motion_keys[0]].shape[0]
    for key in motion_keys:
        if key not in data:
            raise KeyError(f"Missing key in input data: {key}")
        if data[key].shape[0] != total_len:
            raise ValueError(f"Inconsistent length for key: {key}")

    output_dir = output_dir or os.path.join(os.path.dirname(file_path), base_name + "_split")
    os.makedirs(output_dir, exist_ok=True)

    num_segments = total_len // segment_length
    print(f"Splitting into {num_segments} segments of {segment_length} frames each")

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length

        segment_data = {
            "fps": fps,
            "dof_names": dof_names,
            "body_names": body_names,
        }

        for key in motion_keys:
            segment_data[key] = data[key][start:end]

        save_path = os.path.join(output_dir, f"{base_name}_part{i:03d}.npz")
        np.savez_compressed(save_path, **segment_data)
        print(f"Saved: {save_path}")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to .npz motion file")
    parser.add_argument("--length", type=int, default=180, help="Frames per segment")
    args = parser.parse_args()

    split_motion(args.file, segment_length=args.length)
