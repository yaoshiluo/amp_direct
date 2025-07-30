import numpy as np

def clip_angular_velocity():
    input_path = "G1_dance1.npz"
    output_path = "G1_dance1_clipped.npz"
    clip_value = 20.0

    data = np.load(input_path, allow_pickle=True)
    data_dict = dict(data)

    if "body_angular_velocities" not in data_dict:
        print("Error: 'body_angular_velocities' key not found.")
        return

    ang_vel = data_dict["body_angular_velocities"]
    print(f"Original angular velocity stats:")
    print(f"  max: {np.max(ang_vel):.3f}")
    print(f"  min: {np.min(ang_vel):.3f}")
    print(f"  mean: {np.mean(ang_vel):.3f}")
    print(f"  std: {np.std(ang_vel):.3f}")

    # Clip to [-30, 30]
    clipped_ang_vel = np.clip(ang_vel, -clip_value, clip_value)
    data_dict["body_angular_velocities"] = clipped_ang_vel

    # Save new file
    np.savez_compressed(output_path, **data_dict)
    print(f"\nClipped file saved as: {output_path}")
    print(f"New max angular velocity: {np.max(clipped_ang_vel):.3f}")

if __name__ == "__main__":
    clip_angular_velocity()
