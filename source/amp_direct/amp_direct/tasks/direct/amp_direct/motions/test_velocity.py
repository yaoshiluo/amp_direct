import numpy as np

file = "G1_fight_part002.npz"
data = np.load(file, allow_pickle=True)

lin_vel = data["body_linear_velocities"]  # shape: (T, B, 3)
ang_vel = data["body_angular_velocities"]  # shape: (T, B, 3)

print("Linear Velocity:")
print(f"  max: {lin_vel.max():.3f}")
print(f"  mean: {lin_vel.mean():.3f}")
print(f"  std: {lin_vel.std():.3f}")

print("Angular Velocity:")
print(f"  max: {ang_vel.max():.3f}")
print(f"  mean: {ang_vel.mean():.3f}")
print(f"  std: {ang_vel.std():.3f}")
