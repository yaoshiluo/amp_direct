import numpy as np

file1 = "G1_fight.npz"
file2 = "G1_fight_part002.npz"

data1 = np.load(file1)
data2 = np.load(file2)

print(f"Comparing NPZ files:\n - File 1: {file1}\n - File 2: {file2}\n")

for key in data1.files:
    if key not in data2.files:
        print(f"Key '{key}' is missing in {file2}")
        continue

    arr1 = data1[key]
    arr2 = data2[key]

    if arr1.shape != arr2.shape:
        print(f"Key '{key}' shape differs: {arr1.shape} vs {arr2.shape}")
        continue

    # Handle string comparison
    if arr1.dtype.kind in {'U', 'S'}:  # 'U' for unicode, 'S' for bytes
        if np.array_equal(arr1, arr2):
            print(f"Key '{key}' matches exactly (string array)")
        else:
            diff_indices = np.where(arr1 != arr2)
            print(f"Key '{key}' differs at {len(diff_indices[0])} locations (string array)")
        continue

    # Handle numerical comparison
    try:
        if np.allclose(arr1, arr2, rtol=1e-5, atol=1e-8):
            print(f"Key '{key}' matches (shape: {arr1.shape})")
        else:
            diff = np.abs(arr1 - arr2).mean()
            print(f"Key '{key}' differs (shape: {arr1.shape}), mean abs diff: {diff:.6f}")
    except Exception as e:
        print(f"Error comparing key '{key}': {e}")
