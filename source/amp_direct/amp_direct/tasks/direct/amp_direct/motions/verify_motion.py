import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to .npz file")
    args = parser.parse_args()

    data = np.load(args.file, allow_pickle=True)
    print(f"File: {args.file}")
    print(f"Size: {round(data.files.__sizeof__()/1024, 2)} KB")
    print(f"{'-'*50}")

    for key in data.files:
        val = data[key]
        print(f"{key}:")
        print(f"  Type: {type(val)}")
        print(f"  Dtype: {val.dtype}")
        print(f"  Shape: {val.shape}")
        if key == "dof_names":
            print("  Joint names:")
            for i, name in enumerate(val):
                print(f"    {i+1}. {name}")
        print()

if __name__ == "__main__":
    main()
