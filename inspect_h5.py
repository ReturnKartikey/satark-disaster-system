import h5py

file_path = r"C:\Users\Karti\Desktop\DESK\minor_project\Cyclone_Images.h5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

try:
    with h5py.File(file_path, 'r') as f:
        img_ds = f['Images']
        import numpy as np
        channel_4 = img_ds[0, :, :, 3]
        print("4th channel unique values for first image:", np.unique(channel_4))
        channel_4_all = img_ds[:100, :, :, 3]
        print("4th channel unique values for first 100 images:", np.unique(channel_4_all))
except Exception as e:
    print(f"Error reading {file_path}: {e}")
