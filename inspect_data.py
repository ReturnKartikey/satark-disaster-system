import json
import os
import tifffile as tiff
import numpy as np

def inspect():
    s1_json = r'flood/SEN12FLOOD/S1list.json'
    s2_json = r'flood/SEN12FLOOD/S2list.json'
    
    with open(s1_json, 'r') as f:
        s1_data = json.load(f)
    with open(s2_json, 'r') as f:
        s2_data = json.load(f)
        
    print(f"S1 chunks: {len(s1_data)}")
    print(f"S2 chunks: {len(s2_data)}")
    
    # Get a sample
    for key in set(s1_data.keys()).intersection(s2_data.keys()):
        s1_item = s1_data[key]
        s2_item = s2_data[key]
        
        print(f"\nTile: {key}")
        print(f"S1 item keys: {s1_item.keys()}")
        print(f"S2 item keys: {s2_item.keys()}")
        print(f"S1 Flooding: {s1_item.get('FLOODING')}")
        print(f"S2 Flooding: {s2_item.get('FLOODING')}")
        
        # Check files in folder
        folder = f"flood/SEN12FLOOD/{key}"
        if os.path.exists(folder):
            print(f"Files in {folder}: {os.listdir(folder)}")
            
            # Try to read a tif
            tifs = [f for f in os.listdir(folder) if f.endswith('.tif')]
            for tif in tifs[:2]:
                path = os.path.join(folder, tif)
                img = tiff.imread(path)
                print(f"Read {tif}: shape={img.shape}, dtype={img.dtype}, max={img.max()}, min={img.min()}")
            break

if __name__ == '__main__':
    inspect()
