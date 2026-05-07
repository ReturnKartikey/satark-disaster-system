import os
import json

labels_dir = r"c:\Users\Karti\Desktop\minor_project\flood2\sen12floods_s2_labels\sen12floods_s2_labels"
source_dir = r"c:\Users\Karti\Desktop\minor_project\flood2\sen12floods_s2_source\sen12floods_s2_source"

valid_data = []
flooded_count = 0
non_flooded_count = 0

for label_folder in os.listdir(labels_dir):
    if not label_folder.startswith('sen12floods_s2_labels_'):
        continue
    
    suffix = label_folder[len('sen12floods_s2_labels_'):]
    source_folder = f"sen12floods_s2_source_{suffix}"
    
    label_path = os.path.join(labels_dir, label_folder, 'labels.geojson')
    source_folder_path = os.path.join(source_dir, source_folder)
    
    if os.path.exists(label_path) and os.path.exists(source_folder_path):
        with open(label_path, 'r') as f:
            data = json.load(f)
            
        is_flooded = data.get('properties', {}).get('FLOODING', False)
        
        # Check bands
        files = os.listdir(source_folder_path)
        if any('B04.tif' in f for f in files) and any('B03.tif' in f for f in files) and any('B02.tif' in f for f in files):
            valid_data.append((source_folder_path, is_flooded))
            if is_flooded:
                flooded_count += 1
            else:
                non_flooded_count += 1

print(f"Total valid tiles: {len(valid_data)}")
print(f"Flooded: {flooded_count}")
print(f"Non-Flooded: {non_flooded_count}")
