import os
import shutil
import random

# Paths
image_dir = "data/Images/Images"
label_dir = "data/YOLO_Annotations/YOLO_Annotations"
output_image_dir = "data/images"
output_label_dir = "data/labels"

# Create train/val dirs
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_image_dir, split), exist_ok=True)
    os.makedirs(os.path.join(output_label_dir, split), exist_ok=True)

# List all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
random.shuffle(image_files)

# Split ratio
val_split = 0.2
split_idx = int(len(image_files) * (1 - val_split))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Function to move pairs
def move_pairs(file_list, split):
    for filename in file_list:
        name, _ = os.path.splitext(filename)
        label_file = name + ".txt"

        src_img = os.path.join(image_dir, filename)
        src_lbl = os.path.join(label_dir, label_file)

        dst_img = os.path.join(output_image_dir, split, filename)
        dst_lbl = os.path.join(output_label_dir, split, label_file)

        if os.path.exists(src_lbl):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"Warning: No label for {filename}")

# Move files
move_pairs(train_files, "train")
move_pairs(val_files, "val")

print(f"✅ Dataset split: {len(train_files)} train / {len(val_files)} val")
