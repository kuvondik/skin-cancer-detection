import os
import shutil
import pandas as pd

# === PATHS ===
base_dir = "../data/HAM10000"  # adjust to where your raw data is located
metadata_path = os.path.join(base_dir, "HAM10000_metadata.csv")
output_dir = "../data/HAM10000_custom"

# === DIAGNOSIS LABEL MAP ===
# dx column in metadata.csv → folder/class names
label_map = {
    "akiec": "AKIEC",  # Actinic keratoses / intraepithelial carcinoma
    "bcc": "BCC",      # Basal cell carcinoma
    "bkl": "BKL",      # Benign keratosis-like lesions
    "df": "DF",        # Dermatofibroma
    "mel": "MEL",      # Melanoma
    "nv": "NV",        # Melanocytic nevi
    "vasc": "VASC"     # Vascular lesions
}

# === CREATE FOLDERS ===
os.makedirs(output_dir, exist_ok=True)
for folder in label_map.values():
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# === LOAD METADATA ===
meta = pd.read_csv(metadata_path)
print(f"Loaded metadata: {meta.shape[0]} entries")

# === MOVE/COPY FILES ===
image_dirs = [
    os.path.join(base_dir, "HAM10000_images_part_1"),
    os.path.join(base_dir, "HAM10000_images_part_2")
]

missing, copied = 0, 0
for idx, row in meta.iterrows():
    file_name = row["image_id"] + ".jpg"
    label = label_map[row["dx"].lower()]
    src_path = None
    # Find which folder contains the image
    for d in image_dirs:
        candidate = os.path.join(d, file_name)
        if os.path.exists(candidate):
            src_path = candidate
            break
    if src_path is None:
        missing += 1
        continue
    dst_path = os.path.join(output_dir, label, file_name)
    shutil.copy2(src_path, dst_path)
    copied += 1

print(f"✅ Copied {copied} images into {output_dir}")
if missing:
    print(f"⚠️ Missing {missing} files (check your image parts).")