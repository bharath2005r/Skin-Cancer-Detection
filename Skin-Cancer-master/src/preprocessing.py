import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIGURATION
# -----------------------------
# Change these only if your structure is different
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'HAM10000_images')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'HAM10000_metadata.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'split_data')

# -----------------------------
# CREATE FOLDERS SAFELY
# -----------------------------
for folder in ['train/benign', 'train/malignant', 'test/benign', 'test/malignant']:
    full_path = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(full_path, exist_ok=True)

# -----------------------------
# READ METADATA
# -----------------------------
df = pd.read_csv(METADATA_PATH)

# Combine 7 classes into 2 groups
benign_classes = ['nv', 'bkl', 'df']
malignant_classes = ['mel', 'bcc', 'akiec', 'vasc']

df['label'] = df['dx'].apply(lambda x: 'benign' if x in benign_classes else 'malignant')

# -----------------------------
# TRAIN-TEST SPLIT (80/20)
# -----------------------------
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

def copy_images(df, subset):
    """Copy images to split_data/train or split_data/test"""
    for _, row in df.iterrows():
        src = os.path.join(IMAGE_DIR, f"{row['image_id']}.jpg")
        dst = os.path.join(OUTPUT_DIR, subset, row['label'], f"{row['image_id']}.jpg")
        try:
            shutil.copy(src, dst)
        except FileNotFoundError:
            print(f"⚠️ Skipped missing image: {src}")

print("📂 Copying training images...")
copy_images(train_df, 'train')

print("📂 Copying testing images...")
copy_images(test_df, 'test')

print("\n✅ Dataset successfully split and organized!")
print(f"Train: {len(train_df)} images")
print(f"Test: {len(test_df)} images")
print(f"Output directory: {OUTPUT_DIR}")
